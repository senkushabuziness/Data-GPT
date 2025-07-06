import pandas as pd
import json
import io
import os
import logging
import re
import chardet
from groq import Groq
from typing import Dict, Tuple
from Prompts.preprocess_prompt import create_simplified_cleaning_prompt, sanitize_csv

class DataProcessor:
    def __init__(self, session_id: str, model: str = "qwen/qwen3-32b"):
        self.session_id = session_id
        self.db_path = f"/kaggle/working/uploads/{session_id}/my_duckdb.duckdb"
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        os.makedirs(f"/kaggle/working/uploads/{session_id}", exist_ok=True)

    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                self.logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                
                # Fallback to common encodings if confidence is low
                if confidence < 0.7:
                    self.logger.warning(f"Low confidence in encoding detection, trying common encodings")
                    return self.try_common_encodings(file_path)
                
                return encoding
        except Exception as e:
            self.logger.error(f"Error detecting encoding: {str(e)}")
            return self.try_common_encodings(file_path)

    def try_common_encodings(self, file_path: str) -> str:
        """Try common encodings to find one that works"""
        common_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        for encoding in common_encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # Try to read first 1000 characters
                self.logger.info(f"Successfully opened file with encoding: {encoding}")
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # If all fail, default to latin-1 which can handle any byte
        self.logger.warning("All encoding attempts failed, defaulting to latin-1")
        return 'latin-1'

    def load_csv_with_encoding(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with proper encoding detection"""
        encoding = self.detect_encoding(file_path)
        
        try:
            df = pd.read_csv(
                file_path, 
                encoding=encoding,
                na_values=['NA', 'N/A', '-', 'TBD', ''], 
                keep_default_na=False,
                on_bad_lines='skip'  # Skip malformed lines
            )
            self.logger.info(f"Successfully loaded CSV with encoding {encoding}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load CSV with encoding {encoding}: {str(e)}")
            
            # Try with different parameters
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    na_values=['NA', 'N/A', '-', 'TBD', ''],
                    keep_default_na=False,
                    on_bad_lines='skip',
                    quoting=1,  # QUOTE_ALL
                    escapechar='\\'
                )
                self.logger.info(f"Successfully loaded CSV with fallback parameters")
                return df
            except Exception as e2:
                self.logger.error(f"Fallback CSV loading also failed: {str(e2)}")
                raise ValueError(f"Could not load CSV file: {str(e2)}")

    def validate_dataframe(self, df: pd.DataFrame, context: str = "") -> bool:
        prefix = f"{context}: " if context else ""
        if df is None or df.empty or len(df.columns) == 0:
            raise ValueError(f"{prefix}Invalid DataFrame: None={df is None}, empty={df.empty}, columns={len(df.columns)}")
        csv_size = len(df.to_csv(index=False).encode('utf-8'))
        if csv_size > 100000:
            raise ValueError(f"{prefix}Input CSV too large: {csv_size} bytes")
        self.logger.debug(f"{prefix}DataFrame validated: {len(df)} rows, {len(df.columns)} columns")
        return True



    def fallback_processing(self, df: pd.DataFrame, table_name: str) -> Tuple[Dict, pd.DataFrame]:
        """Fallback processing when LLM fails"""
        self.logger.warning("Using fallback processing due to LLM failure")
        
        # Simple heuristic processing
        cleaned_df = df.copy()
        
        # Try to identify fiscal year columns
        fiscal_year_cols = []
        for col in df.columns:
            if any(pattern in str(col).lower() for pattern in ['fy', 'year', '20', '19']):
                fiscal_year_cols.append(col)
        
        if not fiscal_year_cols:
            # If no fiscal year columns found, use index as fiscal year
            cleaned_df['fiscal_year'] = [f"FY_{2020 + i}" for i in range(len(df))]
        else:
            # Use the first fiscal year column
            fy_col = fiscal_year_cols[0]
            cleaned_df['fiscal_year'] = df[fy_col].astype(str)
            if fy_col != 'fiscal_year':
                cleaned_df = cleaned_df.drop(columns=[fy_col])
        
        # Clean column names
        cleaned_df.columns = [self.clean_column_name(col) for col in cleaned_df.columns]
        
        # Ensure fiscal_year is first column
        cols = ['fiscal_year'] + [col for col in cleaned_df.columns if col != 'fiscal_year']
        cleaned_df = cleaned_df[cols]
        
        # Clean numeric data
        for col in cleaned_df.columns:
            if col != 'fiscal_year':
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Create schema
        schema = {
            "columns": {
                "fiscal_year": "VARCHAR(10) PRIMARY KEY"
            }
        }
        
        for col in cleaned_df.columns:
            if col != 'fiscal_year':
                schema["columns"][col] = "DECIMAL(15,2)"
        
        # Create result structure
        result = {
            "cleaned_data": {
                "columns": list(cleaned_df.columns),
                "data": cleaned_df.values.tolist()
            },
            "schema": schema,
            "data_quality_report": {
                "total_rows": len(cleaned_df),
                "total_columns": len(cleaned_df.columns),
                "issues": ["Used fallback processing due to LLM failure"]
            }
        }
        
        return result, cleaned_df

    def clean_column_name(self, col_name: str) -> str:
        """Convert column name to snake_case"""
        # Remove special characters and convert to lowercase
        clean_name = re.sub(r'[^\w\s]', '', str(col_name))
        clean_name = re.sub(r'\s+', '_', clean_name.strip().lower())
        return clean_name

    def preprocess_with_llm(self, df: pd.DataFrame, table_name: str) -> Tuple[Dict, pd.DataFrame]:
        self.validate_dataframe(df, "LLM preprocessing")
        
        try:
            # Prepare CSV sample
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            sample_csv = csv_buffer.getvalue()
            
            # Limit size to avoid token issues
            if len(sample_csv) > 20000:  # Reduced from 50000
                sample_csv = sample_csv[:20000]
                self.logger.warning(f"CSV truncated to 20000 characters to fit model limits")
            
            prompt = self.create_simplified_cleaning_prompt(sample_csv, table_name)
            
            # Try with different models if the first one fails
            models_to_try = [self.model, "llama3-8b-8192", "mixtral-8x7b-32768"]
            
            for model_name in models_to_try:
                try:
                    self.logger.info(f"Attempting LLM processing with model: {model_name}")
                    
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a data analyst. Return only valid JSON, no additional text."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=3000,  # Reduced to avoid issues
                        temperature=0.1,  # Lower temperature for more consistent output
                        top_p=0.9,
                        response_format={"type": "json_object"}
                    )
                    
                    if not response.choices:
                        self.logger.error(f"No choices returned from {model_name}")
                        continue
                    
                    response_text = response.choices[0].message.content
                    if not response_text:
                        self.logger.error(f"Empty response from {model_name}")
                        continue
                    
                    # Try to parse JSON
                    try:
                        result = json.loads(response_text)
                        self.logger.info(f"Successfully parsed JSON from {model_name}")
                        break
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON decode error with {model_name}: {str(e)}")
                        # Try to extract JSON from response
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            try:
                                result = json.loads(json_match.group())
                                self.logger.info(f"Successfully extracted JSON from {model_name}")
                                break
                            except json.JSONDecodeError:
                                continue
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Error with model {model_name}: {str(e)}")
                    continue
            else:
                # If all models fail, use fallback
                self.logger.warning("All LLM models failed, using fallback processing")
                return self.fallback_processing(df, table_name)
            
            # Validate and clean the result
            if not self.validate_llm_result(result):
                self.logger.warning("LLM result validation failed, using fallback")
                return self.fallback_processing(df, table_name)
            
            # Create DataFrame from cleaned data
            cleaned_df = pd.DataFrame(result['cleaned_data']['data'], 
                                    columns=result['cleaned_data']['columns'])
            
            # Ensure fiscal_year column exists and is string
            if 'fiscal_year' not in cleaned_df.columns:
                cleaned_df['fiscal_year'] = [f"FY_{2020 + i}" for i in range(len(cleaned_df))]
                result['cleaned_data']['columns'] = ['fiscal_year'] + result['cleaned_data']['columns']
                result['schema']['columns']['fiscal_year'] = "VARCHAR(10) PRIMARY KEY"
            
            cleaned_df['fiscal_year'] = cleaned_df['fiscal_year'].astype(str)
            
            # Convert other columns to numeric
            for col in cleaned_df.columns:
                if col != 'fiscal_year':
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            
            # Remove duplicates
            cleaned_df = cleaned_df.drop_duplicates(subset=['fiscal_year'])
            
            self.logger.info(f"LLM processing successful. Shape: {cleaned_df.shape}")
            return result, cleaned_df
            
        except Exception as e:
            self.logger.error(f"LLM processing failed: {str(e)}")
            return self.fallback_processing(df, table_name)

    def validate_llm_result(self, result: Dict) -> bool:
        """Validate LLM result structure"""
        try:
            required_fields = ['cleaned_data', 'schema', 'data_quality_report']
            for field in required_fields:
                if field not in result:
                    self.logger.error(f"Missing required field: {field}")
                    return False
            
            cleaned_data = result['cleaned_data']
            if not cleaned_data.get('columns') or not cleaned_data.get('data'):
                self.logger.error("cleaned_data missing columns or data")
                return False
            
            if not isinstance(cleaned_data['columns'], list):
                self.logger.error("cleaned_data columns is not a list")
                return False
            
            if not isinstance(cleaned_data['data'], list):
                self.logger.error("cleaned_data data is not a list")
                return False
            
            # Check if data rows match column count
            if cleaned_data['data']:
                expected_cols = len(cleaned_data['columns'])
                for i, row in enumerate(cleaned_data['data']):
                    if len(row) != expected_cols:
                        self.logger.error(f"Row {i} has {len(row)} columns, expected {expected_cols}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Result validation error: {str(e)}")
            return False