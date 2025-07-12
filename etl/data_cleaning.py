import pandas as pd
import json
import io
import os
import logging
import re
import chardet
from groq import Groq
from typing import Dict, Tuple
from pathlib import Path
from Prompts.preprocess_prompt import create_simplified_cleaning_prompt
from logger import logger

class DataProcessor:
    def __init__(self, session_id: str, model: str = "llama3-70b-8192"):
        self.session_id = session_id
        self.db_path = Path(f"uploads/{session_id}/my_duckdb.duckdb").as_posix()
        self.model = model
        self.logger = logging.getLogger(__name__)
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.logger.setLevel(logging.INFO)
        os.makedirs(Path(f"uploads/{session_id}").as_posix(), exist_ok=True)

    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                self.logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
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
                    f.read(1000)
                self.logger.info(f"Successfully opened file with encoding: {encoding}")
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
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
                on_bad_lines='skip'
            )
            self.logger.info(f"Successfully loaded CSV with encoding {encoding}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load CSV with encoding {encoding}: {str(e)}")
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    na_values=['NA', 'N/A', '-', 'TBD', ''],
                    keep_default_na=False,
                    on_bad_lines='skip',
                    quoting=1,
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
        cleaned_df = df.copy()
        fiscal_year_cols = []
        for col in df.columns:
            if any(pattern in str(col).lower() for pattern in ['fy', 'year', '20', '19']):
                fiscal_year_cols.append(col)
        if not fiscal_year_cols:
            cleaned_df['fiscal_year'] = [f"FY_{2020 + i}" for i in range(len(df))]
        else:
            fy_col = fiscal_year_cols[0]
            cleaned_df['fiscal_year'] = df[fy_col].astype(str)
            if fy_col != 'fiscal_year':
                cleaned_df = cleaned_df.drop(columns=[fy_col])
        cleaned_df.columns = [self.clean_column_name(col) for col in cleaned_df.columns]
        cols = ['fiscal_year'] + [col for col in cleaned_df.columns if col != 'fiscal_year']
        cleaned_df = cleaned_df[cols]
        for col in cleaned_df.columns:
            if col != 'fiscal_year':
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        schema = {
            "columns": {
                "fiscal_year": "VARCHAR(10) PRIMARY KEY"
            }
        }
        for col in cleaned_df.columns:
            if col != 'fiscal_year':
                schema["columns"][col] = "DECIMAL(15,2)"
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
        clean_name = re.sub(r'[^\w\s]', '', str(col_name))
        clean_name = re.sub(r'\s+', '_', clean_name.strip().lower())
        return clean_name

    def preprocess_with_llm(self, df: pd.DataFrame, table_name: str) -> Tuple[Dict, pd.DataFrame]:
        self.validate_dataframe(df, "LLM preprocessing")
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            sample_csv = csv_buffer.getvalue()
            if len(sample_csv) > 20000:
                sample_csv = sample_csv[:20000]
                self.logger.warning(f"CSV truncated to 20000 characters to fit model limits")
            prompt = create_simplified_cleaning_prompt(sample_csv)
            models_to_try = [self.model]
            for model_name in models_to_try:
                try:
                    self.logger.info(f"Attempting LLM processing with model: {model_name}")
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a data analyst. Return only valid JSON with cleaned_data, schema, and data_quality_report fields."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=3000,
                        temperature=0.1,
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
                    try:
                        result = json.loads(response_text)
                        self.logger.info(f"Successfully parsed JSON from {model_name}")
                        break
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON decode error with {model_name}: {str(e)}")
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
                self.logger.warning("All LLM models failed, using fallback processing")
                return self.fallback_processing(df, table_name)
            if not self.validate_llm_result(result):
                self.logger.warning("LLM result validation failed, using fallback")
                return self.fallback_processing(df, table_name)
            cleaned_df = pd.DataFrame(result['cleaned_data']['data'], 
                                    columns=result['cleaned_data']['columns'])
            if 'fiscal_year' not in cleaned_df.columns:
                cleaned_df['fiscal_year'] = [f"FY_{2020 + i}" for i in range(len(cleaned_df))]
                result['cleaned_data']['columns'] = ['fiscal_year'] + result['cleaned_data']['columns']
                result['schema']['columns']['fiscal_year'] = "VARCHAR(10) PRIMARY KEY"
            cleaned_df['fiscal_year'] = cleaned_df['fiscal_year'].astype(str)
            for col in cleaned_df.columns:
                if col != 'fiscal_year':
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
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
                    # Provide default schema if missing
                    if field == 'schema':
                        result['schema'] = {
                            "columns": {col: "DECIMAL(15,2)" for col in result['cleaned_data']['columns']}
                        }
                        result['schema']['columns']['fiscal_year'] = "VARCHAR(10) PRIMARY KEY"
                    else:
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