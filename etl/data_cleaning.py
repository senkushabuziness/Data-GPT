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
import duckdb
import threading
import numpy as np
from google.cloud import storage
from google.api_core.exceptions import GoogleAPIError

class DataProcessor:
    def __init__(self, session_id: str, user_id: str, model: str = "qwen/qwen3-32B", db_base_path: str = "db", gcs_enabled: bool = False):
        self.session_id = session_id
        self.user_id = user_id
        self.db_base_path = db_base_path
        self.db_path = os.path.join(db_base_path, "master.duckdb")
        self.model = model
        self.lock = threading.Lock()
        self.gcs_enabled = gcs_enabled
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        self.client = Groq(api_key=groq_api_key)
        
        self.gcs_client = None
        self.gcs_bucket = None
        if self.gcs_enabled:
            try:
                self.gcs_bucket_name = os.environ.get("BUCKET_NAME")
                if not self.gcs_bucket_name:
                    self.logger.warning("GCS_BUCKET_NAME not set, disabling GCS uploads")
                    self.gcs_enabled = False
                else:
                    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    if not credentials_path or not os.path.exists(credentials_path):
                        self.logger.warning(f"GOOGLE_APPLICATION_CREDENTIALS not set or invalid ({credentials_path}), disabling GCS uploads")
                        self.gcs_enabled = False
                    else:
                        self.gcs_client = storage.Client()
                        self.gcs_bucket = self.gcs_client.bucket(self.gcs_bucket_name)
                        self.logger.info(f"GCS client initialized for bucket: {self.gcs_bucket_name}")
            except GoogleAPIError as e:
                self.logger.error(f"Failed to initialize GCS client: {str(e)}")
                self.gcs_enabled = False
            except Exception as e:
                self.logger.error(f"Unexpected error initializing GCS client: {str(e)}")
                self.gcs_enabled = False
        
        self._init_duckdb()
        
    def _init_duckdb(self):
        try:
            with self.lock:
                os.makedirs(self.db_base_path, exist_ok=True)
                conn = duckdb.connect(self.db_path)
                
                try:
                    schema = conn.execute("DESCRIBE uploads").fetchall()
                    columns = [col[0] for col in schema]
                    if 'file_name' in columns or 'data' in columns:
                        self.logger.info("Detected old uploads table schema, dropping")
                        conn.execute("DROP TABLE uploads")
                    else:
                        self.logger.info("Renaming uploads table to duckdb_users")
                        conn.execute("ALTER TABLE uploads RENAME TO duckdb_users")
                except Exception:
                    self.logger.info("No uploads table found, creating duckdb_users")
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS duckdb_users (
                        user_id STRING,
                        session_id STRING,
                        table_name STRING
                    )
                """)
                conn.commit()
                conn.close()
                self.logger.info(f"DuckDB initialized at: {self.db_path}")
                print(f"✅ DuckDB initialized for session {self.session_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize DuckDB: {e}")
            raise

    def upload_to_gcs(self, file_path: str, destination_blob_name: str) -> str:
        if not self.gcs_enabled:
            self.logger.warning(f"GCS uploads disabled, skipping upload of {file_path}")
            return ""
        # This block will never execute due to gcs_enabled = False
        raise NotImplementedError("GCS uploads are disabled")

    def validate_balance_sheet(self, df: pd.DataFrame) -> bool:
        return True
    
    def detect_encoding(self, file_content: bytes) -> str:
        try:
            result = chardet.detect(file_content[:10000])
            encoding = result['encoding']
            confidence = result['confidence']
            self.logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
            if confidence < 0.7:
                self.logger.warning(f"Low confidence in encoding detection, trying common encodings")
                return self.try_common_encodings(file_content)
            return encoding
        except Exception as e:
            self.logger.error(f"Error detecting encoding: {str(e)}")
            return self.try_common_encodings(file_content)

    def try_common_encodings(self, file_content: bytes) -> str:
        common_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        for encoding in common_encodings:
            try:
                file_content.decode(encoding)
                self.logger.info(f"Successfully decoded file with encoding: {encoding}")
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        self.logger.warning("All encoding attempts failed, defaulting to latin-1")
        return 'latin-1'

    def load_file_with_encoding(self, file_path: str) -> pd.DataFrame:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.xlsx':
                df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
                self.logger.info(f"Successfully loaded Excel file from {file_path}")
            elif file_extension == '.csv':
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                encoding = self.detect_encoding(file_content)
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    na_values=['NA', 'N/A', '-', 'TBD', ''],
                    keep_default_na=False,
                    on_bad_lines='skip'
                )
                self.logger.info(f"Successfully loaded CSV from {file_path} with encoding {encoding}")
            else:
                raise ValueError(f"Unsupported file extension: {file_extension}. Only .xlsx and .csv are supported.")
            
            self.validate_balance_sheet(df)
            return df
        except Exception as e:
            self.logger.error(f"Failed to load file {file_path}: {str(e)}")
            raise ValueError(f"Could not load file {file_path}: {str(e)}")

    def validate_dataframe(self, df: pd.DataFrame, context: str = "") -> bool:
        prefix = f"{context}: " if context else ""
        if df is None or df.empty or len(df.columns) == 0:
            raise ValueError(f"{prefix}Invalid DataFrame: None={df is None}, empty={df.empty}, columns={len(df.columns)}")
        csv_size = len(df.to_csv(index=False).encode('utf-8'))
        if csv_size > 100000:
            raise ValueError(f"{prefix}Input file too large: {csv_size} bytes")
        self.logger.debug(f"{prefix}DataFrame validated: {len(df)} rows, {len(df.columns)} columns")
        return True

    def fallback_processing(self, df: pd.DataFrame, table_name: str) -> Tuple[Dict, pd.DataFrame]:
        self.logger.warning("Using fallback processing due to LLM failure")
        cleaned_df = df.copy()
        
        fiscal_year_cols = []
        for col in df.columns:
            if any(pattern in str(col).lower() for pattern in ['fy', 'year', '20', '19']):
                fiscal_year_cols.append(col)
        
        if not fiscal_year_cols:
            cleaned_df['fiscal_year'] = [f"FY_{2020 + i}" for i in range(len(cleaned_df))]
        else:
            fy_col = fiscal_year_cols[0]
            cleaned_df['fiscal_year'] = df[fy_col].astype(str).replace(['nan', 'NaN', '', 'None'], np.nan)
            if fy_col != 'fiscal_year':
                cleaned_df = cleaned_df.drop(columns=[fy_col])
        
        nan_mask = cleaned_df['fiscal_year'].isna() | (cleaned_df['fiscal_year'] == 'nan')
        if nan_mask.any():
            self.logger.warning(f"Found {nan_mask.sum()} rows with NaN or 'nan' in fiscal_year, assigning unique generated values")
            nan_indices = cleaned_df[nan_mask].index
            base_year = 2020 + len(cleaned_df) - nan_mask.sum()
            for idx, nan_idx in enumerate(nan_indices):
                cleaned_df.at[nan_idx, 'fiscal_year'] = f"FY_{base_year + idx}"
        
        cleaned_df.columns = [self.clean_column_name(col) for col in cleaned_df.columns]
        cols = ['fiscal_year'] + [col for col in cleaned_df.columns if col != 'fiscal_year']
        cleaned_df = cleaned_df[cols]
        
        for col in cleaned_df.columns:
            if col != 'fiscal_year':
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                cleaned_df[col] = cleaned_df[col].where(cleaned_df[col].notna(), None)
        
        if cleaned_df['fiscal_year'].isna().any():
            self.logger.error("fiscal_year column contains NaN values after processing")
            raise ValueError("fiscal_year column contains NaN values after processing")
        if cleaned_df['fiscal_year'].duplicated().any():
            duplicates = cleaned_df[cleaned_df['fiscal_year'].duplicated()]['fiscal_year'].tolist()
            self.logger.error(f"Duplicate fiscal_year values found: {duplicates}")
            raise ValueError(f"fiscal_year column contains duplicate values: {duplicates}")
        
        self.logger.info(f"fiscal_year values: {cleaned_df['fiscal_year'].tolist()}")
        
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
                sample_csv = csv_buffer.getvalue()[:20000]
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
            else:
                cleaned_df['fiscal_year'] = cleaned_df['fiscal_year'].astype(str).replace(['nan', 'NaN', '', 'None'], np.nan)
                nan_mask = cleaned_df['fiscal_year'].isna() | (cleaned_df['fiscal_year'] == 'nan')
                if nan_mask.any():
                    self.logger.warning(f"Found {nan_mask.sum()} rows with NaN or 'nan' in fiscal_year, assigning unique generated values")
                    nan_indices = cleaned_df[nan_mask].index
                    base_year = 2020 + len(cleaned_df) - nan_mask.sum()
                    for idx, nan_idx in enumerate(nan_indices):
                        cleaned_df.at[nan_idx, 'fiscal_year'] = f"FY_{base_year + idx}"
            
            cleaned_df['fiscal_year'] = cleaned_df['fiscal_year'].astype(str)
            for col in cleaned_df.columns:
                if col != 'fiscal_year':
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    cleaned_df[col] = cleaned_df[col].where(cleaned_df[col].notna(), None)
            
            cleaned_df = cleaned_df.drop_duplicates(subset=['fiscal_year'], keep='first')
            
            if cleaned_df['fiscal_year'].isna().any():
                self.logger.error("fiscal_year column contains NaN values after processing")
                raise ValueError("fiscal_year column contains NaN values after processing")
            if cleaned_df['fiscal_year'].duplicated().any():
                duplicates = cleaned_df[cleaned_df['fiscal_year'].duplicated()]['fiscal_year'].tolist()
                self.logger.error(f"Duplicate fiscal_year values found: {duplicates}")
                raise ValueError(f"fiscal_year column contains duplicate values: {duplicates}")
            
            self.logger.info(f"fiscal_year values: {cleaned_df['fiscal_year'].tolist()}")
            
            nan_counts = cleaned_df.isna().sum()
            for col, count in nan_counts.items():
                if count > 0:
                    self.logger.info(f"Column '{col}' has {count} NaN values, converted to NULL")
            
            self.logger.info(f"LLM processing successful. Shape: {cleaned_df.shape}")
            return result, cleaned_df
            
        except Exception as e:
            self.logger.error(f"LLM processing failed: {str(e)}")
            return self.fallback_processing(df, table_name)

    def validate_llm_result(self, result: Dict) -> bool:
        try:
            required_fields = ['cleaned_data', 'schema', 'data_quality_report']
            for field in required_fields:
                if field not in result:
                    self.logger.error(f"Missing required field: {field}")
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

    def save_preprocessed_csv(self, df: pd.DataFrame, table_name: str) -> str:
        try:
            preprocessed_dir = os.path.join(self.db_base_path, "preprocessed")
            os.makedirs(preprocessed_dir, exist_ok=True)
            csv_path = os.path.join(preprocessed_dir, f"{self.session_id}_{table_name}.csv")
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Preprocessed CSV saved to: {csv_path}")
            print(f"💾 Preprocessed CSV saved to: {csv_path}")
            
            # No GCS upload since gcs_enabled is False
            return csv_path
        except Exception as e:
            self.logger.error(f"Failed to save preprocessed CSV: {e}")
            raise

    def load_file_to_duckdb(self, df: pd.DataFrame, table_name: str, llm_result: Dict) -> str:
        try:
            self.validate_dataframe(df, "DuckDB loading")
            
            if df['fiscal_year'].isna().any():
                self.logger.error("fiscal_year column contains NaN values before insertion")
                raise ValueError("fiscal_year column contains NaN values before insertion")
            if df['fiscal_year'].duplicated().any():
                duplicates = df[df['fiscal_year'].duplicated()]['fiscal_year'].tolist()
                self.logger.error(f"Duplicate fiscal_year values before insertion: {duplicates}")
                raise ValueError(f"fiscal_year column contains duplicate values before insertion: {duplicates}")
            
            self.logger.info(f"fiscal_year values before insertion: {df['fiscal_year'].tolist()}")
            
            schema = llm_result.get('schema', {})
            if not schema or 'columns' not in schema:
                raise ValueError("Invalid or missing schema in llm_result")
            
            columns_def = []
            for col, col_type in schema['columns'].items():
                clean_col = self.clean_column_name(col)
                if 'DECIMAL' in col_type.upper():
                    columns_def.append(f'"{clean_col}" {col_type}')
                else:
                    columns_def.append(f'"{clean_col}" {col_type}')
            columns_sql = ", ".join(columns_def)
            
            with self.lock:
                conn = duckdb.connect(self.db_path)
                conn.execute(f"DROP TABLE IF EXISTS \"{table_name}\"")
                conn.execute(f"CREATE TABLE \"{table_name}\" ({columns_sql})")
                
                columns = [self.clean_column_name(col) for col in df.columns]
                placeholders = ", ".join(["?" for _ in columns])
                insert_sql = f"INSERT INTO \"{table_name}\" ({', '.join([f'\"{col}\"' for col in columns])}) VALUES ({placeholders})"
                
                for _, row in df.iterrows():
                    row_values = [None if pd.isna(val) else val for val in row.tolist()]
                    conn.execute(insert_sql, row_values)
                
                conn.execute(
                    """
                    INSERT INTO duckdb_users (user_id, session_id, table_name)
                    VALUES (?, ?, ?)
                    """,
                    (self.user_id, self.session_id, table_name)
                )
                conn.commit()
                
                row_count = conn.execute(f"SELECT COUNT(*) FROM \"{table_name}\"").fetchone()[0]
                conn.close()
                
                self.logger.info(f"Inserted data into table {table_name}. Total rows: {row_count}")
                print(f"📊 Data inserted into table {table_name} (total rows: {row_count})")
                
                # No GCS upload since gcs_enabled is False
                return self.db_path
                
        except Exception as e:
            self.logger.error(f"Failed to load data to DuckDB: {e}")
            raise ValueError(f"Failed to load data to DuckDB: {str(e)}")

    def query_duckdb(self, sql_query: str) -> pd.DataFrame:
        try:
            with self.lock:
                conn = duckdb.connect(self.db_path)
                result_df = conn.execute(sql_query).df()
                conn.close()
                print(f"✅ Query executed successfully, returned {len(result_df)} rows")
                return result_df
        except Exception as e:
            self.logger.error(f"Failed to query DuckDB: {e}")
            raise

    def list_tables(self) -> pd.DataFrame:
        try:
            with self.lock:
                conn = duckdb.connect(self.db_path)
                tables_df = conn.execute("SHOW TABLES").df()
                conn.close()
                print(f"📊 Found {len(tables_df)} tables in database")
                return tables_df
        except Exception as e:
            self.logger.error(f"Failed to list tables: {e}")
            raise

    def get_table_schema(self, table_name: str) -> pd.DataFrame:
        try:
            with self.lock:
                conn = duckdb.connect(self.db_path)
                schema_df = conn.execute(f'DESCRIBE "{table_name}"').df()
                conn.close()
                print(f"📋 Schema for table '{table_name}':")
                print(schema_df.to_string())
                return schema_df
        except Exception as e:
            self.logger.error(f"Failed to get table schema: {e}")
            raise

    def delete_table(self, table_name: str) -> bool:
        try:
            with self.lock:
                conn = duckdb.connect(self.db_path)
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.execute("DELETE FROM duckdb_users WHERE table_name = ?", (table_name,))
                conn.close()
                print(f"🗑️ Table '{table_name}' deleted successfully")
                return True
        except Exception as e:
            self.logger.error(f"Failed to delete table: {e}")
            raise