import pandas as pd
import json
import io
import os
import logging
import re
import chardet
import tempfile
from groq import Groq
from typing import Dict, Tuple
from pathlib import Path
from google.cloud import storage
from Prompts.preprocess_prompt import create_simplified_cleaning_prompt
from logger import logger
import duckdb

class DataProcessor:
    def __init__(self, session_id: str, model: str = "qwen/qwen3-32B", bucket_name: str = "data-gpt"):
        self.session_id = session_id
        self.bucket_name = bucket_name
        self.db_path = f"{session_id}/my_duckdb.duckdb"  # GCS path
        self.model = model
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize Groq client
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        self.client = Groq(api_key=groq_api_key)
        
        # Initialize GCS client
        self.storage_client = None
        self.bucket = None
        self._init_gcs_client()
    
    def _init_gcs_client(self):
        """Initialize Google Cloud Storage client with proper error handling"""
        credentials_path = "service_account_key.json"
        
        try:
            # Check if credentials file exists
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(f"Service account key file not found: {credentials_path}")
            
            # Load and validate credentials
            with open(credentials_path, "r") as f:
                creds_info = json.load(f)
            
            self.logger.info(f"JSON keys: {list(creds_info.keys())}")
            
            # Validate required fields
            required_fields = ["type", "project_id", "private_key", "client_email"]
            missing_fields = [field for field in required_fields if not creds_info.get(field)]
            
            if missing_fields:
                raise ValueError(f"Missing required fields in service account key: {missing_fields}")
            
            # Validate private key format
            private_key = creds_info.get("private_key", "")
            if "BEGIN PRIVATE KEY" not in private_key:
                raise ValueError("Invalid service_account_key.json: Missing or malformed private_key")
            
            # Create storage client
            self.storage_client = storage.Client.from_service_account_json(credentials_path)
            
            # Get bucket reference
            self.bucket = self.storage_client.bucket(self.bucket_name)
            
            # Test bucket access
            try:
                # This will raise an exception if bucket doesn't exist or we don't have access
                self.bucket.reload()
                self.logger.info(f"Successfully connected to GCS bucket: {self.bucket_name}")
                print(f"✅ Successfully connected to GCS bucket: {self.bucket_name}")
            except Exception as bucket_error:
                self.logger.warning(f"Bucket access test failed: {bucket_error}")
                print(f"⚠️ Bucket access test failed: {bucket_error}")
                # Don't raise here - bucket might not exist yet and could be created later
            
        except FileNotFoundError as e:
            self.logger.error(f"Credentials file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in credentials file: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Invalid credentials: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize GCS client: {e}")
            raise
        
    def upload_to_gcs(self, file_content: bytes, destination_path: str) -> str:
        """Upload a file to GCS and return the GCS URI."""
        try:
            blob = self.bucket.blob(destination_path)
            blob.upload_from_string(file_content)
            gcs_uri = f"gs://{self.bucket_name}/{destination_path}"
            self.logger.info(f"Uploaded file to GCS: {gcs_uri}")
            print(f"📤 Uploaded file to GCS: {gcs_uri}")
            return gcs_uri
        except Exception as e:
            self.logger.error(f"Failed to upload to GCS: {e}")
            print(f"❌ Failed to upload to GCS: {e}")
            raise

    def download_from_gcs(self, source_path: str) -> bytes:
        """Download a file from GCS."""
        try:
            blob = self.bucket.blob(source_path)
            file_content = blob.download_as_bytes()
            self.logger.info(f"Downloaded file from GCS: gs://{self.bucket_name}/{source_path}")
            print(f"📥 Downloaded file from GCS: gs://{self.bucket_name}/{source_path}")
            return file_content
        except Exception as e:
            self.logger.error(f"Failed to download from GCS: {e}")
            print(f"❌ Failed to download from GCS: {e}")
            raise

    def detect_encoding(self, file_content: bytes) -> str:
        """Detect file encoding using chardet from file content."""
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
        """Try common encodings to find one that works."""
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

    def load_csv_with_encoding(self, gcs_path: str) -> pd.DataFrame:
        """Load CSV file from GCS with proper encoding detection."""
        try:
            # Download file from GCS
            file_content = self.download_from_gcs(gcs_path)
            encoding = self.detect_encoding(file_content)
            # Load CSV from bytes
            df = pd.read_csv(
                io.BytesIO(file_content),
                encoding=encoding,
                na_values=['NA', 'N/A', '-', 'TBD', ''],
                keep_default_na=False,
                on_bad_lines='skip'
            )
            self.logger.info(f"Successfully loaded CSV from GCS: gs://{self.bucket_name}/{gcs_path} with encoding {encoding}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load CSV with encoding {encoding}: {str(e)}")
            try:
                df = pd.read_csv(
                    io.BytesIO(file_content),
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
                raise ValueError(f"Could not load CSV file from GCS: {str(e2)}")

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
        """Fallback processing when LLM fails."""
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
        """Convert column name to snake_case."""
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
        """Validate LLM result structure."""
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

    def save_preprocessed_csv_to_gcs(self, df: pd.DataFrame, table_name: str) -> str:
        """Save preprocessed CSV to GCS and return the GCS URI."""
        try:
            # Create CSV buffer
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue().encode('utf-8')
            
            # Upload to GCS
            csv_path = f"{self.session_id}/preprocessed_{table_name}.csv"
            gcs_uri = self.upload_to_gcs(csv_content, csv_path)
            
            print(f"💾 Preprocessed CSV saved to: {gcs_uri}")
            return gcs_uri
        except Exception as e:
            self.logger.error(f"Failed to save preprocessed CSV to GCS: {e}")
            print(f"❌ Failed to save preprocessed CSV to GCS: {e}")
            raise

    def load_file_to_duckdb(self, df: pd.DataFrame, table_name: str, llm_result: Dict) -> str:
        """Load a DataFrame into DuckDB and store the database in GCS."""
        try:
            # Validate DataFrame
            self.validate_dataframe(df, "DuckDB loading")
            
            # Save preprocessed CSV to GCS first
            csv_gcs_uri = self.save_preprocessed_csv_to_gcs(df, table_name)
            
            # Create temporary local DuckDB file
            with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as temp_db:
                temp_db_path = temp_db.name
            
            try:
                # Connect to temporary DuckDB
                conn = duckdb.connect(temp_db_path)
                
                # Create table with proper schema
                columns_def = ', '.join([f'"{col}" {dtype}' for col, dtype in llm_result['schema']['columns'].items()])
                create_table_sql = f'CREATE TABLE "{table_name}" ({columns_def})'
                conn.execute(create_table_sql)
                
                # Insert data
                for _, row in df.iterrows():
                    values = [row[col] if not pd.isna(row[col]) else None for col in df.columns]
                    placeholders = ', '.join(['?' for _ in values])
                    conn.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', values)
                
                # Verify insertion
                row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
                self.logger.info(f"Table {table_name} created with {row_count} rows in temporary DuckDB")
                print(f"📊 Table '{table_name}' created with {row_count} rows in DuckDB")
                
                # Close connection before uploading
                conn.close()
                
                # Upload DuckDB file to GCS
                with open(temp_db_path, 'rb') as f:
                    db_content = f.read()
                
                db_gcs_uri = self.upload_to_gcs(db_content, self.db_path)
                print(f"🗄️ DuckDB database uploaded to: {db_gcs_uri}")
                
                return db_gcs_uri
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_db_path):
                    os.unlink(temp_db_path)
                    
        except Exception as e:
            self.logger.error(f"Failed to load DataFrame to DuckDB: {e}")
            print(f"❌ Failed to load DataFrame to DuckDB: {e}")
            raise

    def get_duckdb_from_gcs(self) -> str:
        """Download DuckDB from GCS to a temporary file and return the path."""
        try:
            # Download DuckDB from GCS
            db_content = self.download_from_gcs(self.db_path)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.duckdb', delete=False) as temp_db:
                temp_db.write(db_content)
                temp_db_path = temp_db.name
            
            print(f"📥 DuckDB downloaded from GCS to temporary file: {temp_db_path}")
            return temp_db_path
            
        except Exception as e:
            self.logger.error(f"Failed to download DuckDB from GCS: {e}")
            print(f"❌ Failed to download DuckDB from GCS: {e}")
            raise

    def query_duckdb(self, sql_query: str) -> pd.DataFrame:
        """Execute SQL query on DuckDB stored in GCS."""
        temp_db_path = None
        try:
            # Download DuckDB from GCS
            temp_db_path = self.get_duckdb_from_gcs()
            
            # Connect and query
            conn = duckdb.connect(temp_db_path)
            result_df = conn.execute(sql_query).df()
            conn.close()
            
            print(f"✅ Query executed successfully, returned {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to query DuckDB: {e}")
            print(f"❌ Failed to query DuckDB: {e}")
            raise
        finally:
            # Clean up temporary file
            if temp_db_path and os.path.exists(temp_db_path):
                os.unlink(temp_db_path)