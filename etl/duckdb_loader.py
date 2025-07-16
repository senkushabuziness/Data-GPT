from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import pandas as pd
import duckdb
import io
import os
import tempfile
from google.cloud import storage
from logger import logger
from pathlib import Path
from dotenv import load_dotenv
from etl.data_cleaning import DataProcessor  # Import your DataProcessor class

load_dotenv()

@dataclass
class SessionData:
    session_id: str
    bucket_name: str = "data-gpt"
    db_path: Optional[str] = None
    table_name: Optional[str] = None
    cleaned_df: Optional[pd.DataFrame] = None
    llm_result: Optional[Dict] = None
    
    def __post_init__(self):
        if self.db_path is None:
            self.db_path = f"{self.session_id}/my_duckdb.duckdb"

def load_file_to_duckdb(file_path: str, table_name: str, session_data: SessionData) -> Tuple[pd.DataFrame, Dict]:
    """
    Load data to DuckDB and store the database in GCS under the session_id folder.
    """
    try:
        # Validate session_id
        if not session_data.session_id:
            logger.error("Session ID is missing in session_data")
            raise ValueError("Session ID is missing")

        # Use cleaned_df and llm_result from session_data
        cleaned_df = session_data.cleaned_df
        llm_result = session_data.llm_result

        if cleaned_df is None or cleaned_df.empty:
            logger.error("No data available in session_data.cleaned_df")
            raise ValueError("No data available in cleaned DataFrame")

        # Initialize DataProcessor
        processor = DataProcessor(
            session_id=session_data.session_id,
            bucket_name=os.getenv("BUCKET_NAME", "data-gpt")
        )

        # Create temporary local DuckDB file with proper cleanup
        temp_db_path = None
        conn = None
        
        try:
            # Create temporary file but don't use context manager to avoid premature deletion
            temp_db_fd, temp_db_path = tempfile.mkstemp(suffix='.duckdb')
            os.close(temp_db_fd)  # Close file descriptor but keep the file
            
            # Remove the empty file so DuckDB can create it fresh
            if os.path.exists(temp_db_path):
                os.unlink(temp_db_path)
            
            logger.info(f"Creating DuckDB at temporary path: {temp_db_path}")
            
            # Connect to DuckDB (this will create the file)
            conn = duckdb.connect(temp_db_path)
            
            # Verify connection is working
            conn.execute("SELECT 1 as test").fetchone()
            
            # Create table with schema from llm_result
            columns_def = ', '.join([f'"{col}" {dtype}' for col, dtype in llm_result['schema']['columns'].items()])
            create_table_sql = f'CREATE TABLE "{table_name}" ({columns_def})'
            
            logger.info(f"Creating table with SQL: {create_table_sql}")
            conn.execute(create_table_sql)

            # Insert data row by row with proper error handling
            insert_count = 0
            for index, row in cleaned_df.iterrows():
                try:
                    # Handle NaN values properly
                    values = []
                    for col in cleaned_df.columns:
                        val = row[col]
                        if pd.isna(val):
                            values.append(None)
                        else:
                            values.append(val)
                    
                    placeholders = ', '.join(['?' for _ in values])
                    insert_sql = f'INSERT INTO "{table_name}" VALUES ({placeholders})'
                    conn.execute(insert_sql, values)
                    insert_count += 1
                    
                except Exception as row_error:
                    logger.warning(f"Failed to insert row {index}: {row_error}")
                    continue

            # Commit the transaction
            conn.commit()
            
            # Verify insertion
            row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            logger.info(f"Table {table_name} created with {row_count} rows in temporary DuckDB")
            print(f"📊 Table '{table_name}' created with {row_count} rows in DuckDB")

            # Test a sample query to ensure the database is working
            sample_result = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 1').fetchall()
            logger.info(f"Sample query successful, got {len(sample_result)} rows")
            
            # Close connection properly before file operations
            conn.close()
            conn = None
            
            # Verify the database file exists and is valid
            if not os.path.exists(temp_db_path):
                raise ValueError(f"DuckDB file was not created at {temp_db_path}")
            
            file_size = os.path.getsize(temp_db_path)
            if file_size == 0:
                raise ValueError(f"DuckDB file is empty at {temp_db_path}")
            
            logger.info(f"DuckDB file created successfully, size: {file_size} bytes")
            
            # Test that the file can be opened again
            test_conn = duckdb.connect(temp_db_path)
            test_count = test_conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            test_conn.close()
            
            if test_count != row_count:
                raise ValueError(f"Data verification failed: expected {row_count}, got {test_count}")
            
            # Upload DuckDB file to GCS
            with open(temp_db_path, 'rb') as f:
                db_content = f.read()
            
            if len(db_content) == 0:
                raise ValueError("DuckDB file content is empty")
            
            db_gcs_uri = processor.upload_to_gcs(db_content, session_data.db_path)
            print(f"🗄️ DuckDB database uploaded to: {db_gcs_uri}")

            # Update session_data
            session_data.db_path = f"{session_data.session_id}/my_duckdb.duckdb"
            session_data.table_name = table_name

            print(f"✅ Data successfully loaded to DuckDB in GCS!")
            print(f"🗄️ Database location: {db_gcs_uri}")
            print(f"📊 Table '{table_name}' created with {len(cleaned_df)} rows")

            return cleaned_df, llm_result

        except Exception as e:
            logger.error(f"Error in DuckDB operations: {str(e)}", exc_info=True)
            raise
            
        finally:
            # Clean up connection and temporary file
            if conn:
                try:
                    conn.close()
                except:
                    pass
            
            if temp_db_path and os.path.exists(temp_db_path):
                try:
                    os.unlink(temp_db_path)
                    logger.info(f"Cleaned up temporary file: {temp_db_path}")
                except OSError as e:
                    logger.warning(f"Failed to clean up temporary file {temp_db_path}: {e}")

    except Exception as e:
        logger.error(f"Error loading file to DuckDB: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to load file to DuckDB: {str(e)}")