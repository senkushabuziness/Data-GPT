from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd
from typing import Tuple
from etl.data_cleaning import DataProcessor
from logger import logger
from pathlib import Path
import io


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

# Updated processing function to work with GCS
def process_uploaded_file(file_content: bytes, filename: str, session_data: SessionData) -> Tuple[pd.DataFrame, Dict]:
    """
    Process uploaded file using GCS-based DataProcessor
    """
    try:
        # Initialize DataProcessor with GCS support
        processor = DataProcessor(
            session_id=session_data.session_id,
            bucket_name=session_data.bucket_name
        )
        
        # Upload original file to GCS
        original_file_path = f"{session_data.session_id}/original_{filename}"
        processor.upload_to_gcs(file_content, original_file_path)
        print(f"📤 Original file uploaded to GCS: gs://{session_data.bucket_name}/{original_file_path}")
        
        # Load the file (supports CSV, Excel, etc.)
        if filename.lower().endswith('.csv'):
            df = processor.load_csv_with_encoding(original_file_path)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            # For Excel files, load them first
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        # Process with LLM
        table_name = "data_table"
        llm_result, cleaned_df = processor.preprocess_with_llm(df, table_name)
        
        # Store results in session_data
        session_data.cleaned_df = cleaned_df
        session_data.llm_result = llm_result
        session_data.table_name = table_name
        
        # Load to DuckDB and store in GCS
        db_gcs_uri = processor.load_file_to_duckdb(cleaned_df, table_name, llm_result)
        print(f"✅ File processing completed successfully!")
        print(f"🗄️ DuckDB stored at: {db_gcs_uri}")
        
        return cleaned_df, llm_result
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}", exc_info=True)
        raise ValueError(f"File processing failed: {str(e)}")

# Updated load_file_to_duckdb function for GCS integration
def load_file_to_duckdb(file_path: str, table_name: str, session_data: SessionData) -> Tuple[pd.DataFrame, Dict]:
    """
    Load file to DuckDB using GCS storage - this replaces your original function
    """
    try:
        # Validate session_id
        if not session_data.session_id:
            logger.error("Session ID is missing in session_data")
            raise ValueError("Session ID is missing")

        # Use cleaned_df and llm_result from session_data (set by process_uploaded_file)
        cleaned_df = session_data.cleaned_df
        llm_result = session_data.llm_result

        if cleaned_df is None or cleaned_df.empty:
            logger.error("No data available in session_data.cleaned_df")
            raise ValueError("No data available in cleaned DataFrame")

        # Initialize DataProcessor
        processor = DataProcessor(
            session_id=session_data.session_id,
            bucket_name=session_data.bucket_name
        )

        # Load to DuckDB and store in GCS
        db_gcs_uri = processor.load_file_to_duckdb(cleaned_df, table_name, llm_result)
        
        # Update session_data
        session_data.db_path = f"{session_data.session_id}/my_duckdb.duckdb"
        session_data.table_name = table_name

        print(f"✅ Data successfully loaded to DuckDB in GCS!")
        print(f"🗄️ Database location: {db_gcs_uri}")
        print(f"📊 Table '{table_name}' created with {len(cleaned_df)} rows")

        return cleaned_df, llm_result

    except Exception as e:
        logger.error(f"Error loading file to DuckDB: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to load file to DuckDB: {str(e)}")

# Helper function to query data from GCS-stored DuckDB
def query_data_from_gcs(session_data: SessionData, sql_query: str) -> pd.DataFrame:
    """
    Query data from DuckDB stored in GCS
    """
    try:
        processor = DataProcessor(
            session_id=session_data.session_id,
            bucket_name=session_data.bucket_name
        )
        
        result_df = processor.query_duckdb(sql_query)
        return result_df
        
    except Exception as e:
        logger.error(f"Error querying data from GCS: {str(e)}")
        raise ValueError(f"Failed to query data: {str(e)}")