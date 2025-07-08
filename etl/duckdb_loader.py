import pandas as pd
import duckdb
import os
from typing import Dict, Tuple
from logger import logger
from services.validate_create_session import SessionData
from pathlib import Path

def load_file_to_duckdb(file_path: str, table_name: str, session_data: SessionData) -> Tuple[pd.DataFrame, Dict]:

    try:
        # Validate session_id
        if not session_data.session_id:
            logger.error("Session ID is missing in session_data")
            raise ValueError("Session ID is missing")

        # Use db_path from session_data
        db_path = Path(session_data.db_path).as_posix()
        logger.info(f"Using DuckDB database path: {db_path}")

        # Check if db_path exists and is invalid
        if os.path.exists(db_path):
            try:
                conn = duckdb.connect(db_path)
                conn.execute("SELECT 1")
                conn.close()
            except duckdb.IOException:
                logger.warning(f"Invalid DuckDB file detected at {db_path}. Deleting and recreating.")
                os.remove(db_path)

        # Connect to DuckDB (creates file if it doesn't exist)
        conn = duckdb.connect(db_path)

        # Use cleaned_df and llm_result from session_data (set by process_uploaded_file)
        cleaned_df = session_data.cleaned_df
        llm_result = session_data.llm_result

        if cleaned_df.empty:
            logger.error("No data remaining in session_data.cleaned_df")
            raise ValueError("No data remaining in cleaned DataFrame")

        # Drop existing table
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

        # Create table with schema
        columns_def = ', '.join([f'"{col}" {dtype}' for col, dtype in llm_result['schema']['columns'].items()])
        conn.execute(f'CREATE TABLE "{table_name}" ({columns_def})')

        # Insert data
        for _, row in cleaned_df.iterrows():
            values = [row[col] if not pd.isna(row[col]) else None for col in cleaned_df.columns]
            placeholders = ', '.join(['?' for _ in values])
            conn.execute(f'INSERT INTO "{table_name}" VALUES ({placeholders})', values)

        # Verify insertion
        row_count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        logger.info(f"Table {table_name} created with {row_count} rows")

        # Update session_data
        session_data.db_path = db_path
        session_data.table_name = table_name

        return cleaned_df, llm_result

    except Exception as e:
        logger.error(f"Error loading file to DuckDB: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to load file to DuckDB: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()