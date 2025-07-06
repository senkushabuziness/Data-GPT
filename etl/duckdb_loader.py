import pandas as pd
import duckdb
import os
import logging
from typing import Dict, Tuple
from data_cleaning import load_csv_with_encoding, preprocess_with_llm   
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_file_to_duckdb(file_path: str, table_name: str) -> Tuple[pd.DataFrame, Dict]:
        conn = None
        try:
            # Load the CSV file with encoding detection
            df = load_csv_with_encoding(file_path)
            logger.info(f"Loaded file {file_path} with shape {df.shape}")
            
            # Process with LLM (or fallback)
            llm_result, cleaned_df = preprocess_with_llm(df, table_name)
            
            if cleaned_df.empty:
                raise ValueError("No data remaining after preprocessing")
            
            # Connect to DuckDB
            conn = duckdb.connect(db_path)
            
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
            
            return cleaned_df, llm_result
            
        except Exception as e:
            logger.error(f"Error loading file to DuckDB: {str(e)}")
            raise ValueError(f"Failed to load file to DuckDB: {str(e)}")
        finally:
            if conn:
                conn.close()