from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import pandas as pd
import duckdb
import os
import psycopg2
from urllib.parse import urlparse
from logger import logger
from pathlib import Path
from dotenv import load_dotenv
import threading
import chainlit as cl
import uuid
from etl.data_cleaning import DataProcessor
import re

load_dotenv()

@dataclass
class SessionData:
    session_id: str
    user_id: Optional[str] = None
    db_path: Optional[str] = None
    table_name: Optional[str] = None
    cleaned_df: Optional[pd.DataFrame] = None
    llm_result: Optional[Dict] = None
    
    def __post_init__(self):
        if self.db_path is None:
            self.db_path = os.path.join("db", "master.duckdb")

class MultiUserDuckDBManager:
    def __init__(self, db_base_path: str = "db"):
        self.db_base_path = db_base_path
        self.db_path = os.path.join(db_base_path, "master.duckdb")
        self.lock = threading.Lock()
        self._ensure_db_directory()
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")
        parsed_url = urlparse(database_url)
        self.pg_conn_params = {
            "dbname": parsed_url.path.lstrip('/'),
            "user": parsed_url.username,
            "password": parsed_url.password,
            "host": parsed_url.hostname,
            "port": parsed_url.port or "5432"
        }

    def _ensure_db_directory(self):
        os.makedirs(self.db_base_path, exist_ok=True)

    def get_user_id_from_postgres(self, thread_id: str) -> str:
        logger.debug("Entering get_user_id_from_postgres with thread_id: %s", thread_id)
        try:
            conn = psycopg2.connect(**self.pg_conn_params)
            cursor = conn.cursor()
            
            cursor.execute('SELECT "userId" FROM public."Thread" WHERE id = %s', (thread_id,))
            result = cursor.fetchone()
            logger.info(f"PostgreSQL query result for thread_id {thread_id}: {result}")
            
            if not result or result[0] is None:
                logger.error(f"No userId found for thread id {thread_id}")
                raise ValueError(f"No userId found for thread id {thread_id}")
            
            user_id = str(result[0])
            logger.info(f"Fetched user_id: {user_id} from PostgreSQL Thread table")
            SessionData.user_id = user_id
            cursor.close()
            conn.close()
            return user_id
            
        except Exception as e:
            logger.error(f"Failed to fetch user_id from PostgreSQL for thread_id {thread_id}: {str(e)}")
            raise ValueError(f"Failed to fetch user_id from PostgreSQL: {str(e)}")

    def update_element_urls_for_session(self,session_id: str) -> None:
        try:
            conn = psycopg2.connect(**self.pg_conn_params)
            cursor = conn.cursor()

            # Fetch all rows from Element where threadId = session_id
            cursor.execute('SELECT id, "objectKey" FROM public."Element" WHERE "threadId" like %s', (session_id,))
            rows = cursor.fetchall()
            if not rows:
                print(f"No entries found in Element table for threadId: {session_id}")
                return

            print(f"Found {len(rows)} rows for session_id: {session_id}")

            for row_id, object_key in rows:
                if not object_key:
                    print(f"Skipping row with id {row_id} due to missing objectKey")
                    continue

                url = f"https://storage.googleapis.com/data-gpt/{object_key}"

                # You might want to store this in a specific column (e.g. 'url')
                cursor.execute('UPDATE public."Element" SET "url" =%s  WHERE id = %s', 
                            (url, row_id))
                    
            conn.commit()
            print(f"✅ Updated {len(rows)} rows with GCS URLs for session_id: {session_id}")

        except Exception as e:
            print(f"❌ Error while updating Element table: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()


    async def load_file_to_duckdb(self, file_path: str, session_data: SessionData) -> Tuple[pd.DataFrame, Dict]:
        try:
            if not session_data.session_id:
                logger.error("Session ID is missing")
                raise ValueError("Session ID is missing")
            
            session_data.user_id = self.get_user_id_from_postgres(session_data.session_id)
            logger.info(f"Session data user_id after fetch: {session_data.user_id}")
            
            filename = os.path.splitext(os.path.basename(file_path))[0]
            file_extension = os.path.splitext(file_path)[1].lower()
            table_name = f"{session_data.session_id}_{filename}"
            
            processor = DataProcessor(session_data.session_id, session_data.user_id, db_base_path=self.db_base_path)
            df = processor.load_file_with_encoding(file_path)
        
            
            llm_result, cleaned_df = processor.preprocess_with_llm(df, table_name)
            
            if cleaned_df is None or cleaned_df.empty:
                logger.error("No data available in cleaned_df after processing")
                raise ValueError("No data available in cleaned DataFrame")
            
            session_data.cleaned_df = cleaned_df
            session_data.llm_result = llm_result
            session_data.table_name = table_name
            
            db_path = processor.load_file_to_duckdb(cleaned_df, table_name, llm_result)
            session_data.db_path = db_path
            self.update_element_urls_for_session(session_data.session_id)
            
            logger.info(f"User_id inserted into duckdb_users: {session_data.user_id}")
            print(f"✅ Data successfully loaded to table {table_name}!")
            print(f"👤 User: {session_data.user_id}")
            print(f"📊 Table '{table_name}' created/updated with new data")
            print(f"📍 DuckDB database stored locally at: {db_path}")
            
            return cleaned_df, llm_result
        except Exception as e:
            logger.error(f"Error loading file to DuckDB: {str(e)}")
            raise ValueError(f"Failed to load file to DuckDB: {str(e)}")