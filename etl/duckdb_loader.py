from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import pandas as pd
import duckdb
import os
import psycopg2
from logger import logger
from pathlib import Path
from dotenv import load_dotenv
from etl.data_cleaning import DataProcessor
import threading
import chainlit as cl
import uuid

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
        # PostgreSQL connection parameters from .env
        self.pg_conn_params = {
            "dbname": os.environ.get("POSTGRES_DB", "chainlit"),
            "user": os.environ.get("POSTGRES_USER", "postgres"),
            "password": os.environ.get("POSTGRES_PASSWORD", ""),
            "host": os.environ.get("POSTGRES_HOST", "localhost"),
            "port": os.environ.get("POSTGRES_PORT", "5432")
        }

    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        os.makedirs(self.db_base_path, exist_ok=True)

    def get_user_id_from_postgres(self) -> str:
        """Fetch user_id from Chainlit's PostgreSQL database using current session."""
        try:
            chainlit_user = cl.user_session.get("user")
            if not chainlit_user:
                logger.error("No Chainlit user found in session")
                raise ValueError("No Chainlit user found in session")
            
            conn = psycopg2.connect(**self.pg_conn_params)
            cursor = conn.cursor()
            
            cursor.execute("SELECT id FROM users WHERE identifier = %s", (chainlit_user.identifier,))
            result = cursor.fetchone()
            if not result:
                logger.error(f"No user found with identifier {chainlit_user.identifier}")
                raise ValueError(f"No user found with identifier {chainlit_user.identifier}")
            
            user_id = str(result[0])
            cursor.close()
            conn.close()
            logger.info(f"Fetched user_id: {user_id} from PostgreSQL")
            return user_id
        except Exception as e:
            logger.error(f"Failed to fetch user_id from PostgreSQL: {str(e)}")
            raise ValueError(f"Failed to fetch user_id from PostgreSQL: {str(e)}")

    def load_file_to_duckdb(self, file_path: str, table_name: str, session_data: SessionData) -> Tuple[pd.DataFrame, Dict]:
        try:
            if not session_data.session_id:
                logger.error("Session ID is missing")
                raise ValueError("Session ID is missing")
            
            if not session_data.user_id:
                session_data.user_id = self.get_user_id_from_postgres()
            
            processor = DataProcessor(session_data.session_id, session_data.user_id, db_base_path=self.db_base_path)
            df = processor.load_file_with_encoding(file_path)
            
            llm_result, cleaned_df = processor.preprocess_with_llm(df, table_name)
            
            if cleaned_df is None or cleaned_df.empty:
                logger.error("No data available in cleaned_df after processing")
                raise ValueError("No data available in cleaned DataFrame")
            
            session_data.cleaned_df = cleaned_df
            session_data.llm_result = llm_result
            
            gcs_uri = processor.load_file_to_duckdb(cleaned_df, table_name, llm_result)
            session_data.db_path = processor.db_path
            
            print(f"✅ Data successfully loaded to table {table_name}!")
            print(f"👤 User: {session_data.user_id}")
            print(f"📊 Table '{table_name}' created/updated with new data")
            if gcs_uri and gcs_uri.startswith("gs://"):
                print(f"📤 DuckDB database uploaded to GCS: {gcs_uri}")
            else:
                print(f"⚠️ GCS upload skipped, using local path: {gcs_uri}")
            
            return cleaned_df, llm_result
        except Exception as e:
            logger.error(f"Error loading file to DuckDB: {str(e)}")
            raise ValueError(f"Failed to load file to DuckDB: {str(e)}")

    def query_user_database(self, session_data: SessionData, sql_query: str) -> pd.DataFrame:
        try:
            if not session_data.session_id:
                raise ValueError("Session ID is missing")
            if not session_data.user_id:
                session_data.user_id = self.get_user_id_from_postgres()
            
            processor = DataProcessor(session_data.session_id, session_data.user_id, db_base_path=self.db_base_path)
            return processor.query_duckdb(sql_query)
        except Exception as e:
            logger.error(f"Error querying database: {str(e)}")
            raise ValueError(f"Failed to query database: {str(e)}")

    def list_user_tables(self, session_data: SessionData) -> pd.DataFrame:
        try:
            if not session_data.session_id:
                raise ValueError("Session ID is missing")
            if not session_data.user_id:
                session_data.user_id = self.get_user_id_from_postgres()
            
            processor = DataProcessor(session_data.session_id, session_data.user_id, db_base_path=self.db_base_path)
            with processor.lock:
                conn = duckdb.connect(self.db_path)
                tables_df = conn.execute("SELECT table_name FROM duckdb_users WHERE user_id = ? AND session_id = ?", 
                                       (session_data.user_id, session_data.session_id)).df()
                conn.close()
                print(f"📊 Found {len(tables_df)} tables for user {session_data.user_id} in session {session_data.session_id}")
                return tables_df
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            raise ValueError(f"Failed to list tables: {str(e)}")

    def get_table_info(self, session_data: SessionData, table_name: str) -> Dict:
        try:
            if not session_data.session_id:
                raise ValueError("Session ID is missing")
            if not session_data.user_id:
                session_data.user_id = self.get_user_id_from_postgres()
            
            processor = DataProcessor(session_data.session_id, session_data.user_id, db_base_path=self.db_base_path)
            with processor.lock:
                conn = duckdb.connect(self.db_path)
                table_exists = conn.execute(
                    "SELECT COUNT(*) FROM duckdb_users WHERE user_id = ? AND table_name = ?", 
                    (session_data.user_id, table_name)
                ).fetchone()[0] > 0
                if not table_exists:
                    conn.close()
                    raise ValueError(f"Table {table_name} not found for user {session_data.user_id}")
                
                schema_df = conn.execute(f"DESCRIBE \"{table_name}\"").fetchdf()
                count_result = conn.execute(f'SELECT COUNT(*) as row_count FROM "{table_name}"').fetchone()
                row_count = count_result[0]
                sample_df = conn.execute(f'SELECT * FROM "{table_name}" LIMIT 5').fetchdf()
                conn.close()
                
                return {
                    "table_name": table_name,
                    "user_id": session_data.user_id,
                    "schema": schema_df.to_dict('records'),
                    "row_count": row_count,
                    "sample_data": sample_df.to_dict('records')
                }
        except Exception as e:
            logger.error(f"Error getting table info: {str(e)}")
            raise ValueError(f"Failed to get table info: {str(e)}")

    def delete_user_data(self, session_data: SessionData) -> bool:
        try:
            if not session_data.user_id:
                session_data.user_id = self.get_user_id_from_postgres()
            
            processor = DataProcessor(session_data.session_id, session_data.user_id, db_base_path=self.db_base_path)
            with processor.lock:
                conn = duckdb.connect(self.db_path)
                tables = conn.execute("SELECT table_name FROM duckdb_users WHERE user_id = ?", 
                                    (session_data.user_id,)).fetchall()
                for (table_name,) in tables:
                    conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
                conn.execute("DELETE FROM duckdb_users WHERE user_id = ?", (session_data.user_id,))
                conn.commit()
                conn.close()
                
                logger.info(f"Deleted data for user: {session_data.user_id}")
                return True
        except Exception as e:
            logger.error(f"Error deleting user data: {str(e)}")
            raise ValueError(f"Failed to delete user data: {str(e)}")

    def get_all_users_info(self) -> List[Dict]:
        try:
            processor = DataProcessor(str(uuid.uuid4()), "system", db_base_path=self.db_base_path)
            with processor.lock:
                conn = duckdb.connect(self.db_path)
                users_df = conn.execute("SELECT DISTINCT user_id FROM duckdb_users").fetchdf()
                users_info = []
                for user_id in users_df['user_id']:
                    count_result = conn.execute("SELECT COUNT(*) as table_count FROM duckdb_users WHERE user_id = ?", (user_id,)).fetchone()
                    users_info.append({
                        "user_id": user_id,
                        "table_count": count_result[0]
                    })
                conn.close()
                return users_info
        except Exception as e:
            logger.error(f"Error getting all users info: {str(e)}")
            raise ValueError(f"Failed to get users info: {str(e)}")

    def get_database_stats(self) -> Dict:
        try:
            processor = DataProcessor(str(uuid.uuid4()), "system", db_base_path=self.db_base_path)
            with processor.lock:
                db_file_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                conn = duckdb.connect(self.db_path)
                users_df = conn.execute("SELECT DISTINCT user_id FROM duckdb_users").fetchdf()
                total_tables = conn.execute("SELECT COUNT(*) FROM duckdb_users").fetchone()[0]
                conn.close()
                
                return {
                    "db_file_path": self.db_path,
                    "db_file_size_bytes": db_file_size,
                    "db_file_size_mb": round(db_file_size / (1024 * 1024), 2),
                    "total_users": len(users_df),
                    "total_tables": total_tables,
                    "users_info": self.get_all_users_info()
                }
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            raise ValueError(f"Failed to get database stats: {str(e)}")