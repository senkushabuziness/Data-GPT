from typing import Optional
from fastapi import APIRouter, HTTPException, File, UploadFile, Query, Depends, Request, Body
from fastapi.responses import JSONResponse
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
from uuid import UUID, uuid4
import pandas as pd
import os
import duckdb
import logging
from groq import Groq
from etl.data_cleaning import DataProcessor
from models.response import UploadResponse, QueryRequest
from api.utils import get_schema_from_df
# from llm.nl_to_sql import generate_sql_query_and_execute
from services.upload_service import process_uploaded_file
from services.generate_sql_service import process_sql_query
from services.validate_create_session import SessionData, get_session_data, cookie, backend
from api.states import app_state
from logger import logger

router = APIRouter()


UPLOAD_DIR = "uploads"
DB_PATH = "duckdb\\mydb.duckdb"
os.makedirs(UPLOAD_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app_state = {
    "uploaded_file_path": None,
    "cleaned_df": None,
    "db_path": DB_PATH,
    "table_name": None,
    "client": None
}

@router.post("/create-session")
async def create_session(session_id: Optional[str] = Body(None)):
    if not session_id:
        session_id = str(uuid4())
    try:
        UUID(session_id)  # Validate session_id as a UUID
    except ValueError:
        logger.warning(f"Invalid session_id provided: {session_id}. Generating new UUID.")
        session_id = str(uuid4())

    session_data = SessionData(session_id=session_id)
    await backend.create(UUID(session_id), session_data)
    logger.info(f"Created or reused session: {session_id}")

    response = JSONResponse(content={"message": "Session created or reused", "session_id": session_id})
    cookie.attach_to_response(response=response, session_id=UUID(session_id))
    return response

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), request: Request = None):
    """Upload endpoint that handles session management and calls the file processing function."""
    try:
        # Try to get existing session
        session_data = await get_session_data(request)
    except HTTPException as e:
        if e.status_code == 403:
            logger.info("No valid session found, creating a new one")
            session_id = str(uuid4())
            session_data = SessionData(session_id=session_id)
            await backend.create(UUID(session_id), session_data)
            logger.info(f"Created new session: {session_id}")
            # Set the session cookie explicitly
            response = JSONResponse(content={"message": "Session created", "session_id": session_id})
            cookie.attach_to_response(response=response, session_id=UUID(session_id))
            # Update the request's cookies for the subsequent process_uploaded_file call
            request.cookies["session_cookie"] = str(session_id)
        else:
            logger.error(f"Session error: {str(e)}", exc_info=True)
            raise

    try:
        # Call the main processing function
        result = await process_uploaded_file(file, session_data, request)
        return result
    except Exception as e:
        logger.error(f"Upload error in process_uploaded_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")



# @router.post("/generate-sql")
# async def generate_sql(request_data: QueryRequest, session_data: SessionData = Depends(get_session_data)):
#     """Endpoint to handle SQL query generation and execution."""
#     if (session_data.cleaned_df is None or 
#         session_data.cleaned_df.empty or 
#         not session_data.db_path):
#         logger.error(f"Invalid session data: cleaned_df={session_data.cleaned_df is None}, db_path={session_data.db_path}")
#         raise HTTPException(status_code=400, detail="No valid preprocessed data or database path available. Please upload a file first.")
    
#     return await process_sql_query(request_data, session_data)


@router.get("/test-sql")
async def test_sql_query(query: str = Query(..., description="SQL SELECT query")):
    if not query.upper().strip().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT queries allowed")
    try:
        conn = duckdb.connect(app_state["db_path"])
        df = conn.execute(query).fetch_df()
        conn.close()
        df = df.replace([float('inf'), float('-inf')], None).fillna(None)
        return {
            "data": df.to_dict(orient="records"),
            "row_count": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema-info")
async def get_schema_info():
    if app_state["cleaned_df"] is None:
        raise HTTPException(status_code=400, detail="No preprocessed data found")
    df = app_state["cleaned_df"]
    return {
        "table_name": app_state["table_name"],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "sample_data": df.head(3).to_dict(orient="records")
    }

# Endpoint to test session validity
@router.get("/test-session")
async def test_session(session_data: SessionData = Depends(get_session_data)):

    return {
        "message": "Session is valid",
        "session_id": session_data.session_id
    }

