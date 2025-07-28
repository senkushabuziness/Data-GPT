from typing import Optional
from fastapi import APIRouter, HTTPException, File, UploadFile, Query, Form, Depends, Request, Body
from fastapi.responses import JSONResponse
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
from uuid import UUID, uuid4
import pandas as pd
import os
import duckdb
import logging
from groq import Groq 
from etl.data_cleaning import DataProcessor
from models.response import UploadResponse
from services.upload_service import process_uploaded_file
from services.validate_create_session import SessionData, get_session_data, cookie, backend
from api.states import app_state
from logger import logger

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
from pydantic import BaseModel

class SessionRequest(BaseModel):
    session_id: Optional[str] = None
    user_id: Optional[str] = None

@router.post("/create-session")
async def create_session(request: SessionRequest = Body(...)):
    session_id = request.session_id
    user_id = request.user_id
    if not session_id:
        session_id = str(uuid4())
    if not user_id:
        user_id = str(uuid4())
    try:
        UUID(session_id)
    except ValueError:
        logger.warning(f"Invalid session_id provided: {session_id}. Generating new UUID.")
        session_id = str(uuid4())
    try:
        UUID(user_id)
    except ValueError:
        logger.warning(f"Invalid user_id provided: {user_id}. Generating new UUID.")
        user_id = str(uuid4())

    session_data = SessionData(session_id=session_id, user_id=user_id)  
    await backend.create(UUID(session_id), session_data)
    logger.info(f"Created or reused session: {session_id} for user: {user_id}")

    response = JSONResponse(content={"message": "Session created or reused", "session_id": session_id, "user_id": user_id})
    cookie.attach_to_response(response=response, session_id=UUID(session_id))
    return response

@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), session_id: str = Form(...)):
    try:
        try:
            UUID(session_id)
        except ValueError:
            logger.warning(f"Invalid session_id provided: {session_id}. Generating new UUID.")
            session_id = str(uuid4())

        session_data = await backend.read(UUID(session_id))
        if not session_data:
            session_data = SessionData(session_id=session_id, user_id=str(uuid4()))
            await backend.create(UUID(session_id), session_data)
            logger.info(f"Created new session: {session_id}")

        result = await process_uploaded_file(file, session_data, None)
        return result
    except Exception as e:
        logger.error(f"Upload error in process_uploaded_file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/test-sql")
async def test_sql_query(query: str = Query(..., description="SQL SELECT query")):
    if not query.upper().strip().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT queries allowed")
    try:
        processor = DataProcessor(session_id=str(uuid4()), user_id="system")
        df = processor.query_duckdb(query)
        df = df.replace([float('inf'), float('-inf')], None).fillna(None)
        return {
            "data": df.to_dict(orient="records"),
            "row_count": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/schema-info")
async def get_schema_info(session_data: SessionData = Depends(get_session_data)):
    try:
        processor = DataProcessor(session_data.session_id, session_data.user_id)
        schema_df = processor.get_table_schema("uploads")
        return {
            "table_name": "Uploads",
            "columns": schema_df['column_name'].tolist(),
            "dtypes": {row['column_name']: row['column_type'] for _, row in schema_df.iterrows()},
            "row_count": processor.query_duckdb(f"SELECT COUNT(*) FROM Uploads WHERE user_id = '{session_data.user_id}'").iloc[0, 0]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/test-session")
async def test_session(session_data: SessionData = Depends(get_session_data)):
    return {
        "message": "Session is valid",
        "session_id": session_data.session_id,
        "user_id": session_data.user_id
    }