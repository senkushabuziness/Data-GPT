# services/upload_service.py
from fastapi import UploadFile, Request, HTTPException
from uuid import UUID
import os
import re
import pandas as pd
import io
from pathlib import Path
from logger import logger
from services.validate_create_session import SessionData, backend
from etl.data_cleaning import DataProcessor
from etl.duckdb_loader import load_file_to_duckdb
from logger import logger
from api.states import app_state

async def process_uploaded_file(file: UploadFile, session_data: SessionData, request: Request) -> dict:
    
    try:
        # Validate session_id
        if not session_data.session_id:
            logger.error("Session ID is missing")
            raise HTTPException(status_code=400, detail="Invalid session: session_id is missing")

        # Check file extension
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            logger.error(f"Unsupported file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx) files are supported")

        # Create session-specific upload directory
        session_upload_dir = Path(f"uploads/{session_data.session_id}").as_posix()
        os.makedirs(session_upload_dir, exist_ok=True)
        file_path = Path(session_upload_dir) / file.filename
        content = await file.read()

        # Log content based on file type
        if file.filename.endswith('.csv'):
            logger.info(f"Raw CSV content (first 1000 chars):\n{content.decode('utf-8')[:1000]}")
        else:
            logger.info(f"Excel file uploaded: {file.filename}, size: {len(content)} bytes")

        # Validate file content and read into DataFrame
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content), na_values=['NA', 'N/A', '-', 'TBD', ''], keep_default_na=False)
            else:
                df = pd.read_excel(io.BytesIO(content), sheet_name=0, na_values=['NA', 'N/A', '-', 'TBD', ''], keep_default_na=False)
            logger.info(f"Input file sample:\n{df.head().to_string()}")
            if df.empty or len(df.columns) < 2:
                logger.error("File is empty or has insufficient columns")
                raise ValueError("File is empty or has insufficient columns")
            df = df.dropna(how='all')
            logger.info(f"After empty row removal, DataFrame shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to parse file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")

        with open(file_path, "wb") as f:
            f.write(content)

        # Generate table name
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', Path(file.filename).stem).strip('_')
        if not table_name or not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', table_name):
            table_name = f"uploaded_table_{session_data.session_id}"
            logger.warning(f"Invalid table name derived; using: {table_name}")

        # Process with LLM
        processor = DataProcessor(session_id=session_data.session_id)
        llm_result, cleaned_df = processor.preprocess_with_llm(df, table_name)
        if cleaned_df.empty:
            logger.error("LLM processing resulted in empty DataFrame")
            raise HTTPException(status_code=500, detail="LLM processing failed: Invalid output DataFrame")

        logger.info(f"LLM processed DataFrame shape: {cleaned_df.shape}, columns: {list(cleaned_df.columns)}")

        # Save preprocessed file as CSV
        preprocessed_file_path = Path(session_upload_dir) / f"preprocessed_{Path(file.filename).stem}.csv"
        cleaned_df.to_csv(preprocessed_file_path, index=False)
        logger.info(f"Cleaned DataFrame saved to {preprocessed_file_path}")

        # Store cleaned_df in app_state instead of session_data
        app_state[session_data.session_id] = {
            "cleaned_df": cleaned_df,
            "db_path": processor.db_path,
            "llm_result": llm_result,
            "uploaded_file_path": preprocessed_file_path.as_posix(),
            "table_name": table_name
        }

        # Set session_data.db_path before calling load_file_to_duckdb
        session_data.db_path = processor.db_path
        session_data.cleaned_df = cleaned_df  # Also set cleaned_df for duckdb_loader
        session_data.llm_result = llm_result  # Set llm_result for duckdb_loader

        # Load to DuckDB
        try:
            cleaned_df, llm_result = load_file_to_duckdb(preprocessed_file_path.as_posix(), table_name, session_data)
        except ValueError as e:
            logger.error(f"DuckDB loading failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"DuckDB loading failed: {str(e)}")

        # Update session data (without cleaned_df)
        session_data.uploaded_file_path = preprocessed_file_path.as_posix()
        session_data.table_name = table_name
        await backend.update(UUID(session_data.session_id), session_data)

        return {
            "message": f"File {file.filename} uploaded, processed, and saved to DuckDB successfully",
            "headers": llm_result['cleaned_data']['columns'],
            "row_count": len(cleaned_df),
            "table_name": table_name,
            "data_quality_issues": llm_result.get('data_quality_issues', []),
            "recommendations": llm_result.get('recommendations', []),
            "schema_info": llm_result.get('schema', {}),
            "dimension_tables": llm_result.get('dimension_tables', []),
            "sample_queries": llm_result.get('sample_queries', {})
        }

    except Exception as e:
        logger.error(f"Upload error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")