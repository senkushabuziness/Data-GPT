from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from uuid import uuid4, UUID
import os
import re
import pandas as pd
import io
from pathlib import Path
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

async def process_uploaded_file(file: UploadFile, session_data: 'SessionData', request: Request) -> dict:
    """Process a CSV or Excel file and store it in a session-specific DuckDB database."""
    try:
        # Check file extension
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx) files are supported")

        session_id = session_data.session_id
        session_upload_dir = f"/kaggle/working/uploads/{session_id}"
        os.makedirs(session_upload_dir, exist_ok=True)
        file_path = os.path.join(session_upload_dir, file.filename)
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
            else:  # .xlsx file
                df = pd.read_excel(io.BytesIO(content), sheet_name=0, na_values=['NA', 'N/A', '-', 'TBD', ''], keep_default_na=False)
            
            logger.info(f"Input file sample:\n{df.head().to_string()}")
            if df.empty or len(df.columns) < 2:
                raise ValueError("File is empty or has insufficient columns")
            df = df.dropna(how='all')
            logger.info(f"After empty row removal, DataFrame shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to parse file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")

        with open(file_path, "wb") as f:
            f.write(content)

        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', Path(file.filename).stem).strip('_')
        if not table_name or not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', table_name):
            table_name = f"uploaded_table_{session_id}"
            logger.warning(f"Invalid table name derived; using: {table_name}")

        processor = DataProcessor(session_id=session_id)
        llm_result, cleaned_df = processor.preprocess_with_llm(df, table_name)
        if cleaned_df.empty:
            raise HTTPException(status_code=500, detail="LLM processing failed: Invalid output DataFrame")

        logger.info(f"LLM processed DataFrame shape: {cleaned_df.shape}, columns: {list(cleaned_df.columns)}")
        
        # Save preprocessed file as CSV regardless of input format
        preprocessed_file_path = f"{session_upload_dir}/preprocessed_{Path(file.filename).stem}.csv"
        cleaned_df.to_csv(preprocessed_file_path, index=False)
        logger.info(f"Cleaned DataFrame saved to {preprocessed_file_path}")

        # Always pass the preprocessed CSV file to DuckDB, not the original file
        cleaned_df, llm_result = processor.load_file_to_duckdb(preprocessed_file_path, table_name)

        # Update session data
        session_data.cleaned_df = cleaned_df
        session_data.uploaded_file_path = preprocessed_file_path
        session_data.db_path = processor.db_path
        session_data.table_name = table_name
        await backend.update(UUID(session_id), session_data)

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
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")