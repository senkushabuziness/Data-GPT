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
from etl.duckdb_loader import MultiUserDuckDBManager
from api.states import app_state
import json

async def process_uploaded_file(file: UploadFile, session_data: SessionData, request: Request) -> dict:
    try:
        # Validate session_id
        logger.info(f"Received session_id: {session_data.session_id}, user_id: {session_data.user_id}")
        if not session_data.session_id:
            logger.error("Session ID is missing")
            raise HTTPException(status_code=400, detail="Invalid session: session_id is missing")

        # Check file extension
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            logger.error(f"Unsupported file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only CSV and Excel (.xlsx) files are supported")

        # Create temporary upload directory
        temp_upload_dir = Path(f"/tmp/uploads/{session_data.session_id}").as_posix()
        os.makedirs(temp_upload_dir, exist_ok=True)
        file_path = Path(temp_upload_dir) / file.filename
        content = await file.read()

        # Log content based on file type
        if file.filename.endswith('.csv'):
            logger.info(f"Raw CSV content (first 1000 chars):\n{content.decode('utf-8')[:1000]}")
        elif file.filename.endswith('.xlsx'):
            logger.info(f"Excel file uploaded: {file.filename}, size: {len(content)} bytes")

        # Read into DataFrame
        try:
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content), na_values=['NA', 'N/A', '-', 'TBD', ''], keep_default_na=False)
            else:
                df = pd.read_excel(io.BytesIO(content), sheet_name=0, na_values=['NA', 'N/A', '-', 'TBD', ''], keep_default_na=False)
            # Add missing year columns for balance sheets
            if 'Balance Sheet' in df.to_string().lower():
                expected_years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
                for year in expected_years:
                    if year not in df.columns:
                        df[year] = pd.NA
            logger.info(f"Input file sample:\n{df.head().to_string()}")
        except Exception as e:
            logger.error(f"Failed to parse file: {str(e)}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Invalid file: {str(e)}")

        # Initialize DataProcessor with GCS enabled
        processor = DataProcessor(
            session_id=session_data.session_id, 
            user_id=session_data.user_id or "temp_validation",
            gcs_enabled=True
        )

        # Validate against templates
        is_valid, validation_message = processor.validate_against_templates(df)
        if not is_valid:
            logger.error(f"Template validation failed: {validation_message}")
            raise HTTPException(status_code=400, detail=validation_message)

        # Generate table name
        table_name = re.sub(r'[^a-zA-Z0-9_]', '_', Path(file.filename).stem).strip('_')
        if not table_name or not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', table_name):
            table_name = f"uploaded_file_{session_data.session_id}"
            logger.warning(f"Invalid table name derived; using: {table_name}")

        # Process with LLM
        db_manager = MultiUserDuckDBManager(db_base_path="db")
        session_data.user_id = db_manager.get_user_id_from_postgres(session_data.session_id)
        logger.info(f"Updated user_id from postgres: {session_data.user_id}")
        llm_result, cleaned_df = processor.preprocess_with_llm(df, table_name)
        if cleaned_df.empty:
            logger.error("LLM processing resulted in empty DataFrame")
            raise HTTPException(status_code=500, detail="LLM processing failed: Invalid output DataFrame")

        logger.info(f"LLM processed DataFrame shape: {cleaned_df.shape}, columns: {list(cleaned_df.columns)}")

        # Save preprocessed file as CSV
        preprocessed_file_path = Path(temp_upload_dir) / f"preprocessed_{Path(file.filename).stem}.csv"
        cleaned_df.to_csv(preprocessed_file_path, index=False)
        logger.info(f"Cleaned DataFrame saved to {preprocessed_file_path}")

        # Store in app_state
        logger.info(f"Updating app_state for session: {session_data.session_id}")
        # app_state[session_data.session_id] = {
        #     "db_path": processor.db_path.as_posix() if isinstance(processor.db_path, Path) else str(processor.db_path),
        #     "uploaded_file_path": preprocessed_file_path.as_posix(),
        #     "table_name": table_name
        # }
        if session_data.session_id not in app_state:
            logger.warning(f"Session {session_data.session_id} not found in app_state, creating it now")
            app_state[session_data.session_id] = {
                "db_path": processor.db_path.as_posix() if isinstance(processor.db_path, Path) else str(processor.db_path),
                "uploaded_file_path": preprocessed_file_path.as_posix(),
                "table_name": table_name
    }
        session_data.db_path = processor.db_path
        session_data.cleaned_df = cleaned_df
        session_data.llm_result = llm_result

        # Load to DuckDB
        cleaned_df, llm_result = await db_manager.load_file_to_duckdb(preprocessed_file_path.as_posix(), session_data)

        # Update session data
        session_data.uploaded_file_path = preprocessed_file_path.as_posix()
        session_data.table_name = table_name
        await backend.update(UUID(session_data.session_id), session_data)
        logger.info(f"Session data updated for session_id: {session_data.session_id}")
        logger.info(f"app_state variable exists: {locals().get('app_state', 'NOT_FOUND')}")
        logger.info(f"session_data.session_id: {session_data.session_id}")
        logger.info(f"app_state content: {app_state}")
        logger.info(f"app_state.get result: {app_state.get(session_data.session_id, {})}")
        # Prepare response with app_state as a dictionary keyed by session_id
        response_dict = {
            "message": f"File {file.filename} uploaded, validated against templates, processed, and saved to DuckDB successfully",
            "headers": llm_result['cleaned_data']['columns'],
            "row_count": len(cleaned_df),
            "table_name": table_name,
            "app_state":  {session_data.session_id: app_state.get(session_data.session_id, {})},
            # "app_state_str": json.dumps({session_data.session_id: app_state.get(session_data.session_id, {})}, indent=2),
            "data_quality_issues": llm_result.get('data_quality_report', {}).get('issues', []),
            "recommendations": llm_result.get('data_quality_report', {}).get('recommendations', []),
            "schema_info": llm_result.get('schema', {}),
            "dimension_tables": llm_result.get('dimension_tables', []),
            "sample_queries": llm_result.get('sample_queries', {}),
            "validation_message": validation_message
        }
        logger.info(f"Final Response being sent: {response_dict}")

        if 'app_state' in response_dict:
            logger.info(f"__++++==========\nresponse_dict['app_state']: {response_dict['app_state']}")

        logger.info(f"Response being sent: {response_dict}")
        return response_dict
    except HTTPException as e:
        logger.error(f"Upload error: {str(e.detail)}", exc_info=True)
        raise e
    except Exception as e:
        logger.error(f"Unexpected error during file processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")