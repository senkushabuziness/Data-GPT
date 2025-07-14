from pydantic import BaseModel
from fastapi_sessions.backends.implementations import InMemoryBackend
from fastapi_sessions.frontends.implementations import SessionCookie, CookieParameters
from fastapi_sessions.session_verifier import SessionVerifier
from fastapi import HTTPException, Request
from uuid import UUID
from typing import Dict, Optional, Any
import pandas as pd


class SessionData(BaseModel):
    session_id: str
    cleaned_df: Optional[pd.DataFrame] = None
    db_path: Optional[str] = None
    llm_result: Optional[Dict[str, Any]] = None
    uploaded_file_path: Optional[str] = None
    table_name: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

cookie = SessionCookie(
    cookie_name="session_cookie",
    identifier="general_verifier",
    auto_error=True,
    secret_key="YOUR_SECRET_KEY",  # Replace with a secure key
    cookie_params=CookieParameters(),
)

backend = InMemoryBackend[UUID, SessionData]()

class BasicVerifier(SessionVerifier[UUID, SessionData]):
    def __init__(self, *, identifier: str, auto_error: bool):
        self._identifier = identifier
        self._auto_error = auto_error

    @property
    def identifier(self):
        return self._identifier

    @property
    def auto_error(self):
        return self._auto_error

    @property
    def backend(self):
        return backend

    async def verify_session(self, session_id: UUID) -> SessionData | None:
        return await backend.read(session_id)

verifier = BasicVerifier(identifier="general_verifier", auto_error=True)

async def get_session_data(request: Request = None):
    try:
        session_id = cookie.get_session_id_from_request(request)
        if not session_id:
            raise HTTPException(status_code=403, detail="No session found")
        session_data = await backend.read(session_id)
        if not session_data:
            raise HTTPException(status_code=403, detail="Invalid session")
        return session_data
    except Exception as e:
        raise HTTPException(status_code=403, detail=f"Session error: {str(e)}")