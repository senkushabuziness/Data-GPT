from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd

class UploadResponse(BaseModel):
    message: str
    headers: List[str]
    row_count: int
    table_name: str
    data_quality_issues: List[str]
    recommendations: List[str]
    schema_info: Dict[str, Any]
    dimension_tables: List[Any]
    sample_queries: Dict[str, Any]

class QueryRequest(BaseModel):
    input_query: str