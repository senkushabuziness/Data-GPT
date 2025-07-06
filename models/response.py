from pydantic import BaseModel
from typing import List, Dict

class UploadResponse(BaseModel):
    message: str
    headers: list
    row_count: int
    table_name: str
    data_quality_issues: list
    recommendations: list
    schema_info: dict
    dimension_tables: list
    sample_queries: dict

class QueryRequest(BaseModel):
    input_query: str