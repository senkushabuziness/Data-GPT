# llm/nl_to_sql.py

import requests
from utils.config import OLLAMA_MODEL, OLLAMA_URL

def generate_sql_from_prompt(prompt: str) -> str:
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    sql_query = response.json()["response"].strip()

    if sql_query.startswith(" "):
        sql_query = sql_query.strip("`")
        sql_query = sql_query.replace("sql", "", 1).strip()

    return sql_query
