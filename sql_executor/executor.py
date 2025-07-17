# sql_executor/executor.py

import pandas as pd
import duckdb
import requests
import chainlit as cl
from llm.nl_to_sql import generate_sql_from_prompt

con = duckdb.connect()
session_dfs = {}

async def handle_file_uploadsss(msg: cl.Message):
    session_id = cl.context.session.id
    file = msg.elements[0]
    df = pd.read_csv(file.path)
    session_dfs[session_id] = df
    con.register("df", df)
    await cl.Message("✅ File uploaded. Now ask your data questions.").send()

async def handle_sql_message(msg: cl.Message):
    session_id = cl.context.session.id
    df = session_dfs.get(session_id)
    if df is None:
        await cl.Message("⚠️ Please upload a CSV file first.").send()
        return

    user_question = msg.content.strip()
    prompt = f"""
You are an expert data analyst.
Your ONLY job is to generate the DuckDB SQL query to answer the user's question using the table named 'df'.

IMPORTANT:
- Do NOT provide any explanations or clarifications.
- Do NOT guess file names or mention dataset paths.
- Only return the SQL code starting with SELECT or other relevant SQL keywords.

Question:
{user_question}

Table schema:
{df.dtypes.to_string()}

Sample rows:
{df.head(5).to_markdown()}
"""


    sql_query = generate_sql_from_prompt(prompt)
    await cl.Message(f"📝 **Generated SQL:**\n```\n{sql_query}\n```").send()

    try:
        result = con.execute(sql_query).fetchdf()
        await cl.Message(f"✅ **Result:**\n\n{result.head(10).to_markdown()}").send()
    except Exception as e:
        await cl.Message(f"❌ **SQL Error:** {str(e)}").send()