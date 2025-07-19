import pandas as pd
import duckdb
import chainlit as cl
from llm.nl_to_sql import generate_sql_from_prompt
from api.states import app_state

async def handle_sql_message(msg: cl.Message):
    session_id = cl.context.session.id
    # Retrieve session data from app_state
    print(f"Session ID: {session_id}")
    session_data = app_state.get(session_id)
    if not session_data or "cleaned_df" not in session_data or "db_path" not in session_data or "table_name" not in session_data:
        await cl.Message("⚠️ Please upload a file first.").send()
        return

    df = session_data["cleaned_df"]
    db_path = session_data["db_path"]
    table_name = session_data["table_name"]

    user_question = msg.content.strip()
    prompt = f"""
You are an expert data analyst.
Your ONLY job is to generate the DuckDB SQL query to answer the user's question using the table named '{table_name}'.

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
        # Connect to the specific DuckDB database for the session
        con = duckdb.connect(db_path)
        try:
            result = con.execute(sql_query).fetchdf()
            await cl.Message(f"✅ **Result:**\n\n{result.head(10).to_markdown()}").send()
        finally:
            con.close()  # Ensure the connection is closed
    except Exception as e:
        await cl.Message(f"❌ **SQL Error:** {str(e)}").send()