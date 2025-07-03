# charts/chart_generator.py

import pandas as pd
import requests
import chainlit as cl
from utils.config import OLLAMA_MODEL, OLLAMA_URL

def generate_visualization_recommendation(df: pd.DataFrame, user_question: str) -> dict:
    data_info = f"""
Shape: {df.shape}
Columns:
{df.dtypes.to_string()}
Sample:
{df.head().to_string()}
"""
    viz_prompt = f"""
Recommend a Plotly chart for: "{user_question}"
Data Info:
{data_info}
"""
    response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": viz_prompt, "stream": False})
    return response.json()["response"].strip()

@cl.on_message
async def chart_handler(msg: cl.Message):
    user_question = msg.content
    df = pd.DataFrame({"category": ["A", "B", "C"], "value": [10, 20, 15]})  # Placeholder example

    viz = generate_visualization_recommendation(df, user_question)
    await cl.Message(f"Chart Recommendation:\n{viz}").send()
