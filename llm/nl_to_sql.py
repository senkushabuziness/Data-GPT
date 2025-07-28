import json
import requests
from utils.config import OLLAMA_MODEL, OLLAMA_URL
import re
from groq import Groq
from logger import logger  # Import logger from your existing setup
from api.states import app_state  # Import app_state to access session data

def generate_sql_from_prompt(prompt: str) -> dict:

    try:
        # Initialize Groq client
        client = Groq()


        
        table_name =f"062363d8-0158-416e-8195-4fbfa1db1f69.4ee2626f-9464-4023-9e7d-b6c10c2bc22a_preprocessed_Conta_template"
        # Enhance prompt with table_name context
        enhanced_prompt = f"""
        Using the DuckDB table '{table_name}', convert the following natural language question into a precise, optimized SQL query: 
        {prompt}
        """



        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert SQL data analyst specializing in DuckDB. Your task is to convert natural language questions into precise, optimized SQL queries. You must respond ONLY with valid JSON format containing the keys: 'sql_query' (the generated SQL), 'summary' (a brief explanation), and 'interpretation' (data context). Do not include additional text, explanations, or markdown formatting outside the JSON."""
                },
                {
                    "role": "user",
                    "content": enhanced_prompt
                }
            ],
            temperature=0.1,
            max_completion_tokens=2048,
            top_p=0.95,
            stream=False,
            stop=None,
        )

        raw_response = completion.choices[0].message.content.strip()
        # logger.info(f"Raw SQL LLM response for session {session_id}: {raw_response}")

        # Extract JSON
        json_start = raw_response.find('{')
        json_end = raw_response.rfind('}')
        
        if json_start != -1 and json_end != -1:
            json_str = raw_response[json_start:json_end + 1]
            json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
            
            try:
                response_dict = json.loads(json_str)
                # Ensure required keys are present
                required_keys = ["sql_query", "summary", "interpretation"]
                if not all(key in response_dict for key in required_keys):
                    missing_keys = [key for key in required_keys if key not in response_dict]
                    logger.warning(f"Missing required keys in JSON response: {missing_keys}")
                    return {
                        "error": f"Missing required keys: {missing_keys}",
                        "sql_query": "",
                        "raw_response": raw_response
                    }
                return response_dict
            except json.JSONDecodeError as e:
                # logger.error(f"JSON decode error for session {session_id}: {str(e)}", exc_info=True)
                return {
                    "error": f"Invalid JSON in response: {str(e)}",
                    "sql_query": "",
                    "raw_response": raw_response
                }
        else:
            # logger.error(f"No JSON object found in response for session {session_id}: {raw_response}")
            return {
                "error": "No JSON object found in response",
                "sql_query": "",
                "raw_response": raw_response
            }
        
    except Exception as e:
        # logger.error(f"Error generating SQL for session {session_id}: {str(e)}", exc_info=True)
        return {
            "error": f"Request error: {str(e)}",
            "sql_query": "",
            "summary": "",
            "interpretation": "",
            "raw_response": str(e)
        }