# utils/config.py

import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")


CHAINLIT_AUTH_PROVIDER = os.getenv("CHAINLIT_AUTH_PROVIDER")
CHAINLIT_AUTH_SECRET = os.getenv("CHAINLIT_AUTH_SECRET")

# Google OAuth
GOOGLE_CLIENT_ID = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")

# GitHub OAuth
GITHUB_CLIENT_ID = os.getenv("OAUTH_GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("OAUTH_GITHUB_CLIENT_SECRET")

# ======================
# 🗄️ DATABASE CONFIG
# ======================

DATABASE_URL = os.getenv("DATABASE_URL")
CHAINLIT_DATA_LAYER = os.getenv("CHAINLIT_DATA_LAYER")


#AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
#AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
#AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
#AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME")
#AZURE_BLOB_ENDPOINT = os.getenv("AZURE_BLOB_ENDPOINT")

BUCKET_NAME = os.getenv("BUCKET_NAME")
APP_GCS_PROJECT_ID = os.getenv("APP_GCS_PROJECT_ID")
APP_GCS_CLIENT_EMAIL = os.getenv("APP_GCS_CLIENT_EMAIL")
APP_GCS_PRIVATE_KEY = os.getenv("APP_GCS_PRIVATE_KEY")


'''
if __name__ == "__main__":
    print("Google Client ID:", GOOGLE_CLIENT_ID)
    print("GitHub Client ID:", GITHUB_CLIENT_ID)
    print("Database URL:", DATABASE_URL)
    print("Azure Blob Endpoint:", AZURE_BLOB_ENDPOINT)
'''


MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "local")  # "local" or "hosted"

LLAMA_HOSTED_URL = os.getenv("LLAMA_HOSTED_URL", "https://ollama-546561582790.asia-south1.run.app/api/chat")
LLAMA_HOSTED_MODEL = os.getenv("LLAMA_HOSTED_MODEL", "llama3.1")
