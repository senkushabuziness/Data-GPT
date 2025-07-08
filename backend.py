from fastapi import FastAPI
from pyngrok import ngrok
import uvicorn
from api.routes import router
from dotenv import load_dotenv
import os
from logger import logger

load_dotenv()

app = FastAPI(title="Financial Data Processor API")
app.include_router(router)
# Initialize app state
app.state.current_file_path = None
app.state.current_db_table_path = None

@app.get("/")
def read_root():
    logger.info("Root endpoint was accessed.")
    return {"message": "Hello World"}

host = os.getenv("HOST", "localhost")
port = int(os.getenv("PORT", 8000))   

if __name__ == "__main__":
    uvicorn.run("backend:app", host=host, port=port, reload=True)
