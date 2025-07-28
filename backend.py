from fastapi import FastAPI
from pyngrok import ngrok
import uvicorn
from api.routes import router
from dotenv import load_dotenv
import os
from logger import logger
from api.states import app_state

load_dotenv()

app = FastAPI(title="Financial Data Processor API")
app.include_router(router)

@app.get("/")
def read_root():
    logger.info("Root endpoint was accessed.")
    return {"message": "Hello World"}


host = os.getenv("HOST", "localhost")
port = int(os.getenv("PORT", 8001))

if __name__ == "__main__":
    uvicorn.run("backend:app", host=host, port=port, reload=True)