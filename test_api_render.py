from fastapi import FastAPI
import logging
import os
import sys
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Print debugging information
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")
logger.info(f"PYTHONPATH: {sys.path}")

try:
    logger.info(f"src directory contents: {os.listdir('./src')}")
except Exception as e:
    logger.error(f"Error listing src directory: {e}")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Obesity Risk API Test is running"}

@app.get("/health")
async def health():
    return {"status": "ok", "message": "Test API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000))) 