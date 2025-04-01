from fastapi import FastAPI
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

# Print debugging information
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print(f"Python version: {sys.version}")
print(f"Available files in src: {os.listdir('.')}")

app = FastAPI()

@app.get("/health")
async def health():
    logger.info("Health check endpoint called on test API")
    return {"status": "ok", "message": "Test API is running"}

@app.get("/")
async def root():
    logger.info("Root endpoint called on test API")
    return {"message": "Obesity Risk Test API is running. This is a fallback API for troubleshooting."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 