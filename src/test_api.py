from fastapi import FastAPI
import uvicorn
import os
import sys

# Print debugging information
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print(f"Python version: {sys.version}")
print(f"Available files in src: {os.listdir('.')}")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Obesity Risk API Test is running"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 