#!/bin/bash
set -o errexit

# Create necessary directories
mkdir -p models logs data/train

# Start the API server
cd src
uvicorn api:app --host 0.0.0.0 --port $PORT 