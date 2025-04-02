#!/bin/bash
set -e

echo "Starting Obesity Risk API on Render"

# Print Python version
python --version

# Print environment
echo "PYTHONPATH: $PYTHONPATH"
echo "PORT: $PORT"

# List available directories
echo "Available model files:"
find . -name "*.joblib" -o -name "*.pkl" -o -name "*.json" | grep -v "__pycache__"

# Start the API
python -m uvicorn src.api:app --host 0.0.0.0 --port $PORT --log-level debug 