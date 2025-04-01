#!/bin/bash
# Use bash strict mode
set -euo pipefail

# Enable debug mode to see commands being executed
set -x

echo "Starting the application..."

# Print directory structure for debugging
echo "Current directory: $(pwd)"
ls -la

# Create necessary directories
mkdir -p models logs data/train

# Check for model files and report status
echo "Checking for model files..."
if [ -f "models/model.joblib" ]; then
  echo "Model files found in models directory"
else
  echo "Warning: Model files not found in expected location"
  echo "Searching for model files..."
  find . -name "*.joblib" -o -name "*.pkl" -o -name "*.json" -type f
fi

# Print Python information
echo "Python version:"
python --version

# Print PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

# List available Python modules
pip list

# Test if we can import key modules
echo "Testing imports..."
python -c "import fastapi" || echo "Failed to import fastapi"
python -c "import uvicorn" || echo "Failed to import uvicorn"
python -c "import pandas" || echo "Failed to import pandas"
python -c "import numpy" || echo "Failed to import numpy"
python -c "import sklearn" || echo "Failed to import sklearn"
python -c "import xgboost" || echo "Failed to import xgboost"

# Start the API server with better error handling
cd src || { echo "src directory not found!"; exit 1; }
echo "Starting uvicorn server..."

# Try to start the main API, with more detailed error reporting
if python -c "import api" 2> import_error.log; then
  echo "Main API module loaded successfully, starting server"
  python -m uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
else
  echo "Failed to import main API. Error details:"
  cat import_error.log
  echo "Directory contents:"
  ls -la
  echo "Falling back to test API"
  if python -c "import test_api" 2> test_import_error.log; then
    python -m uvicorn test_api:app --host 0.0.0.0 --port "${PORT:-8000}"
  else
    echo "Failed to import test_api as well. Error details:"
    cat test_import_error.log
    exit 1
  fi
fi 