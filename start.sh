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

# Copy model files if they exist in a different location
if [ -f "models/model.joblib" ]; then
  echo "Model files found in models directory"
else
  echo "Warning: Model files not found in expected location"
  find . -name "*.joblib" -type f
fi

# Print Python information
echo "Python version:"
python --version

# Print PYTHONPATH
echo "PYTHONPATH: $PYTHONPATH"

# List available Python modules
pip list

# Start the API server with better error handling
cd src || { echo "src directory not found!"; exit 1; }
echo "Starting uvicorn server..."

# Try to start the main API, but fall back to test API if it fails
if python -c "import api" 2>/dev/null; then
  echo "Main API module loaded successfully, starting server"
  python -m uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}"
else
  echo "Failed to import main API, falling back to test API"
  python -m uvicorn test_api:app --host 0.0.0.0 --port "${PORT:-8000}"
fi 