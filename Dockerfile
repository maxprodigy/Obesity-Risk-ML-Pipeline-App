# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models/ ./models/
COPY data/ ./data/
COPY src/ ./src/
COPY risk_prediction_app.py .

# Copy everything else
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "risk_prediction_app.py", "--server.port=8501", "--server.enableCORS=false"]