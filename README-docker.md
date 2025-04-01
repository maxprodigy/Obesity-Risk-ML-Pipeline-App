# Running the Obesity Risk Prediction System with Docker

This guide explains how to run the Obesity Risk Prediction System using Docker and Docker Compose.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ML-Pipeline-Obesity-Risk
   ```

2. Make sure your model files are in the `models` directory:
   - `model.joblib`
   - `scaler.joblib`

3. Start the application using Docker Compose:
   ```bash
   # On Windows
   docker-compose up -d
   
   # On Linux/Mac (you might need to use sudo)
   docker-compose up -d
   ```

4. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

5. To stop the application:
   ```bash
   docker-compose down
   ```

## Docker Components

The system consists of two main Docker containers:

1. **Backend (FastAPI)**: 
   - Runs on port 8000
   - Provides the ML model and prediction API
   - Manages data processing and model training

2. **Frontend (React)**:
   - Runs on port 3000 (port 80 inside the container, mapped to 3000)
   - Provides the user interface for data input and visualization
   - Communicates with the backend API

## Troubleshooting

### Common Issues

1. **Model files not found**: 
   Make sure your `models` directory contains the trained model files:
   - `model.joblib`
   - `scaler.joblib`

2. **Import errors**:
   The Docker environment is configured to match your local development environment's import structure. If you see import errors, check:
   - The `PYTHONPATH` environment variable in `docker-compose.yml`
   - The directory structure inside the container

3. **Container fails to start**:
   ```bash
   # Check what's happening in the container
   docker-compose logs backend
   ```

4. **Docker Desktop not running**:
   If you see errors like "Cannot connect to Docker daemon", make sure Docker Desktop is running on your system.

## Building Individual Components

If you need to build or rebuild a specific component:

```bash
# Build and start only the backend
docker-compose up -d --build backend

# Build and start only the frontend
docker-compose up -d --build frontend
```

## Volumes and Persistence

The Docker setup includes volume mappings for:
- `/models`: To persist trained models between container restarts
- `/logs`: To persist application logs for debugging
- `/data`: To persist training and test data

## Production Deployment

For production deployment, use the production Docker Compose file:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

This will:
- Use proper restart policies for containers
- Set up Nginx as a reverse proxy
- Configure for production environment

## Additional Commands

1. **Check container status**:
   ```bash
   docker-compose ps
   ```

2. **View container logs**:
   ```bash
   # All containers
   docker-compose logs

   # Specific container
   docker-compose logs backend
   docker-compose logs frontend
   ```

3. **Restart containers**:
   ```bash
   docker-compose restart backend
   docker-compose restart frontend
   ```

4. **Rebuild containers after code changes**:
   ```bash
   docker-compose up -d --build
   ```

## Additional Information

- **API Documentation**: Available at http://localhost:8000/docs when the backend is running
- **Health Check**: The backend provides a `/health` endpoint for monitoring
- **Data Persistence**: Make sure the `data` directory contains the required training datasets 