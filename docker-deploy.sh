#!/bin/bash

# Make script exit if any command fails
set -e

# Function for development mode
run_dev() {
    echo "Starting in development mode..."
    docker-compose up --build -d
    echo "Development environment is running"
    echo "Frontend: http://localhost:3000"
    echo "Backend API: http://localhost:8000"
    echo "API Docs: http://localhost:8000/docs"
}

# Function for production mode
run_prod() {
    echo "Starting in production mode..."
    docker-compose -f docker-compose.prod.yml up --build -d
    echo "Production environment is running"
    echo "Application is available at: http://localhost"
    echo "API Docs: http://localhost/docs"
}

# Function to stop containers
stop() {
    if [ "$1" = "prod" ]; then
        echo "Stopping production environment..."
        docker-compose -f docker-compose.prod.yml down
    else
        echo "Stopping development environment..."
        docker-compose down
    fi
    echo "Environment stopped"
}

# Function to check logs
view_logs() {
    if [ "$1" = "frontend" ]; then
        echo "Showing frontend logs..."
        docker-compose logs -f frontend
    elif [ "$1" = "backend" ]; then
        echo "Showing backend logs..."
        docker-compose logs -f backend
    elif [ "$1" = "nginx" ]; then
        echo "Showing nginx logs..."
        docker-compose logs -f nginx
    else
        echo "Showing all logs..."
        docker-compose logs -f
    fi
}

# Check command
case "$1" in
    dev)
        run_dev
        ;;
    prod)
        run_prod
        ;;
    stop)
        stop $2
        ;;
    logs)
        view_logs $2
        ;;
    *)
        echo "Usage: $0 {dev|prod|stop|logs}"
        echo "  dev - Run in development mode"
        echo "  prod - Run in production mode"
        echo "  stop [prod] - Stop containers (specify 'prod' for production mode)"
        echo "  logs [frontend|backend|nginx] - View logs (specify component or all)"
        exit 1
esac

exit 0 