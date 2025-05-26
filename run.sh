
#!/bin/bash

# Driver Drowsiness Detection System - Run Script
# Automated setup and execution script for different environments

set -e  # Exit on error

echo "🚗 Driver Drowsiness Detection System Setup"
echo "==========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check camera availability
check_camera() {
    echo "📹 Checking camera availability..."
    if ls /dev/video* >/dev/null 2>&1; then
        echo "✅ Camera devices found: $(ls /dev/video*)"
        return 0
    else
        echo "❌ No camera devices found in /dev/video*"
        return 1
    fi
}

# Function to setup Python environment
setup_python_env() {
    echo "🐍 Setting up Python environment..."
    
    if ! command_exists python3; then
        echo "❌ Python 3 is not installed. Please install Python 3.8+"
        exit 1
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
    echo "✅ Python environment setup complete"
}

# Function to download required models
download_models() {
    echo "📥 Downloading required models..."
    
    if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
        echo "Downloading dlib face landmarks predictor..."
        wget -q --show-progress http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        bunzip2 shape_predictor_68_face_landmarks.dat.bz2
        echo "✅ Face landmarks model downloaded"
    else
        echo "✅ Face landmarks model already exists"
    fi
}

# Function to run with Docker
run_docker() {
    echo "🐳 Running with Docker..."
    
    if ! command_exists docker; then
        echo "❌ Docker is not installed. Please install Docker."
        exit 1
    fi
    
    if ! command_exists docker-compose; then
        echo "❌ Docker Compose is not installed. Please install Docker Compose."
        exit 1
    fi
    
    # Build and run with Docker Compose
    echo "Building Docker image..."
    docker-compose build
    
    echo "Starting container..."
    docker-compose up -d
    
    echo "✅ System is running in Docker container"
    echo "📊 Dashboard: http://localhost:5000/dashboard"
    echo "📹 Video Feed: http://localhost:5000/video_feed"
    echo "📚 API Docs: http://localhost:5000/api/docs"
    
    # Show logs
    echo ""
    echo "📋 Container logs (press Ctrl+C to stop viewing):"
    docker-compose logs -f
}

# Function to run locally
run_local() {
    echo "💻 Running locally..."
    
    # Setup Python environment
    setup_python_env
    
    # Download models
    download_models
    
    # Check camera
    if ! check_camera; then
        echo "⚠️  Warning: No camera detected. The system may not work properly."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Start the application
    echo "🚀 Starting Driver Drowsiness Detection System..."
    echo "📊 Dashboard: http://localhost:5000/dashboard"
    echo "📹 Video Feed: http://localhost:5000/video_feed"
    echo "📚 API Docs: http://localhost:5000/api/docs"
    echo ""
    echo "Press Ctrl+C to stop the system"
    echo ""
    
    # Activate virtual environment and run
    source venv/bin/activate
    python app.py
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [docker|local|help]"
    echo ""
    echo "Commands:"
    echo "  docker  - Run using Docker (recommended)"
    echo "  local   - Run locally with Python virtual environment"
    echo "  help    - Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 docker   # Run with Docker"
    echo "  $0 local    # Run locally"
}

# Function to cleanup
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    if command_exists docker-compose; then
        docker-compose down >/dev/null 2>&1 || true
    fi
    echo "✅ Cleanup complete"
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Main script logic
case "${1:-docker}" in
    "docker")
        run_docker
        ;;
    "local")
        run_local
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
