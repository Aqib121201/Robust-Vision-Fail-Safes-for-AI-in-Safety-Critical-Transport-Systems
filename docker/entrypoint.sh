#!/bin/bash

# Entrypoint script for Robust Vision Fail-Safes Docker container

set -e

echo "🚀 Starting Robust Vision Fail-Safes System..."

# Check if model exists, if not run training
if [ ! -f "models/vehicle_classifier.h5" ]; then
    echo "📊 Model not found. Running training pipeline..."
    python run_pipeline.py --step train
fi

# Check if we should run the full pipeline
if [ "$RUN_PIPELINE" = "true" ]; then
    echo "🔄 Running complete pipeline..."
    python run_pipeline.py --step all
fi

# Start the application
echo "🌐 Starting Streamlit application..."
exec streamlit run app/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
