# Makefile for Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems

.PHONY: help install test clean run-pipeline run-app docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install        - Install dependencies"
	@echo "  test           - Run all tests"
	@echo "  test-coverage  - Run tests with coverage"
	@echo "  clean          - Clean generated files"
	@echo "  run-pipeline   - Run complete pipeline"
	@echo "  run-app        - Start Streamlit app"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run Docker container"
	@echo "  format         - Format code with black"
	@echo "  lint           - Lint code with flake8"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf logs/*.log
	rm -rf visualizations/*.png
	rm -rf visualizations/*.jpg
	rm -rf visualizations/*.pdf
	rm -rf models/*.h5
	rm -rf models/*.pkl

# Run complete pipeline
run-pipeline:
	@echo "Running complete pipeline..."
	python run_pipeline.py --step all

# Run individual pipeline steps
run-data:
	@echo "Running data preprocessing..."
	python run_pipeline.py --step data

run-train:
	@echo "Running model training..."
	python run_pipeline.py --step train

run-adversarial:
	@echo "Running adversarial testing..."
	python run_pipeline.py --step adversarial

run-failsafe:
	@echo "Running fail-safe evaluation..."
	python run_pipeline.py --step failsafe

run-explain:
	@echo "Running explainability analysis..."
	python run_pipeline.py --step explain

# Start Streamlit app
run-app:
	@echo "Starting Streamlit app..."
	cd app && streamlit run app.py

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -f docker/Dockerfile -t robust-vision-failsafes .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8501:8501 robust-vision-failsafes

# Code formatting and linting
format:
	@echo "Formatting code with black..."
	black src/ tests/ app/ run_pipeline.py

lint:
	@echo "Linting code with flake8..."
	flake8 src/ tests/ app/ run_pipeline.py

# Development setup
dev-setup: install format lint
	@echo "Development environment setup complete!"

# Quick test run
quick-test:
	@echo "Running quick tests..."
	python -c "from src.data_preprocessing import DataPreprocessor; print('Data preprocessing OK')"
	python -c "from src.model_training import VehicleClassifier; print('Model training OK')"
	python -c "from src.failsafe_handler import FailSafeHandler; print('Fail-safe handler OK')"
	python -c "from src.explainability import ModelExplainer; print('Explainability OK')"
	@echo "All modules imported successfully!"

# Generate documentation
docs:
	@echo "Generating documentation..."
	python -c "import src; help(src)" > docs/module_docs.txt
	@echo "Documentation generated in docs/module_docs.txt"

# Performance benchmark
benchmark:
	@echo "Running performance benchmark..."
	python -c "
import time
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.model_training import VehicleClassifier

# Benchmark data preprocessing
start_time = time.time()
preprocessor = DataPreprocessor()
train_data, val_data, test_data = preprocessor.load_cifar10_data()
preprocessing_time = time.time() - start_time
print(f'Data preprocessing time: {preprocessing_time:.2f} seconds')

# Benchmark model training (with small dataset)
start_time = time.time()
classifier = VehicleClassifier()
# Note: This would require actual training data
training_time = time.time() - start_time
print(f'Model training time: {training_time:.2f} seconds')
"
