# 🚀 Quick Start Guide

## 🎯 Get Started in 5 Minutes

This guide will get you up and running with the **Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems** in just a few minutes.

---

## 📋 Prerequisites

- **Python 3.8+** (3.9 recommended)
- **Git** (for cloning)
- **Docker** (optional, for containerized deployment)

---

## 🚀 Option 1: Local Installation (Recommended)

### Step 1: Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd "Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run the Pipeline
```bash
# Run complete pipeline (data → train → test → analyze)
python run_pipeline.py

# Or run individual steps
python run_pipeline.py --step data      # Data preprocessing
python run_pipeline.py --step train     # Model training
python run_pipeline.py --step adversarial # Adversarial testing
python run_pipeline.py --step failsafe  # Fail-safe evaluation
python run_pipeline.py --step explain   # SHAP analysis
```

### Step 3: Start Web Interface
```bash
# Start the Streamlit app
streamlit run app/app.py

# Open browser to http://localhost:8501
```

---

## 🐳 Option 2: Docker Deployment

### Step 1: Build and Run
```bash
# Build Docker image
docker build -f docker/Dockerfile -t robust-vision-failsafes .

# Run container
docker run -p 8501:8501 robust-vision-failsafes
```

### Step 2: Access Web Interface
- Open browser to `http://localhost:8501`
- The system will automatically train the model on first run

---

## 🛠️ Option 3: Using Makefile (Easiest)

### One-Command Setup
```bash
# Complete setup and run
make install && make run-pipeline && make run-app
```

### Individual Commands
```bash
make install          # Install dependencies
make run-pipeline     # Run complete pipeline
make run-app          # Start web interface
make test             # Run tests
make docker-build     # Build Docker image
make docker-run       # Run Docker container
```

---

## 🌐 Web Interface Guide

### Page 1: Model Prediction & Safety Analysis
1. **Upload Image**: Click "Browse files" to upload a vehicle image
2. **View Results**: See prediction, confidence, and safety analysis
3. **Safety Status**: Check the safety level indicator (🟢🟡🔴)

### Page 2: Safety Monitoring Dashboard
1. **Real-Time Status**: Monitor current safety status
2. **Event Timeline**: View recent safety events
3. **Statistics**: Check performance metrics

### Page 3: Explainability Analysis
1. **Upload Image**: Upload image for SHAP analysis
2. **View Explanations**: See feature importance
3. **Interpret Results**: Understand model decisions

### Page 4: System Status & Performance
1. **Health Check**: Monitor system health
2. **Performance Metrics**: View accuracy and timing
3. **Resource Usage**: Check CPU/memory usage

---

## 🔧 Configuration

### Quick Configuration Changes
Edit `src/config.py` for common settings:

```python
# Safety thresholds
FAILSAFE_CONFIG = {
    'confidence_threshold': 0.7,  # Adjust safety sensitivity
    'uncertainty_threshold': 0.3,
}

# Model training
MODEL_CONFIG = {
    'epochs': 50,  # Training duration
    'learning_rate': 0.001,  # Learning rate
}
```

### Advanced Configuration
Use `configs/default_config.yaml` for detailed parameter tuning:

```yaml
failsafe:
  confidence_threshold: 0.7
  uncertainty_threshold: 0.3
  fallback_action: "stop"
```

---

## 🧪 Testing the System

### Run Tests
```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Quick system test
make quick-test
```

### Test Individual Components
```bash
# Test data preprocessing
python -m pytest tests/test_data_preprocessing.py -v

# Test fail-safe handler
python -m pytest tests/test_failsafe_handler.py -v
```

---

## 📊 Expected Results

### Model Performance
- **Training Time**: ~5-10 minutes (CPU), ~2-5 minutes (GPU)
- **Accuracy**: 85%+ on clean data
- **Safety Response**: <0.1 seconds

### Generated Files
After running the pipeline, you'll find:
- `models/vehicle_classifier.h5` - Trained model
- `visualizations/` - Performance plots and SHAP visualizations
- `logs/vision_system.log` - System logs
- `data/processed/` - Preprocessed data

---

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure you're in the project directory
cd "Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems"

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**2. Memory Issues**
```bash
# Reduce batch size in config
MODEL_CONFIG['batch_size'] = 16  # Instead of 32
```

**3. GPU Issues**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
```

**4. Streamlit Issues**
```bash
# Clear Streamlit cache
streamlit cache clear
```

### Getting Help
- Check `logs/vision_system.log` for detailed error messages
- Review the comprehensive `README.md`
- Run `make help` for available commands

---

## 🎯 Next Steps

### For Researchers
1. **Explore the Code**: Review `src/` modules
2. **Modify Parameters**: Edit `configs/default_config.yaml`
3. **Add New Attacks**: Extend `src/adversarial_testing.py`
4. **Custom Datasets**: Modify `src/data_preprocessing.py`

### For Developers
1. **Add Tests**: Create new test files in `tests/`
2. **Extend Interface**: Modify `app/app.py`
3. **Optimize Performance**: Profile and improve code
4. **Add Features**: Implement new safety mechanisms

### For Deployment
1. **Production Setup**: Configure for your environment
2. **Monitoring**: Set up logging and alerting
3. **Scaling**: Deploy with Kubernetes
4. **Security**: Implement access controls

---

## 📚 Additional Resources

- **Full Documentation**: `README.md`
- **Project Summary**: `PROJECT_SUMMARY.md`
- **Configuration Guide**: `configs/default_config.yaml`
- **API Reference**: Inline documentation in `src/` modules

---

## 🎉 You're Ready!

Your **Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems** is now running! 

**🚗 Safe AI for Autonomous Transport - Ready to Deploy! 🛡️**

---

*Need help? Check the comprehensive documentation or run `make help` for all available commands.*
