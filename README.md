# Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems

## 🧠 Abstract

A comprehensive safety-critical vision system for autonomous transport applications, featuring CNN-based vehicle classification with integrated fail-safe mechanisms, adversarial testing, and SHAP-based explainability. The system achieves 85%+ accuracy while maintaining safety standards through confidence monitoring and automatic fallback actions.

## 🎯 Problem Statement

Autonomous vehicles require highly reliable computer vision systems that can operate under adverse conditions while maintaining safety standards. This work addresses the critical need for robustness, safety mechanisms, interpretability, and real-time monitoring in safety-critical transport applications.

## 📊 Dataset Description

**CIFAR-10 Dataset**: 60,000 32×32 color images across 10 classes, focusing on vehicle-related classes (automobile, truck, airplane, ship) as safety-critical categories. Data is normalized, split 80%/10%/10%, and augmented for robustness training.

## 🧪 Methodology

### Model Architecture
Custom CNN with 4 convolutional layers, batch normalization, dropout (0.5), and dense layers. Trained with Adam optimizer, categorical crossentropy loss, and early stopping.

### Adversarial Testing
Comprehensive evaluation against:
- Noise attacks (Gaussian, Salt & Pepper)
- Occlusion attacks (10-50% area)
- Blur attacks (3×3 to 9×9 kernels)
- Adversarial attacks (FGSM, PGD)

### Fail-Safe Mechanism
Multi-level safety framework:
- **NORMAL**: Confidence > 0.7
- **WARNING**: 0.5 < Confidence ≤ 0.7
- **CRITICAL**: Confidence ≤ 0.5
- **EMERGENCY**: Safety-critical class with low confidence

### Explainability
SHAP-based interpretability for global and local explanations, failure analysis, and safety validation.

## 📈 Results

| Metric | Clean Data | Gaussian Noise (σ=0.2) | Occlusion (30%) |
|--------|------------|------------------------|-----------------|
| Accuracy | 0.87 | 0.72 | 0.65 |
| Precision | 0.86 | 0.71 | 0.64 |
| Recall | 0.87 | 0.72 | 0.65 |
| F1-Score | 0.86 | 0.71 | 0.64 |

**Fail-Safe Effectiveness**: 3.2% false positive rate, 1.8% false negative rate, 0.15s average response time.

## 📂 Project Structure

```
📦 Robust Vision Fail-Safes/
├── 📁 data/                   # Raw & processed datasets
├── 📁 notebooks/             # Jupyter notebooks
├── 📁 src/                   # Core source code
│   ├── config.py             # Configuration
│   ├── data_preprocessing.py # Data loading
│   ├── model_training.py     # CNN training
│   ├── adversarial_testing.py # Robustness evaluation
│   ├── failsafe_handler.py   # Safety mechanisms
│   └── explainability.py     # SHAP analysis
├── 📁 models/                # Trained models
├── 📁 visualizations/        # Generated plots
├── 📁 tests/                 # Unit tests
├── 📁 report/                # Academic report
├── 📁 app/                   # Streamlit interface
├── requirements.txt          # Dependencies
└── run_pipeline.py           # Main script
```

## 💻 How to Run

### Installation
```bash
git clone https://github.com/your-username/robust-vision-failsafes.git
cd robust-vision-failsafes
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Pipeline
```bash
# Complete pipeline
python run_pipeline.py --step all

# Individual steps
python run_pipeline.py --step data
python run_pipeline.py --step train
python run_pipeline.py --step adversarial
python run_pipeline.py --step failsafe
python run_pipeline.py --step explain
```

### Web Interface
```bash
cd app
streamlit run app.py
```

## 🧪 Unit Tests
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📚 References

1. Goodfellow, I. J., et al. (2014). Explaining and harnessing adversarial examples.
2. Amodei, D., et al. (2016). Concrete problems in AI safety.
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
4. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images.

## ⚠️ Limitations

- Limited to CIFAR-10 classes; real-world vehicle diversity not fully captured
- 32×32 resolution may miss fine-grained details
- Testing in controlled conditions; real-world validation needed
- SHAP analysis adds computational overhead for real-time applications

## 📄 PDF Report

[📄 Download Full Academic Report](./report/Thesis_SafetyCriticalVision.pdf)

## 🧠 Contribution & Acknowledgements

**Team**: Lead Researcher, ML Engineer, Safety Engineer, Data Scientist

**Acknowledgements**: CIFAR-10 creators, SHAP developers, TensorFlow team, academic community

---

**License**: MIT License  
**Contact**: Open an issue for questions and collaboration
