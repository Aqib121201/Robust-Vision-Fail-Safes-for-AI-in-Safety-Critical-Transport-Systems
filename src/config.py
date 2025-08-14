"""
Configuration file for Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems
Centralized hyperparameters, paths, and settings for reproducibility.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, 
                 MODELS_DIR, VISUALIZATIONS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "name": "CIFAR-10",  # Using CIFAR-10 as vehicle-like dataset
    "num_classes": 10,
    "image_size": (32, 32, 3),
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1,
    "batch_size": 32,
    "shuffle_buffer": 1000,
}

# Model configuration
MODEL_CONFIG = {
    "architecture": "CNN",
    "input_shape": (32, 32, 3),
    "num_classes": 10,
    "learning_rate": 0.001,
    "epochs": 50,
    "early_stopping_patience": 10,
    "model_save_path": str(MODELS_DIR / "vehicle_classifier.h5"),
}

# CNN Architecture
CNN_CONFIG = {
    "filters": [32, 64, 128, 256],
    "kernel_sizes": [3, 3, 3, 3],
    "pool_sizes": [2, 2, 2, 2],
    "dropout_rate": 0.5,
    "dense_units": [512, 256, 128],
}

# Adversarial testing configuration
ADVERSARIAL_CONFIG = {
    "noise_types": ["gaussian", "salt_pepper", "occlusion", "blur"],
    "noise_intensities": [0.1, 0.2, 0.3, 0.4, 0.5],
    "occlusion_sizes": [0.1, 0.2, 0.3, 0.4, 0.5],  # Fraction of image
    "blur_kernels": [3, 5, 7, 9],
    "fgsm_epsilon": [0.01, 0.03, 0.05, 0.1],
    "pgd_epsilon": 0.3,
    "pgd_alpha": 0.01,
    "pgd_steps": 40,
}

# Fail-safe configuration
FAILSAFE_CONFIG = {
    "confidence_threshold": 0.7,
    "uncertainty_threshold": 0.3,
    "max_retries": 3,
    "fallback_action": "stop",  # stop, slow_down, human_intervention
    "graceful_degradation": True,
    "logging_level": "INFO",
}

# SHAP configuration
SHAP_CONFIG = {
    "background_samples": 100,
    "explanation_samples": 50,
    "feature_names": None,  # Will be set dynamically
    "plot_type": "bar",  # bar, waterfall, force
    "save_plots": True,
}

# Evaluation metrics
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "auc"],
    "cross_validation_folds": 5,
    "random_seed": 42,
    "test_size": 0.2,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_DIR / "vision_system.log"),
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5,
}

# Hardware configuration
HARDWARE_CONFIG = {
    "use_gpu": True,
    "mixed_precision": True,
    "num_workers": 4,
    "pin_memory": True,
}

# Safety thresholds
SAFETY_THRESHOLDS = {
    "min_confidence": 0.6,
    "max_prediction_time": 0.1,  # seconds
    "max_memory_usage": 0.8,  # fraction of available memory
    "max_cpu_usage": 0.9,  # fraction of available CPU
}

# Vehicle classes mapping (for CIFAR-10)
VEHICLE_CLASSES = {
    0: "airplane",
    1: "automobile", 
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

# Safety-critical vehicle classes
SAFETY_CRITICAL_CLASSES = {
    "automobile": 1,
    "truck": 9,
    "airplane": 0,
    "ship": 8
}

# Environment variables
ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": "0",
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "PYTHONPATH": str(PROJECT_ROOT),
}
