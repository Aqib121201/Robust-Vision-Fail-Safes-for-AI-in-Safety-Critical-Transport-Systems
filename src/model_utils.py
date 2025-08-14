"""
Model utilities for Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems
Contains helper functions for model evaluation, visualization, and safety analysis.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from .config import VEHICLE_CLASSES, SAFETY_CRITICAL_CLASSES, VISUALIZATIONS_DIR

logger = logging.getLogger(__name__)

def calculate_confidence_metrics(predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    """
    Calculate confidence-based metrics for model evaluation.
    
    Args:
        predictions: Model predictions (probabilities)
        true_labels: True labels (one-hot encoded)
        
    Returns:
        Dictionary with confidence metrics
    """
    confidences = np.max(predictions, axis=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    
    # Calculate accuracy by confidence bins
    confidence_bins = [0.0, 0.5, 0.7, 0.9, 1.0]
    bin_accuracies = []
    bin_counts = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(predicted_classes[mask] == true_classes[mask])
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0.0)
            bin_counts.append(0)
    
    # Calculate calibration metrics
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_classes, confidences, n_bins=10
    )
    
    # Calculate expected calibration error (ECE)
    ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    
    metrics = {
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'confidence_bins': confidence_bins[1:],
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts,
        'expected_calibration_error': ece,
        'calibration_fractions': fraction_of_positives,
        'calibration_means': mean_predicted_value
    }
    
    return metrics

def analyze_safety_critical_performance(predictions: np.ndarray, 
                                      true_labels: np.ndarray) -> Dict[str, Any]:
    """
    Analyze model performance specifically on safety-critical classes.
    
    Args:
        predictions: Model predictions (probabilities)
        true_labels: True labels (one-hot encoded)
        
    Returns:
        Dictionary with safety-critical performance analysis
    """
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Identify safety-critical samples
    safety_critical_mask = np.isin(true_classes, list(SAFETY_CRITICAL_CLASSES.values()))
    non_critical_mask = ~safety_critical_mask
    
    # Calculate metrics for safety-critical vs non-critical
    safety_critical_accuracy = np.mean(predicted_classes[safety_critical_mask] == true_classes[safety_critical_mask])
    non_critical_accuracy = np.mean(predicted_classes[non_critical_mask] == true_classes[non_critical_mask])
    
    safety_critical_confidence = np.mean(confidences[safety_critical_mask])
    non_critical_confidence = np.mean(confidences[non_critical_mask])
    
    # Analyze false negatives (missed safety-critical detections)
    safety_critical_false_negatives = np.sum(
        (predicted_classes[safety_critical_mask] != true_classes[safety_critical_mask]) &
        (confidences[safety_critical_mask] > 0.7)  # High confidence but wrong
    )
    
    # Analyze false positives (non-critical classified as safety-critical)
    non_critical_false_positives = np.sum(
        np.isin(predicted_classes[non_critical_mask], list(SAFETY_CRITICAL_CLASSES.values()))
    )
    
    analysis = {
        'safety_critical_accuracy': safety_critical_accuracy,
        'non_critical_accuracy': non_critical_accuracy,
        'safety_critical_confidence': safety_critical_confidence,
        'non_critical_confidence': non_critical_confidence,
        'safety_critical_samples': np.sum(safety_critical_mask),
        'non_critical_samples': np.sum(non_critical_mask),
        'safety_critical_false_negatives': safety_critical_false_negatives,
        'non_critical_false_positives': non_critical_false_positives,
        'false_negative_rate': safety_critical_false_negatives / np.sum(safety_critical_mask),
        'false_positive_rate': non_critical_false_positives / np.sum(non_critical_mask)
    }
    
    return analysis

def create_performance_visualizations(predictions: np.ndarray, 
                                    true_labels: np.ndarray,
                                    save_plots: bool = True) -> None:
    """
    Create comprehensive performance visualizations.
    
    Args:
        predictions: Model predictions (probabilities)
        true_labels: True labels (one-hot encoded)
        save_plots: Whether to save plots to disk
    """
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Confusion Matrix
    plt.subplot(3, 3, 1)
    cm = confusion_matrix(true_classes, predicted_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(VEHICLE_CLASSES.values()),
                yticklabels=list(VEHICLE_CLASSES.values()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # 2. Confidence Distribution
    plt.subplot(3, 3, 2)
    plt.hist(confidences, bins=50, alpha=0.7, color='blue')
    plt.title('Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    
    # 3. Accuracy vs Confidence
    plt.subplot(3, 3, 3)
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_centers = []
    
    for i in range(len(confidence_bins) - 1):
        mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(predicted_classes[mask] == true_classes[mask])
            bin_accuracies.append(bin_accuracy)
            bin_centers.append((confidence_bins[i] + confidence_bins[i + 1]) / 2)
    
    plt.plot(bin_centers, bin_accuracies, 'o-', color='red')
    plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    plt.title('Accuracy vs Confidence')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 4. Per-class Accuracy
    plt.subplot(3, 3, 4)
    class_accuracies = []
    class_names = []
    
    for i in range(10):
        mask = true_classes == i
        if np.sum(mask) > 0:
            class_accuracy = np.mean(predicted_classes[mask] == true_classes[mask])
            class_accuracies.append(class_accuracy)
            class_names.append(VEHICLE_CLASSES[i])
    
    colors = ['red' if i in SAFETY_CRITICAL_CLASSES.values() else 'blue' for i in range(10)]
    plt.bar(range(len(class_accuracies)), class_accuracies, color=colors, alpha=0.7)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    
    # 5. Per-class Confidence
    plt.subplot(3, 3, 5)
    class_confidences = []
    
    for i in range(10):
        mask = true_classes == i
        if np.sum(mask) > 0:
            class_confidence = np.mean(confidences[mask])
            class_confidences.append(class_confidence)
    
    plt.bar(range(len(class_confidences)), class_confidences, color=colors, alpha=0.7)
    plt.title('Per-Class Average Confidence')
    plt.xlabel('Class')
    plt.ylabel('Average Confidence')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    
    # 6. Safety-Critical vs Non-Critical Performance
    plt.subplot(3, 3, 6)
    safety_analysis = analyze_safety_critical_performance(predictions, true_labels)
    
    categories = ['Safety-Critical', 'Non-Critical']
    accuracies = [safety_analysis['safety_critical_accuracy'], safety_analysis['non_critical_accuracy']]
    confidences = [safety_analysis['safety_critical_confidence'], safety_analysis['non_critical_confidence']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy', color='blue', alpha=0.7)
    plt.bar(x + width/2, confidences, width, label='Confidence', color='red', alpha=0.7)
    plt.title('Safety-Critical vs Non-Critical Performance')
    plt.xlabel('Category')
    plt.ylabel('Score')
    plt.xticks(x, categories)
    plt.legend()
    
    # 7. Error Analysis
    plt.subplot(3, 3, 7)
    errors = predicted_classes != true_classes
    error_confidences = confidences[errors]
    correct_confidences = confidences[~errors]
    
    plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green')
    plt.hist(error_confidences, bins=30, alpha=0.7, label='Errors', color='red')
    plt.title('Confidence Distribution: Correct vs Errors')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 8. Calibration Plot
    plt.subplot(3, 3, 8)
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_classes, confidences, n_bins=10
    )
    
    plt.plot(mean_predicted_value, fraction_of_positives, 'o-', label='Model')
    plt.plot([0, 1], [0, 1], '--', label='Perfectly Calibrated')
    plt.title('Calibration Plot')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Summary Statistics
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Calculate summary statistics
    overall_accuracy = np.mean(predicted_classes == true_classes)
    mean_confidence = np.mean(confidences)
    std_confidence = np.std(confidences)
    
    summary_text = f"""
    Performance Summary:
    
    Overall Accuracy: {overall_accuracy:.3f}
    Mean Confidence: {mean_confidence:.3f}
    Confidence Std: {std_confidence:.3f}
    
    Safety-Critical Accuracy: {safety_analysis['safety_critical_accuracy']:.3f}
    Non-Critical Accuracy: {safety_analysis['non_critical_accuracy']:.3f}
    
    False Negative Rate: {safety_analysis['false_negative_rate']:.3f}
    False Positive Rate: {safety_analysis['false_positive_rate']:.3f}
    """
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(VISUALIZATIONS_DIR / "performance_analysis.png", dpi=300, bbox_inches='tight')
        logger.info("Performance visualizations saved")
    
    plt.show()

def generate_performance_report(predictions: np.ndarray, 
                              true_labels: np.ndarray,
                              model_name: str = "Vehicle Classifier") -> str:
    """
    Generate a comprehensive performance report.
    
    Args:
        predictions: Model predictions (probabilities)
        true_labels: True labels (one-hot encoded)
        model_name: Name of the model
        
    Returns:
        Formatted performance report string
    """
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Calculate basic metrics
    overall_accuracy = np.mean(predicted_classes == true_classes)
    mean_confidence = np.mean(confidences)
    
    # Get classification report
    class_report = classification_report(true_classes, predicted_classes, 
                                       target_names=list(VEHICLE_CLASSES.values()),
                                       output_dict=True)
    
    # Safety analysis
    safety_analysis = analyze_safety_critical_performance(predictions, true_labels)
    
    # Confidence metrics
    confidence_metrics = calculate_confidence_metrics(predictions, true_labels)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append(f"PERFORMANCE REPORT - {model_name.upper()}")
    report.append("=" * 80)
    report.append("")
    
    # Overall performance
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 20)
    report.append(f"Accuracy: {overall_accuracy:.4f}")
    report.append(f"Mean Confidence: {mean_confidence:.4f}")
    report.append(f"Confidence Std: {confidence_metrics['std_confidence']:.4f}")
    report.append(f"Expected Calibration Error: {confidence_metrics['expected_calibration_error']:.4f}")
    report.append("")
    
    # Per-class performance
    report.append("PER-CLASS PERFORMANCE")
    report.append("-" * 20)
    for class_name in VEHICLE_CLASSES.values():
        if class_name in class_report:
            metrics = class_report[class_name]
            is_critical = class_name in SAFETY_CRITICAL_CLASSES
            critical_mark = " (SAFETY-CRITICAL)" if is_critical else ""
            
            report.append(f"{class_name}{critical_mark}:")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1-Score: {metrics['f1-score']:.4f}")
            report.append(f"  Support: {metrics['support']}")
            report.append("")
    
    # Safety analysis
    report.append("SAFETY-CRITICAL ANALYSIS")
    report.append("-" * 25)
    report.append(f"Safety-Critical Accuracy: {safety_analysis['safety_critical_accuracy']:.4f}")
    report.append(f"Non-Critical Accuracy: {safety_analysis['non_critical_accuracy']:.4f}")
    report.append(f"Safety-Critical Confidence: {safety_analysis['safety_critical_confidence']:.4f}")
    report.append(f"Non-Critical Confidence: {safety_analysis['non_critical_confidence']:.4f}")
    report.append(f"False Negative Rate: {safety_analysis['false_negative_rate']:.4f}")
    report.append(f"False Positive Rate: {safety_analysis['false_positive_rate']:.4f}")
    report.append("")
    
    # Confidence analysis
    report.append("CONFIDENCE ANALYSIS")
    report.append("-" * 18)
    report.append(f"Min Confidence: {confidence_metrics['min_confidence']:.4f}")
    report.append(f"Max Confidence: {confidence_metrics['max_confidence']:.4f}")
    report.append("")
    
    report.append("Confidence Bins Accuracy:")
    for i, (bin_edge, accuracy, count) in enumerate(zip(
        confidence_metrics['confidence_bins'],
        confidence_metrics['bin_accuracies'],
        confidence_metrics['bin_counts']
    )):
        if count > 0:
            report.append(f"  [{bin_edge-0.1:.1f}, {bin_edge:.1f}): {accuracy:.4f} ({count} samples)")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 15)
    
    if safety_analysis['false_negative_rate'] > 0.05:
        report.append("• High false negative rate detected - consider model retraining")
    
    if safety_analysis['safety_critical_accuracy'] < 0.8:
        report.append("• Safety-critical accuracy below threshold - review model performance")
    
    if confidence_metrics['expected_calibration_error'] > 0.1:
        report.append("• Poor calibration detected - consider confidence calibration")
    
    if mean_confidence < 0.7:
        report.append("• Low average confidence - review model training")
    
    report.append("• Monitor safety-critical class performance closely")
    report.append("• Implement confidence thresholds for fail-safe mechanisms")
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)

def save_model_artifacts(model: tf.keras.Model, 
                        predictions: np.ndarray,
                        true_labels: np.ndarray,
                        model_name: str = "vehicle_classifier") -> None:
    """
    Save model artifacts and performance analysis.
    
    Args:
        model: Trained Keras model
        predictions: Model predictions
        true_labels: True labels
        model_name: Name for the model
    """
    import json
    from datetime import datetime
    
    # Save model
    model_path = VISUALIZATIONS_DIR / f"{model_name}.h5"
    model.save(str(model_path))
    
    # Generate and save performance report
    report = generate_performance_report(predictions, true_labels, model_name)
    report_path = VISUALIZATIONS_DIR / f"{model_name}_performance_report.txt"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Save performance metrics as JSON
    metrics = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'overall_accuracy': float(np.mean(np.argmax(predictions, axis=1) == np.argmax(true_labels, axis=1))),
        'mean_confidence': float(np.mean(np.max(predictions, axis=1))),
        'safety_analysis': analyze_safety_critical_performance(predictions, true_labels),
        'confidence_metrics': calculate_confidence_metrics(predictions, true_labels)
    }
    
    metrics_path = VISUALIZATIONS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create visualizations
    create_performance_visualizations(predictions, true_labels)
    
    logger.info(f"Model artifacts saved to {VISUALIZATIONS_DIR}")
    logger.info(f"Model: {model_path}")
    logger.info(f"Report: {report_path}")
    logger.info(f"Metrics: {metrics_path}")

if __name__ == "__main__":
    # Test the utility functions
    print("Model utilities module loaded successfully!")
    print("Available functions:")
    print("- calculate_confidence_metrics()")
    print("- analyze_safety_critical_performance()")
    print("- create_performance_visualizations()")
    print("- generate_performance_report()")
    print("- save_model_artifacts()")
