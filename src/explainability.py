"""
Explainability module for Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems
Implements SHAP analysis and other interpretability techniques for model understanding and failure detection.
"""

import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import confusion_matrix
import pandas as pd

from .config import (
    SHAP_CONFIG, VISUALIZATIONS_DIR, VEHICLE_CLASSES, 
    SAFETY_CRITICAL_CLASSES, LOGGING_CONFIG
)

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Model explainability class using SHAP and other interpretability techniques.
    Provides insights into model decisions and failure modes.
    """
    
    def __init__(self, model: tf.keras.Model, config: Dict[str, Any] = None):
        """
        Initialize the model explainer.
        
        Args:
            model: Trained Keras model to explain
            config: Configuration dictionary for explainability
        """
        self.model = model
        self.config = config or SHAP_CONFIG
        self.explainer = None
        self.background_data = None
        
        logger.info("ModelExplainer initialized")
    
    def prepare_background_data(self, background_images: np.ndarray) -> None:
        """
        Prepare background data for SHAP explainer.
        
        Args:
            background_images: Background images for SHAP
        """
        logger.info("Preparing background data for SHAP...")
        
        # Sample background data if too large
        if len(background_images) > self.config['background_samples']:
            indices = np.random.choice(
                len(background_images), 
                self.config['background_samples'], 
                replace=False
            )
            self.background_data = background_images[indices]
        else:
            self.background_data = background_images
        
        logger.info(f"Background data prepared with {len(self.background_data)} samples")
    
    def create_shap_explainer(self) -> shap.Explainer:
        """
        Create SHAP explainer for the model.
        
        Returns:
            SHAP explainer object
        """
        if self.background_data is None:
            raise ValueError("Background data not prepared. Call prepare_background_data first.")
        
        logger.info("Creating SHAP explainer...")
        
        # Create SHAP explainer
        self.explainer = shap.Explainer(
            self.model, 
            self.background_data,
            output_names=list(VEHICLE_CLASSES.values())
        )
        
        logger.info("SHAP explainer created successfully")
        return self.explainer
    
    def explain_predictions(self, images: np.ndarray, 
                          labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate SHAP explanations for predictions.
        
        Args:
            images: Images to explain
            labels: True labels (optional)
            
        Returns:
            Dictionary with SHAP explanations and analysis
        """
        if self.explainer is None:
            self.create_shap_explainer()
        
        logger.info(f"Generating SHAP explanations for {len(images)} images...")
        
        # Sample images if too many
        if len(images) > self.config['explanation_samples']:
            indices = np.random.choice(len(images), self.config['explanation_samples'], replace=False)
            sample_images = images[indices]
            sample_labels = labels[indices] if labels is not None else None
        else:
            sample_images = images
            sample_labels = labels
        
        # Generate SHAP values
        shap_values = self.explainer(sample_images)
        
        # Get predictions
        predictions = self.model.predict(sample_images)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        explanations = {
            'shap_values': shap_values,
            'predictions': predictions,
            'predicted_classes': predicted_classes,
            'confidences': confidences,
            'images': sample_images,
            'true_labels': sample_labels
        }
        
        logger.info("SHAP explanations generated successfully")
        return explanations
    
    def analyze_failure_cases(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze failure cases using SHAP explanations.
        
        Args:
            explanations: SHAP explanations from explain_predictions
            
        Returns:
            Dictionary with failure analysis
        """
        logger.info("Analyzing failure cases...")
        
        shap_values = explanations['shap_values']
        predicted_classes = explanations['predicted_classes']
        true_labels = explanations['true_labels']
        confidences = explanations['confidences']
        
        if true_labels is None:
            logger.warning("No true labels provided for failure analysis")
            return {}
        
        # Find failure cases
        failures = predicted_classes != np.argmax(true_labels, axis=1)
        failure_indices = np.where(failures)[0]
        
        failure_analysis = {
            'total_samples': len(predicted_classes),
            'failure_count': len(failure_indices),
            'failure_rate': len(failure_indices) / len(predicted_classes),
            'failure_cases': [],
            'low_confidence_failures': [],
            'high_confidence_failures': []
        }
        
        # Analyze each failure case
        for idx in failure_indices:
            true_class = np.argmax(true_labels[idx])
            pred_class = predicted_classes[idx]
            confidence = confidences[idx]
            
            failure_case = {
                'index': idx,
                'true_class': true_class,
                'predicted_class': pred_class,
                'confidence': confidence,
                'true_class_name': VEHICLE_CLASSES[true_class],
                'predicted_class_name': VEHICLE_CLASSES[pred_class],
                'is_safety_critical': true_class in SAFETY_CRITICAL_CLASSES.values()
            }
            
            failure_analysis['failure_cases'].append(failure_case)
            
            # Categorize by confidence
            if confidence < 0.5:
                failure_analysis['low_confidence_failures'].append(failure_case)
            else:
                failure_analysis['high_confidence_failures'].append(failure_case)
        
        logger.info(f"Failure analysis completed: {len(failure_indices)} failures found")
        return failure_analysis
    
    def create_shap_visualizations(self, explanations: Dict[str, Any], 
                                 save_plots: bool = True) -> None:
        """
        Create SHAP visualizations for model explanations.
        
        Args:
            explanations: SHAP explanations from explain_predictions
            save_plots: Whether to save plots to disk
        """
        logger.info("Creating SHAP visualizations...")
        
        shap_values = explanations['shap_values']
        images = explanations['images']
        predicted_classes = explanations['predicted_classes']
        true_labels = explanations['true_labels']
        
        # 1. Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values.values, 
            images,
            class_names=list(VEHICLE_CLASSES.values()),
            show=False
        )
        if save_plots:
            plt.savefig(VISUALIZATIONS_DIR / "shap_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Force plot for individual predictions
        for i in range(min(5, len(images))):  # Show first 5 examples
            plt.figure(figsize=(10, 6))
            true_class = np.argmax(true_labels[i]) if true_labels is not None else None
            pred_class = predicted_classes[i]
            
            title = f"Prediction: {VEHICLE_CLASSES[pred_class]}"
            if true_labels is not None:
                title += f" | True: {VEHICLE_CLASSES[true_class]}"
            
            shap.force_plot(
                shap_values.base_values[i],
                shap_values.values[i],
                images[i],
                class_names=list(VEHICLE_CLASSES.values()),
                show=False
            )
            plt.title(title)
            
            if save_plots:
                plt.savefig(VISUALIZATIONS_DIR / f"shap_force_plot_{i}.png", 
                           dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Waterfall plot for specific classes
        for class_name, class_idx in SAFETY_CRITICAL_CLASSES.items():
            class_samples = predicted_classes == class_idx
            if np.any(class_samples):
                sample_idx = np.where(class_samples)[0][0]
                
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(
                    shap_values[sample_idx],
                    show=False
                )
                plt.title(f"SHAP Waterfall Plot - {class_name}")
                
                if save_plots:
                    plt.savefig(VISUALIZATIONS_DIR / f"shap_waterfall_{class_name}.png", 
                               dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info("SHAP visualizations created successfully")
    
    def create_confusion_matrix_analysis(self, predictions: np.ndarray, 
                                       labels: np.ndarray) -> Dict[str, Any]:
        """
        Create confusion matrix analysis for model predictions.
        
        Args:
            predictions: Model predictions
            labels: True labels
            
        Returns:
            Dictionary with confusion matrix analysis
        """
        logger.info("Creating confusion matrix analysis...")
        
        y_true = np.argmax(labels, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate per-class metrics
        class_metrics = {}
        for i, class_name in VEHICLE_CLASSES.items():
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
        
        # Create confusion matrix visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(VEHICLE_CLASSES.values()),
            yticklabels=list(VEHICLE_CLASSES.values())
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        save_path = VISUALIZATIONS_DIR / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        analysis = {
            'confusion_matrix': cm,
            'class_metrics': class_metrics,
            'overall_accuracy': np.sum(np.diag(cm)) / np.sum(cm)
        }
        
        logger.info("Confusion matrix analysis completed")
        return analysis
    
    def analyze_feature_importance(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze feature importance using SHAP values.
        
        Args:
            explanations: SHAP explanations from explain_predictions
            
        Returns:
            Dictionary with feature importance analysis
        """
        logger.info("Analyzing feature importance...")
        
        shap_values = explanations['shap_values']
        predicted_classes = explanations['predicted_classes']
        
        # Calculate mean absolute SHAP values for each class
        feature_importance = {}
        
        for class_name, class_idx in VEHICLE_CLASSES.items():
            class_samples = predicted_classes == class_idx
            if np.any(class_samples):
                class_shap_values = shap_values.values[class_samples]
                mean_importance = np.mean(np.abs(class_shap_values), axis=0)
                feature_importance[class_name] = mean_importance
        
        # Create feature importance visualization
        if feature_importance:
            plt.figure(figsize=(15, 10))
            
            # Plot for safety-critical classes
            safety_classes = list(SAFETY_CRITICAL_CLASSES.keys())
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            for i, class_name in enumerate(safety_classes):
                if class_name in feature_importance:
                    row, col = i // 2, i % 2
                    
                    # Reshape importance to image shape
                    importance_img = feature_importance[class_name].reshape(32, 32, 3)
                    importance_img = np.mean(importance_img, axis=2)  # Average across channels
                    
                    axes[row, col].imshow(importance_img, cmap='hot')
                    axes[row, col].set_title(f'Feature Importance - {class_name}')
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            save_path = VISUALIZATIONS_DIR / "feature_importance_safety_critical.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Feature importance analysis completed")
        return feature_importance
    
    def generate_explanation_report(self, explanations: Dict[str, Any], 
                                  failure_analysis: Dict[str, Any],
                                  confusion_analysis: Dict[str, Any]) -> str:
        """
        Generate a comprehensive explanation report.
        
        Args:
            explanations: SHAP explanations
            failure_analysis: Failure analysis results
            confusion_analysis: Confusion matrix analysis
            
        Returns:
            Formatted report string
        """
        logger.info("Generating explanation report...")
        
        report = []
        report.append("=" * 60)
        report.append("MODEL EXPLANABILITY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Model performance summary
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 30)
        report.append(f"Total samples analyzed: {len(explanations['predicted_classes'])}")
        report.append(f"Overall accuracy: {confusion_analysis['overall_accuracy']:.4f}")
        
        if failure_analysis:
            report.append(f"Failure rate: {failure_analysis['failure_rate']:.4f}")
            report.append(f"Total failures: {failure_analysis['failure_count']}")
        
        report.append("")
        
        # Safety-critical analysis
        report.append("SAFETY-CRITICAL CLASS ANALYSIS")
        report.append("-" * 35)
        
        for class_name, class_idx in SAFETY_CRITICAL_CLASSES.items():
            if class_name in confusion_analysis['class_metrics']:
                metrics = confusion_analysis['class_metrics'][class_name]
                report.append(f"{class_name}:")
                report.append(f"  Precision: {metrics['precision']:.4f}")
                report.append(f"  Recall: {metrics['recall']:.4f}")
                report.append(f"  F1-Score: {metrics['f1_score']:.4f}")
                report.append("")
        
        # Failure analysis
        if failure_analysis and failure_analysis['failure_cases']:
            report.append("FAILURE ANALYSIS")
            report.append("-" * 20)
            report.append(f"Low confidence failures: {len(failure_analysis['low_confidence_failures'])}")
            report.append(f"High confidence failures: {len(failure_analysis['high_confidence_failures'])}")
            report.append("")
            
            # Show some example failures
            report.append("EXAMPLE FAILURE CASES:")
            for i, failure in enumerate(failure_analysis['failure_cases'][:5]):
                report.append(f"  {i+1}. {failure['true_class_name']} → {failure['predicted_class_name']} "
                            f"(confidence: {failure['confidence']:.3f})")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 15)
        
        if failure_analysis and failure_analysis['failure_rate'] > 0.1:
            report.append("• High failure rate detected - consider model retraining")
        
        if failure_analysis and len(failure_analysis['high_confidence_failures']) > 0:
            report.append("• High-confidence failures detected - review model calibration")
        
        report.append("• Monitor safety-critical class performance closely")
        report.append("• Implement confidence thresholds for fail-safe mechanisms")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report to file
        report_path = VISUALIZATIONS_DIR / "explainability_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Explanation report saved to {report_path}")
        return report_text

def main():
    """Main function for testing explainability."""
    from .data_preprocessing import DataPreprocessor
    from .model_training import VehicleClassifier
    
    # Load data and model
    preprocessor = DataPreprocessor()
    train_dataset, val_dataset, test_dataset = preprocessor.get_data_generators()
    
    classifier = VehicleClassifier()
    classifier.load_model()
    
    # Get test data
    x_test, y_test = preprocessor.test_data
    
    # Initialize explainer
    explainer = ModelExplainer(classifier.model)
    
    # Prepare background data
    explainer.prepare_background_data(x_test[:100])  # Use first 100 samples as background
    
    # Generate explanations
    explanations = explainer.explain_predictions(x_test[:50], y_test[:50])
    
    # Analyze failures
    failure_analysis = explainer.analyze_failure_cases(explanations)
    
    # Create confusion matrix analysis
    confusion_analysis = explainer.create_confusion_matrix_analysis(
        explanations['predictions'], explanations['true_labels']
    )
    
    # Create visualizations
    explainer.create_shap_visualizations(explanations)
    explainer.analyze_feature_importance(explanations)
    
    # Generate report
    report = explainer.generate_explanation_report(
        explanations, failure_analysis, confusion_analysis
    )
    
    print("Explainability analysis completed!")
    print(f"Failure rate: {failure_analysis.get('failure_rate', 0):.4f}")

if __name__ == "__main__":
    main()
