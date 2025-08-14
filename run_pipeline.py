#!/usr/bin/env python3
"""
Main orchestrator script for Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems
Runs the complete pipeline: data preprocessing, model training, adversarial testing, and fail-safe evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path
import time
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import LOGGING_CONFIG, VISUALIZATIONS_DIR, MODELS_DIR
from src.data_preprocessing import DataPreprocessor
from src.model_training import VehicleClassifier
from src.adversarial_testing import AdversarialTester
from src.failsafe_handler import FailSafeHandler, SafetyMonitor
from src.explainability import ModelExplainer

# Setup logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def run_data_preprocessing() -> DataPreprocessor:
    """Run data preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("=" * 60)
    
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    train_data, val_data, test_data = preprocessor.load_cifar10_data()
    
    # Get dataset statistics
    stats = preprocessor.get_dataset_statistics()
    logger.info("Dataset Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Save processed data
    preprocessor.save_processed_data()
    
    logger.info("Data preprocessing completed successfully!")
    return preprocessor

def run_model_training(preprocessor: DataPreprocessor) -> VehicleClassifier:
    """Run model training pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 2: MODEL TRAINING")
    logger.info("=" * 60)
    
    # Get data generators
    train_dataset, val_dataset, test_dataset = preprocessor.get_data_generators()
    
    # Initialize and train classifier
    classifier = VehicleClassifier()
    training_results = classifier.train_model(train_dataset, val_dataset)
    
    # Evaluate model
    metrics = classifier.evaluate_model(test_dataset)
    logger.info("Model Evaluation Results:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Save model
    classifier.save_model()
    
    logger.info("Model training completed successfully!")
    return classifier

def run_adversarial_testing(classifier: VehicleClassifier, preprocessor: DataPreprocessor) -> AdversarialTester:
    """Run adversarial testing pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 3: ADVERSARIAL TESTING")
    logger.info("=" * 60)
    
    # Get test data
    x_test, y_test = preprocessor.test_data
    
    # Initialize adversarial tester
    tester = AdversarialTester(classifier.model)
    
    # Run noise robustness tests
    logger.info("Testing noise robustness...")
    noise_results = tester.test_noise_robustness(x_test, y_test)
    
    # Run adversarial attacks
    logger.info("Testing adversarial attacks...")
    adv_results = tester.test_adversarial_attacks(x_test, y_test)
    
    # Analyze confidence drops
    logger.info("Analyzing confidence drops...")
    conf_analysis = tester.analyze_confidence_drops(x_test, y_test)
    
    # Create visualizations
    logger.info("Creating visualizations...")
    tester.visualize_attacks(x_test, y_test)
    tester.plot_robustness_results()
    
    logger.info("Adversarial testing completed successfully!")
    return tester

def run_failsafe_evaluation(classifier: VehicleClassifier, preprocessor: DataPreprocessor) -> FailSafeHandler:
    """Run fail-safe evaluation pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 4: FAIL-SAFE EVALUATION")
    logger.info("=" * 60)
    
    # Get test data
    x_test, y_test = preprocessor.test_data
    
    # Initialize fail-safe handler
    failsafe_handler = FailSafeHandler()
    
    # Initialize safety monitor
    safety_monitor = SafetyMonitor(failsafe_handler)
    safety_monitor.start_monitoring()
    
    # Test fail-safe mechanisms
    logger.info("Testing fail-safe mechanisms...")
    safety_results = []
    
    for i in range(min(100, len(x_test))):  # Test first 100 samples
        image = x_test[i:i+1]
        true_label = y_test[i:i+1]
        
        # Get model prediction
        prediction = classifier.model.predict(image, verbose=0)
        
        # Process through safety monitor
        result = safety_monitor.process_prediction(prediction[0], image[0], true_label[0])
        safety_results.append(result)
        
        if i % 20 == 0:
            logger.info(f"Processed {i+1} samples...")
    
    # Stop monitoring
    safety_monitor.stop_monitoring()
    
    # Get safety statistics
    stats = failsafe_handler.get_safety_statistics()
    logger.info("Safety Statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")
    
    # Save safety log
    failsafe_handler.save_safety_log()
    
    logger.info("Fail-safe evaluation completed successfully!")
    return failsafe_handler

def run_explainability_analysis(classifier: VehicleClassifier, preprocessor: DataPreprocessor) -> ModelExplainer:
    """Run explainability analysis pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 5: EXPLAINABILITY ANALYSIS")
    logger.info("=" * 60)
    
    # Get test data
    x_test, y_test = preprocessor.test_data
    
    # Initialize explainer
    explainer = ModelExplainer(classifier.model)
    
    # Prepare background data
    logger.info("Preparing background data...")
    explainer.prepare_background_data(x_test[:100])
    
    # Generate explanations
    logger.info("Generating SHAP explanations...")
    explanations = explainer.explain_predictions(x_test[:50], y_test[:50])
    
    # Analyze failures
    logger.info("Analyzing failure cases...")
    failure_analysis = explainer.analyze_failure_cases(explanations)
    
    # Create confusion matrix analysis
    logger.info("Creating confusion matrix analysis...")
    confusion_analysis = explainer.create_confusion_matrix_analysis(
        explanations['predictions'], explanations['true_labels']
    )
    
    # Create visualizations
    logger.info("Creating SHAP visualizations...")
    explainer.create_shap_visualizations(explanations)
    explainer.analyze_feature_importance(explanations)
    
    # Generate report
    logger.info("Generating explanation report...")
    report = explainer.generate_explanation_report(
        explanations, failure_analysis, confusion_analysis
    )
    
    logger.info("Explainability analysis completed successfully!")
    return explainer

def run_complete_pipeline():
    """Run the complete safety-critical vision system pipeline."""
    start_time = time.time()
    
    logger.info("🚀 Starting Robust Vision Fail-Safes Pipeline")
    logger.info("=" * 80)
    
    try:
        # Step 1: Data Preprocessing
        preprocessor = run_data_preprocessing()
        
        # Step 2: Model Training
        classifier = run_model_training(preprocessor)
        
        # Step 3: Adversarial Testing
        tester = run_adversarial_testing(classifier, preprocessor)
        
        # Step 4: Fail-Safe Evaluation
        failsafe_handler = run_failsafe_evaluation(classifier, preprocessor)
        
        # Step 5: Explainability Analysis
        explainer = run_explainability_analysis(classifier, preprocessor)
        
        # Pipeline completion
        total_time = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
        logger.info("=" * 80)
        
        # Summary
        logger.info("📊 PIPELINE SUMMARY:")
        logger.info(f"  • Data samples processed: {len(preprocessor.train_data[0]) + len(preprocessor.test_data[0])}")
        logger.info(f"  • Model saved to: {MODELS_DIR}")
        logger.info(f"  • Visualizations saved to: {VISUALIZATIONS_DIR}")
        logger.info(f"  • Safety events logged: {len(failsafe_handler.safety_events)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        return False

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems"
    )
    
    parser.add_argument(
        "--step",
        type=str,
        choices=["all", "data", "train", "adversarial", "failsafe", "explain"],
        default="all",
        help="Pipeline step to run"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run pipeline
    if args.step == "all":
        success = run_complete_pipeline()
    else:
        logger.info(f"Running step: {args.step}")
        # Individual step execution can be implemented here
        success = run_complete_pipeline()  # For now, run complete pipeline
    
    if success:
        logger.info("✅ Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
