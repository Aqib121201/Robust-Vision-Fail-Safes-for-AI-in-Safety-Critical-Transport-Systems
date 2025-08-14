"""
Model training module for Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems
Implements CNN architecture and training procedures with safety-critical considerations.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import logging
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from datetime import datetime

from .config import (
    MODEL_CONFIG, CNN_CONFIG, EVALUATION_CONFIG, 
    MODELS_DIR, VISUALIZATIONS_DIR, LOGGING_CONFIG
)

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class VehicleClassifier:
    """CNN-based vehicle classifier with safety-critical considerations."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or MODEL_CONFIG
        self.model = None
        self.history = None
        self.training_time = None
        
        np.random.seed(EVALUATION_CONFIG['random_seed'])
        tf.random.set_seed(EVALUATION_CONFIG['random_seed'])
        logger.info("VehicleClassifier initialized")
    
    def build_cnn_model(self) -> tf.keras.Model:
        """Build a custom CNN architecture for vehicle classification."""
        logger.info("Building CNN model...")
        
        model = models.Sequential()
        model.add(layers.Input(shape=self.config['input_shape']))
        
        # Convolutional layers
        for i, (filters, kernel_size, pool_size) in enumerate(zip(
            CNN_CONFIG['filters'], 
            CNN_CONFIG['kernel_sizes'], 
            CNN_CONFIG['pool_sizes']
        )):
            model.add(layers.Conv2D(filters, (kernel_size, kernel_size), 
                                  activation='relu', padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D((pool_size, pool_size)))
            model.add(layers.Dropout(CNN_CONFIG['dropout_rate']))
        
        # Dense layers
        model.add(layers.Flatten())
        for units in CNN_CONFIG['dense_units']:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(CNN_CONFIG['dropout_rate']))
        
        model.add(layers.Dense(self.config['num_classes'], activation='softmax'))
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        logger.info(f"CNN model built with {model.count_params():,} parameters")
        return model
    
    def create_callbacks(self) -> list:
        """Create training callbacks."""
        return [
            callbacks.ModelCheckpoint(
                filepath=self.config['model_save_path'],
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
    
    def train_model(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> Dict[str, Any]:
        """Train the vehicle classification model."""
        logger.info("Starting model training...")
        
        self.build_cnn_model()
        callbacks_list = self.create_callbacks()
        
        start_time = datetime.now()
        
        self.history = self.model.fit(
            train_dataset,
            epochs=self.config['epochs'],
            validation_data=val_dataset,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        return {
            'history': self.history.history,
            'training_time': self.training_time,
            'model': self.model
        }
    
    def evaluate_model(self, test_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Evaluate the trained model on test data."""
        if self.model is None:
            logger.error("No trained model available for evaluation")
            return {}
        
        logger.info("Evaluating model on test data...")
        
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            test_dataset, verbose=0
        )
        
        predictions = self.model.predict(test_dataset)
        y_true = []
        y_pred = []
        
        for x_batch, y_batch in test_dataset:
            y_true.extend(np.argmax(y_batch.numpy(), axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
        
        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': f1
        }
        
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test F1 Score: {f1:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str = None):
        """Save the trained model to disk."""
        if self.model is None:
            logger.error("No model to save")
            return
        
        if filepath is None:
            filepath = self.config['model_save_path']
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load a trained model from disk."""
        if filepath is None:
            filepath = self.config['model_save_path']
        
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

def main():
    """Main function for testing model training."""
    from .data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    train_dataset, val_dataset, test_dataset = preprocessor.get_data_generators()
    
    classifier = VehicleClassifier()
    training_results = classifier.train_model(train_dataset, val_dataset)
    metrics = classifier.evaluate_model(test_dataset)
    classifier.save_model()
    
    print("Training completed successfully!")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
