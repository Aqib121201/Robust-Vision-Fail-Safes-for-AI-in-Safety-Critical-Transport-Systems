"""
Data preprocessing module for Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems
Handles dataset loading, preprocessing, and augmentation for vehicle classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
from typing import Tuple, Dict, Any
import pickle
from pathlib import Path

from .config import (
    DATASET_CONFIG, PROCESSED_DATA_DIR, RAW_DATA_DIR, 
    VEHICLE_CLASSES, SAFETY_CRITICAL_CLASSES, LOGGING_CONFIG
)

# Setup logging
logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing class for vehicle classification dataset.
    Handles loading, preprocessing, and augmentation of CIFAR-10 data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data preprocessor.
        
        Args:
            config: Configuration dictionary for data preprocessing
        """
        self.config = config or DATASET_CONFIG
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_names = VEHICLE_CLASSES
        self.safety_critical_classes = SAFETY_CRITICAL_CLASSES
        
        logger.info("DataPreprocessor initialized")
    
    def load_cifar10_data(self) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Load CIFAR-10 dataset and split into train/val/test sets.
        
        Returns:
            Tuple of (train_data, val_data, test_data) where each is (X, y)
        """
        logger.info("Loading CIFAR-10 dataset...")
        
        # Load CIFAR-10 data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train = to_categorical(y_train, self.config['num_classes'])
        y_test = to_categorical(y_test, self.config['num_classes'])
        
        # Split training data into train and validation
        split_idx = int(len(x_train) * self.config['train_split'])
        x_val = x_train[split_idx:]
        y_val = y_train[split_idx:]
        x_train = x_train[:split_idx]
        y_train = y_train[:split_idx]
        
        # Split test data
        test_split_idx = int(len(x_test) * (1 - self.config['test_split']))
        x_test_final = x_test[test_split_idx:]
        y_test_final = y_test[test_split_idx:]
        
        self.train_data = (x_train, y_train)
        self.val_data = (x_val, y_val)
        self.test_data = (x_test_final, y_test_final)
        
        logger.info(f"Dataset loaded - Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test_final)}")
        
        return self.train_data, self.val_data, self.test_data
    
    def create_data_generators(self) -> Tuple[ImageDataGenerator, ImageDataGenerator, ImageDataGenerator]:
        """
        Create data generators for training, validation, and testing with augmentation.
        
        Returns:
            Tuple of (train_generator, val_generator, test_generator)
        """
        logger.info("Creating data generators with augmentation...")
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        # Validation and test data generators (no augmentation)
        val_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        
        return train_datagen, val_datagen, test_datagen
    
    def get_data_generators(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create TensorFlow data generators for efficient training.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.train_data is None:
            self.load_cifar10_data()
        
        x_train, y_train = self.train_data
        x_val, y_val = self.val_data
        x_test, y_test = self.test_data
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        
        # Configure datasets
        train_dataset = train_dataset.shuffle(
            self.config['shuffle_buffer']
        ).batch(self.config['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = val_dataset.batch(self.config['batch_size']).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.config['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        logger.info("TensorFlow data generators created")
        return train_dataset, val_dataset, test_dataset
    
    def get_safety_critical_samples(self, data_type: str = 'test') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract samples from safety-critical vehicle classes.
        
        Args:
            data_type: Type of data to extract ('train', 'val', 'test')
            
        Returns:
            Tuple of (X, y) for safety-critical samples
        """
        if self.test_data is None:
            self.load_cifar10_data()
        
        if data_type == 'train':
            x_data, y_data = self.train_data
        elif data_type == 'val':
            x_data, y_data = self.val_data
        else:
            x_data, y_data = self.test_data
        
        # Get indices of safety-critical classes
        safety_indices = []
        for class_name, class_idx in self.safety_critical_classes.items():
            class_samples = np.where(np.argmax(y_data, axis=1) == class_idx)[0]
            safety_indices.extend(class_samples)
        
        safety_indices = np.array(safety_indices)
        
        if len(safety_indices) == 0:
            logger.warning(f"No safety-critical samples found in {data_type} data")
            return np.array([]), np.array([])
        
        x_safety = x_data[safety_indices]
        y_safety = y_data[safety_indices]
        
        logger.info(f"Extracted {len(x_safety)} safety-critical samples from {data_type} data")
        return x_safety, y_safety
    
    def save_processed_data(self, filename: str = "processed_data.pkl"):
        """
        Save processed data to disk for later use.
        
        Args:
            filename: Name of the file to save
        """
        if self.train_data is None:
            logger.error("No data to save. Load data first.")
            return
        
        data_dict = {
            'train_data': self.train_data,
            'val_data': self.val_data,
            'test_data': self.test_data,
            'class_names': self.class_names,
            'safety_critical_classes': self.safety_critical_classes,
            'config': self.config
        }
        
        save_path = PROCESSED_DATA_DIR / filename
        with open(save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        
        logger.info(f"Processed data saved to {save_path}")
    
    def load_processed_data(self, filename: str = "processed_data.pkl") -> bool:
        """
        Load processed data from disk.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            True if successful, False otherwise
        """
        load_path = PROCESSED_DATA_DIR / filename
        
        if not load_path.exists():
            logger.error(f"Processed data file not found: {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            self.train_data = data_dict['train_data']
            self.val_data = data_dict['val_data']
            self.test_data = data_dict['test_data']
            self.class_names = data_dict['class_names']
            self.safety_critical_classes = data_dict['safety_critical_classes']
            self.config = data_dict['config']
            
            logger.info(f"Processed data loaded from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            return False
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        if self.train_data is None:
            self.load_cifar10_data()
        
        x_train, y_train = self.train_data
        x_val, y_val = self.val_data
        x_test, y_test = self.test_data
        
        # Class distribution
        train_classes = np.argmax(y_train, axis=1)
        val_classes = np.argmax(y_val, axis=1)
        test_classes = np.argmax(y_test, axis=1)
        
        stats = {
            'total_samples': len(x_train) + len(x_val) + len(x_test),
            'train_samples': len(x_train),
            'val_samples': len(x_val),
            'test_samples': len(x_test),
            'image_shape': x_train.shape[1:],
            'num_classes': self.config['num_classes'],
            'class_distribution': {
                'train': np.bincount(train_classes),
                'val': np.bincount(val_classes),
                'test': np.bincount(test_classes)
            },
            'safety_critical_samples': {
                'train': len(self.get_safety_critical_samples('train')[0]),
                'val': len(self.get_safety_critical_samples('val')[0]),
                'test': len(self.get_safety_critical_samples('test')[0])
            }
        }
        
        return stats

def main():
    """Main function for testing data preprocessing."""
    preprocessor = DataPreprocessor()
    
    # Load data
    train_data, val_data, test_data = preprocessor.load_cifar10_data()
    
    # Get statistics
    stats = preprocessor.get_dataset_statistics()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save processed data
    preprocessor.save_processed_data()
    
    # Test data generators
    train_dataset, val_dataset, test_dataset = preprocessor.get_data_generators()
    print(f"Data generators created - Train batches: {len(train_dataset)}, "
          f"Val batches: {len(val_dataset)}, Test batches: {len(test_dataset)}")

if __name__ == "__main__":
    main()
