"""
Unit tests for data preprocessing module.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

from src.data_preprocessing import DataPreprocessor
from src.config import DATASET_CONFIG, VEHICLE_CLASSES, SAFETY_CRITICAL_CLASSES

class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
    
    def test_initialization(self):
        """Test DataPreprocessor initialization."""
        assert self.preprocessor.config == DATASET_CONFIG
        assert self.preprocessor.class_names == VEHICLE_CLASSES
        assert self.preprocessor.safety_critical_classes == SAFETY_CRITICAL_CLASSES
        assert self.preprocessor.train_data is None
        assert self.preprocessor.val_data is None
        assert self.preprocessor.test_data is None
    
    @patch('src.data_preprocessing.cifar10.load_data')
    def test_load_cifar10_data(self, mock_load_data):
        """Test CIFAR-10 data loading."""
        # Mock CIFAR-10 data
        mock_x_train = np.random.randint(0, 255, (50000, 32, 32, 3), dtype=np.uint8)
        mock_y_train = np.random.randint(0, 10, (50000, 1), dtype=np.uint8)
        mock_x_test = np.random.randint(0, 255, (10000, 32, 32, 3), dtype=np.uint8)
        mock_y_test = np.random.randint(0, 10, (10000, 1), dtype=np.uint8)
        
        mock_load_data.return_value = ((mock_x_train, mock_y_train), (mock_x_test, mock_y_test))
        
        # Test data loading
        train_data, val_data, test_data = self.preprocessor.load_cifar10_data()
        
        # Verify data shapes
        x_train, y_train = train_data
        x_val, y_val = val_data
        x_test, y_test = test_data
        
        assert x_train.shape[0] == 40000  # 80% of 50000
        assert x_val.shape[0] == 10000    # 20% of 50000
        assert x_test.shape[0] == 1000    # 10% of 10000
        
        # Verify data normalization
        assert np.max(x_train) <= 1.0
        assert np.min(x_train) >= 0.0
        assert x_train.dtype == np.float32
        
        # Verify categorical labels
        assert y_train.shape[1] == 10  # 10 classes
        assert y_val.shape[1] == 10
        assert y_test.shape[1] == 10
    
    def test_get_safety_critical_samples(self):
        """Test extraction of safety-critical samples."""
        # Create mock data
        x_data = np.random.random((100, 32, 32, 3))
        y_data = np.zeros((100, 10))
        
        # Set some samples as safety-critical classes
        safety_indices = [0, 1, 2, 3]  # automobile, truck, airplane, ship
        for i, class_idx in enumerate(safety_indices):
            y_data[i, class_idx] = 1
        
        # Set other samples as non-safety-critical
        for i in range(4, 100):
            y_data[i, 5] = 1  # dog class
        
        self.preprocessor.train_data = (x_data, y_data)
        
        # Test safety-critical sample extraction
        x_safety, y_safety = self.preprocessor.get_safety_critical_samples('train')
        
        assert len(x_safety) == 4
        assert len(y_safety) == 4
        
        # Verify all extracted samples are safety-critical
        for i in range(len(y_safety)):
            class_idx = np.argmax(y_safety[i])
            assert class_idx in SAFETY_CRITICAL_CLASSES.values()
    
    def test_get_dataset_statistics(self):
        """Test dataset statistics calculation."""
        # Create mock data
        x_train = np.random.random((100, 32, 32, 3))
        y_train = np.zeros((100, 10))
        y_train[:, 0] = 1  # All samples are airplane class
        
        x_val = np.random.random((20, 32, 32, 3))
        y_val = np.zeros((20, 10))
        y_val[:, 1] = 1  # All samples are automobile class
        
        x_test = np.random.random((10, 32, 32, 3))
        y_test = np.zeros((10, 10))
        y_test[:, 9] = 1  # All samples are truck class
        
        self.preprocessor.train_data = (x_train, y_train)
        self.preprocessor.val_data = (x_val, y_val)
        self.preprocessor.test_data = (x_test, y_test)
        
        # Get statistics
        stats = self.preprocessor.get_dataset_statistics()
        
        # Verify statistics
        assert stats['total_samples'] == 130
        assert stats['train_samples'] == 100
        assert stats['val_samples'] == 20
        assert stats['test_samples'] == 10
        assert stats['image_shape'] == (32, 32, 3)
        assert stats['num_classes'] == 10
        
        # Verify class distribution
        assert stats['class_distribution']['train'][0] == 100  # All airplane
        assert stats['class_distribution']['val'][1] == 20     # All automobile
        assert stats['class_distribution']['test'][9] == 10    # All truck
    
    def test_data_generators(self):
        """Test TensorFlow data generator creation."""
        # Create mock data
        x_train = np.random.random((100, 32, 32, 3))
        y_train = np.zeros((100, 10))
        y_train[:, 0] = 1
        
        x_val = np.random.random((20, 32, 32, 3))
        y_val = np.zeros((20, 10))
        y_val[:, 1] = 1
        
        x_test = np.random.random((10, 32, 32, 3))
        y_test = np.zeros((10, 10))
        y_test[:, 9] = 1
        
        self.preprocessor.train_data = (x_train, y_train)
        self.preprocessor.val_data = (x_val, y_val)
        self.preprocessor.test_data = (x_test, y_test)
        
        # Get data generators
        train_dataset, val_dataset, test_dataset = self.preprocessor.get_data_generators()
        
        # Verify dataset types
        assert isinstance(train_dataset, tf.data.Dataset)
        assert isinstance(val_dataset, tf.data.Dataset)
        assert isinstance(test_dataset, tf.data.Dataset)
        
        # Verify dataset contents
        for x_batch, y_batch in train_dataset:
            assert x_batch.shape[0] <= DATASET_CONFIG['batch_size']
            assert y_batch.shape[1] == 10
            break
    
    def test_save_and_load_processed_data(self, tmp_path):
        """Test saving and loading processed data."""
        # Create mock data
        x_train = np.random.random((100, 32, 32, 3))
        y_train = np.zeros((100, 10))
        y_train[:, 0] = 1
        
        x_val = np.random.random((20, 32, 32, 3))
        y_val = np.zeros((20, 10))
        y_val[:, 1] = 1
        
        x_test = np.random.random((10, 32, 32, 3))
        y_test = np.zeros((10, 10))
        y_test[:, 9] = 1
        
        self.preprocessor.train_data = (x_train, y_train)
        self.preprocessor.val_data = (x_val, y_val)
        self.preprocessor.test_data = (x_test, y_test)
        
        # Save data
        save_path = tmp_path / "test_data.pkl"
        self.preprocessor.save_processed_data(str(save_path))
        
        # Create new preprocessor and load data
        new_preprocessor = DataPreprocessor()
        success = new_preprocessor.load_processed_data(str(save_path))
        
        assert success
        assert new_preprocessor.train_data is not None
        assert new_preprocessor.val_data is not None
        assert new_preprocessor.test_data is not None
        
        # Verify data integrity
        x_train_loaded, y_train_loaded = new_preprocessor.train_data
        np.testing.assert_array_equal(x_train, x_train_loaded)
        np.testing.assert_array_equal(y_train, y_train_loaded)

if __name__ == "__main__":
    pytest.main([__file__])
