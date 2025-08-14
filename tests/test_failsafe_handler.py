"""
Unit tests for fail-safe handler module.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch

from src.failsafe_handler import (
    FailSafeHandler, SafetyMonitor, SafetyLevel, FallbackAction, SafetyEvent
)
from src.config import FAILSAFE_CONFIG, SAFETY_CRITICAL_CLASSES

class TestFailSafeHandler:
    """Test cases for FailSafeHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.handler = FailSafeHandler()
    
    def test_initialization(self):
        """Test FailSafeHandler initialization."""
        assert self.handler.config == FAILSAFE_CONFIG
        assert self.handler.safety_events == []
        assert self.handler.failure_count == 0
        assert self.handler.backup_model is None
        assert self.handler.ensemble_models == []
    
    def test_check_prediction_safety_normal(self):
        """Test safety check with normal confidence."""
        # High confidence prediction
        prediction = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005])
        confidence = 0.8
        
        safety_level, action = self.handler.check_prediction_safety(prediction, confidence)
        
        assert safety_level == SafetyLevel.NORMAL
        assert action == FallbackAction.CONTINUE_WITH_CAUTION
        assert len(self.handler.safety_events) == 0  # No safety event logged
    
    def test_check_prediction_safety_low_confidence(self):
        """Test safety check with low confidence."""
        # Low confidence prediction
        prediction = np.array([0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.05, 0.02, 0.02, 0.01])
        confidence = 0.3
        
        safety_level, action = self.handler.check_prediction_safety(prediction, confidence)
        
        assert safety_level == SafetyLevel.CRITICAL
        assert action == FallbackAction.SLOW_DOWN
        assert len(self.handler.safety_events) == 1
        assert self.handler.failure_count == 1
    
    def test_check_prediction_safety_safety_critical(self):
        """Test safety check with safety-critical class and moderate confidence."""
        # Safety-critical class (automobile) with moderate confidence
        prediction = np.array([0.1, 0.6, 0.1, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])
        confidence = 0.6  # Below threshold * 1.2 for safety-critical
        
        safety_level, action = self.handler.check_prediction_safety(prediction, confidence)
        
        assert safety_level == SafetyLevel.CRITICAL
        assert action == FallbackAction.STOP  # Safety-critical class gets STOP action
        assert len(self.handler.safety_events) == 1
    
    def test_check_prediction_safety_high_uncertainty(self):
        """Test safety check with high uncertainty."""
        # High uncertainty prediction
        prediction = np.array([0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
        confidence = 0.15
        uncertainty = 1.0 - confidence  # 0.85
        
        safety_level, action = self.handler.check_prediction_safety(prediction, confidence)
        
        assert safety_level == SafetyLevel.WARNING
        assert action == FallbackAction.SLOW_DOWN
        assert len(self.handler.safety_events) == 1
    
    def test_determine_fallback_action(self):
        """Test fallback action determination."""
        # Test different safety levels
        assert self.handler._determine_fallback_action(SafetyLevel.NORMAL, 1) == FallbackAction.CONTINUE_WITH_CAUTION
        assert self.handler._determine_fallback_action(SafetyLevel.WARNING, 1) == FallbackAction.SLOW_DOWN
        assert self.handler._determine_fallback_action(SafetyLevel.CRITICAL, 1) == FallbackAction.SLOW_DOWN
        assert self.handler._determine_fallback_action(SafetyLevel.EMERGENCY, 1) == FallbackAction.STOP
        
        # Test safety-critical class with critical level
        safety_critical_class = list(SAFETY_CRITICAL_CLASSES.values())[0]
        assert self.handler._determine_fallback_action(SafetyLevel.CRITICAL, safety_critical_class) == FallbackAction.STOP
    
    def test_execute_fallback_action(self):
        """Test fallback action execution."""
        prediction = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005])
        image = np.random.random((32, 32, 3))
        
        # Test STOP action
        result = self.handler.execute_fallback_action(FallbackAction.STOP, prediction, image)
        assert result['action_executed'] == 'stop'
        assert result['success'] == True
        assert 'System stopped' in result['message']
        
        # Test SLOW_DOWN action
        result = self.handler.execute_fallback_action(FallbackAction.SLOW_DOWN, prediction, image)
        assert result['action_executed'] == 'slow_down'
        assert result['success'] == True
        assert 'System slowed down' in result['message']
        
        # Test HUMAN_INTERVENTION action
        result = self.handler.execute_fallback_action(FallbackAction.HUMAN_INTERVENTION, prediction, image)
        assert result['action_executed'] == 'human_intervention'
        assert result['success'] == True
        assert 'Requesting human intervention' in result['message']
        
        # Test CONTINUE_WITH_CAUTION action
        result = self.handler.execute_fallback_action(FallbackAction.CONTINUE_WITH_CAUTION, prediction, image)
        assert result['action_executed'] == 'continue_with_caution'
        assert result['success'] == True
        assert 'Continuing with caution' in result['message']
    
    def test_execute_fallback_action_with_backup_model(self):
        """Test fallback action with backup model."""
        # Create mock backup model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([[0.1, 0.7, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005]])
        self.handler.backup_model = mock_model
        
        prediction = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005])
        image = np.random.random((32, 32, 3))
        
        result = self.handler.execute_fallback_action(FallbackAction.USE_BACKUP_MODEL, prediction, image)
        assert result['action_executed'] == 'use_backup_model'
        assert result['success'] == True
        assert result['new_prediction'] is not None
        assert result['confidence'] == 0.7
    
    def test_execute_fallback_action_no_backup_model(self):
        """Test fallback action without backup model."""
        prediction = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005])
        image = np.random.random((32, 32, 3))
        
        result = self.handler.execute_fallback_action(FallbackAction.USE_BACKUP_MODEL, prediction, image)
        assert result['action_executed'] == 'use_backup_model'
        assert result['success'] == False
        assert 'Backup model not available' in result['message']
    
    def test_get_safety_statistics_empty(self):
        """Test safety statistics with no events."""
        stats = self.handler.get_safety_statistics()
        assert stats['total_events'] == 0
    
    def test_get_safety_statistics_with_events(self):
        """Test safety statistics with events."""
        # Create some safety events
        prediction = np.array([0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.05, 0.02, 0.02, 0.01])
        confidence = 0.3
        
        self.handler.check_prediction_safety(prediction, confidence)
        self.handler.check_prediction_safety(prediction, confidence)
        
        stats = self.handler.get_safety_statistics()
        
        assert stats['total_events'] == 2
        assert stats['failure_count'] == 2
        assert 'critical' in stats['safety_levels']
        assert stats['safety_levels']['critical'] == 2
        assert 'low_confidence' in stats['failure_types']
        assert stats['failure_types']['low_confidence'] == 2
    
    def test_add_backup_model(self):
        """Test adding backup model."""
        mock_model = Mock()
        self.handler.add_backup_model(mock_model)
        assert self.handler.backup_model == mock_model
    
    def test_add_ensemble_model(self):
        """Test adding ensemble model."""
        mock_model1 = Mock()
        mock_model2 = Mock()
        
        self.handler.add_ensemble_model(mock_model1)
        assert len(self.handler.ensemble_models) == 1
        assert self.handler.ensemble_models[0] == mock_model1
        
        self.handler.add_ensemble_model(mock_model2)
        assert len(self.handler.ensemble_models) == 2
        assert self.handler.ensemble_models[1] == mock_model2
    
    def test_get_ensemble_prediction(self):
        """Test ensemble prediction."""
        # Create mock ensemble models
        mock_model1 = Mock()
        mock_model1.predict.return_value = np.array([[0.1, 0.7, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005]])
        
        mock_model2 = Mock()
        mock_model2.predict.return_value = np.array([[0.15, 0.6, 0.15, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005]])
        
        self.handler.add_ensemble_model(mock_model1)
        self.handler.add_ensemble_model(mock_model2)
        
        image = np.random.random((32, 32, 3))
        ensemble_pred, confidence = self.handler.get_ensemble_prediction(image)
        
        assert ensemble_pred.shape == (10,)
        assert confidence == 0.65  # Average of 0.7 and 0.6
    
    def test_get_ensemble_prediction_no_models(self):
        """Test ensemble prediction with no models."""
        image = np.random.random((32, 32, 3))
        
        with pytest.raises(ValueError, match="No ensemble models available"):
            self.handler.get_ensemble_prediction(image)
    
    def test_reset_safety_counters(self):
        """Test resetting safety counters."""
        # Create some events first
        prediction = np.array([0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.05, 0.02, 0.02, 0.01])
        confidence = 0.3
        
        self.handler.check_prediction_safety(prediction, confidence)
        assert len(self.handler.safety_events) == 1
        assert self.handler.failure_count == 1
        
        # Reset counters
        self.handler.reset_safety_counters()
        assert len(self.handler.safety_events) == 0
        assert self.handler.failure_count == 0

class TestSafetyMonitor:
    """Test cases for SafetyMonitor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.failsafe_handler = FailSafeHandler()
        self.monitor = SafetyMonitor(self.failsafe_handler)
    
    def test_initialization(self):
        """Test SafetyMonitor initialization."""
        assert self.monitor.failsafe_handler == self.failsafe_handler
        assert self.monitor.monitoring_active == False
        assert self.monitor.monitoring_thread is None
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        assert self.monitor.monitoring_active == False
        
        self.monitor.start_monitoring()
        assert self.monitor.monitoring_active == True
        
        self.monitor.stop_monitoring()
        assert self.monitor.monitoring_active == False
    
    def test_process_prediction(self):
        """Test processing prediction through safety monitor."""
        prediction = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005])
        image = np.random.random((32, 32, 3))
        true_label = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # automobile
        
        result = self.monitor.process_prediction(prediction, image, true_label)
        
        assert 'safety_level' in result
        assert 'confidence' in result
        assert 'prediction' in result
        assert 'fallback_action' in result
        assert 'action_result' in result
        assert 'timestamp' in result
        
        assert result['confidence'] == 0.8
        assert result['prediction'] == 1  # automobile class

if __name__ == "__main__":
    pytest.main([__file__])
