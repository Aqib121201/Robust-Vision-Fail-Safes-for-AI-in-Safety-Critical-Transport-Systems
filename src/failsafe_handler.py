"""
Fail-safe handler module for Robust Vision Fail-Safes for AI in Safety-Critical Transport Systems
Implements safety mechanisms and fallback actions for handling model failures and low confidence predictions.
"""

import numpy as np
import tensorflow as tf
import logging
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
import time
from dataclasses import dataclass
import json

from .config import (
    FAILSAFE_CONFIG, SAFETY_THRESHOLDS, VEHICLE_CLASSES, 
    SAFETY_CRITICAL_CLASSES, LOGGING_CONFIG
)

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class SafetyLevel(Enum):
    """Safety levels for different failure scenarios."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class FallbackAction(Enum):
    """Available fallback actions when failures are detected."""
    STOP = "stop"
    SLOW_DOWN = "slow_down"
    HUMAN_INTERVENTION = "human_intervention"
    USE_BACKUP_MODEL = "use_backup_model"
    CONTINUE_WITH_CAUTION = "continue_with_caution"

@dataclass
class SafetyEvent:
    """Data class for safety events."""
    timestamp: float
    safety_level: SafetyLevel
    confidence: float
    prediction: int
    true_label: Optional[int] = None
    failure_type: str = ""
    action_taken: FallbackAction = FallbackAction.CONTINUE_WITH_CAUTION
    metadata: Dict[str, Any] = None

class FailSafeHandler:
    """
    Fail-safe handler for safety-critical vision systems.
    Monitors model predictions and implements appropriate safety measures.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the fail-safe handler.
        
        Args:
            config: Configuration dictionary for fail-safe settings
        """
        self.config = config or FAILSAFE_CONFIG
        self.safety_events = []
        self.failure_count = 0
        self.last_action_time = time.time()
        self.backup_model = None
        self.ensemble_models = []
        
        logger.info("FailSafeHandler initialized")
    
    def check_prediction_safety(self, 
                               prediction: np.ndarray, 
                               confidence: float,
                               true_label: Optional[int] = None) -> Tuple[SafetyLevel, FallbackAction]:
        """
        Check if a prediction is safe and determine appropriate action.
        
        Args:
            prediction: Model prediction probabilities
            confidence: Prediction confidence
            true_label: True label (if available)
            
        Returns:
            Tuple of (safety_level, fallback_action)
        """
        predicted_class = np.argmax(prediction)
        max_confidence = np.max(prediction)
        
        # Check confidence threshold
        if max_confidence < self.config['confidence_threshold']:
            safety_level = SafetyLevel.CRITICAL
            action = self._determine_fallback_action(safety_level, predicted_class)
            
            self._log_safety_event(
                SafetyLevel.CRITICAL, max_confidence, predicted_class, 
                true_label, "low_confidence", action
            )
            return safety_level, action
        
        # Check uncertainty threshold
        uncertainty = 1.0 - max_confidence
        if uncertainty > self.config['uncertainty_threshold']:
            safety_level = SafetyLevel.WARNING
            action = self._determine_fallback_action(safety_level, predicted_class)
            
            self._log_safety_event(
                SafetyLevel.WARNING, max_confidence, predicted_class,
                true_label, "high_uncertainty", action
            )
            return safety_level, action
        
        # Check if prediction is for safety-critical class
        if predicted_class in SAFETY_CRITICAL_CLASSES.values():
            if max_confidence < self.config['confidence_threshold'] * 1.2:  # Higher threshold for safety-critical
                safety_level = SafetyLevel.CRITICAL
                action = self._determine_fallback_action(safety_level, predicted_class)
                
                self._log_safety_event(
                    SafetyLevel.CRITICAL, max_confidence, predicted_class,
                    true_label, "safety_critical_low_confidence", action
                )
                return safety_level, action
        
        # Normal operation
        safety_level = SafetyLevel.NORMAL
        action = FallbackAction.CONTINUE_WITH_CAUTION
        
        return safety_level, action
    
    def _determine_fallback_action(self, safety_level: SafetyLevel, predicted_class: int) -> FallbackAction:
        """
        Determine appropriate fallback action based on safety level and prediction.
        
        Args:
            safety_level: Current safety level
            predicted_class: Predicted class
            
        Returns:
            Fallback action to take
        """
        if safety_level == SafetyLevel.EMERGENCY:
            return FallbackAction.STOP
        
        elif safety_level == SafetyLevel.CRITICAL:
            # For safety-critical classes, be more conservative
            if predicted_class in SAFETY_CRITICAL_CLASSES.values():
                return FallbackAction.STOP
            else:
                return FallbackAction.SLOW_DOWN
        
        elif safety_level == SafetyLevel.WARNING:
            return FallbackAction.SLOW_DOWN
        
        else:
            return FallbackAction.CONTINUE_WITH_CAUTION
    
    def _log_safety_event(self, safety_level: SafetyLevel, confidence: float, 
                         prediction: int, true_label: Optional[int], 
                         failure_type: str, action: FallbackAction):
        """Log a safety event."""
        event = SafetyEvent(
            timestamp=time.time(),
            safety_level=safety_level,
            confidence=confidence,
            prediction=prediction,
            true_label=true_label,
            failure_type=failure_type,
            action_taken=action,
            metadata={
                'failure_count': self.failure_count,
                'time_since_last_action': time.time() - self.last_action_time
            }
        )
        
        self.safety_events.append(event)
        self.failure_count += 1
        self.last_action_time = time.time()
        
        logger.warning(f"Safety event: {safety_level.value} - {failure_type} - Action: {action.value}")
    
    def execute_fallback_action(self, action: FallbackAction, 
                              prediction: np.ndarray, 
                              image: np.ndarray) -> Dict[str, Any]:
        """
        Execute the determined fallback action.
        
        Args:
            action: Fallback action to execute
            prediction: Original model prediction
            image: Input image
            
        Returns:
            Dictionary with action results
        """
        result = {
            'action_executed': action.value,
            'timestamp': time.time(),
            'success': True,
            'message': '',
            'new_prediction': None,
            'confidence': None
        }
        
        try:
            if action == FallbackAction.STOP:
                result['message'] = "System stopped due to safety concerns"
                self._execute_stop_action()
                
            elif action == FallbackAction.SLOW_DOWN:
                result['message'] = "System slowed down due to uncertainty"
                self._execute_slow_down_action()
                
            elif action == FallbackAction.HUMAN_INTERVENTION:
                result['message'] = "Requesting human intervention"
                self._execute_human_intervention()
                
            elif action == FallbackAction.USE_BACKUP_MODEL:
                if self.backup_model is not None:
                    backup_prediction = self.backup_model.predict(image[np.newaxis, ...])
                    result['new_prediction'] = backup_prediction[0]
                    result['confidence'] = np.max(backup_prediction[0])
                    result['message'] = "Using backup model prediction"
                else:
                    result['success'] = False
                    result['message'] = "Backup model not available"
                    
            elif action == FallbackAction.CONTINUE_WITH_CAUTION:
                result['message'] = "Continuing with caution"
                
        except Exception as e:
            result['success'] = False
            result['message'] = f"Error executing action: {str(e)}"
            logger.error(f"Error executing fallback action: {e}")
        
        return result
    
    def _execute_stop_action(self):
        """Execute stop action."""
        logger.critical("STOP ACTION EXECUTED - System halted for safety")
        # In a real system, this would trigger emergency braking, etc.
        
    def _execute_slow_down_action(self):
        """Execute slow down action."""
        logger.warning("SLOW DOWN ACTION EXECUTED - Reducing system speed")
        # In a real system, this would reduce vehicle speed
        
    def _execute_human_intervention(self):
        """Execute human intervention action."""
        logger.warning("HUMAN INTERVENTION REQUESTED - Manual control needed")
        # In a real system, this would alert the driver
    
    def add_backup_model(self, model: tf.keras.Model):
        """Add a backup model for redundancy."""
        self.backup_model = model
        logger.info("Backup model added")
    
    def add_ensemble_model(self, model: tf.keras.Model):
        """Add a model to the ensemble."""
        self.ensemble_models.append(model)
        logger.info(f"Ensemble model added. Total: {len(self.ensemble_models)}")
    
    def get_ensemble_prediction(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Get ensemble prediction from multiple models.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (ensemble_prediction, confidence)
        """
        if not self.ensemble_models:
            raise ValueError("No ensemble models available")
        
        predictions = []
        for model in self.ensemble_models:
            pred = model.predict(image[np.newaxis, ...], verbose=0)
            predictions.append(pred[0])
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        confidence = np.max(ensemble_pred)
        
        return ensemble_pred, confidence
    
    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get safety statistics and failure analysis."""
        if not self.safety_events:
            return {'total_events': 0}
        
        stats = {
            'total_events': len(self.safety_events),
            'failure_count': self.failure_count,
            'safety_levels': {},
            'failure_types': {},
            'actions_taken': {},
            'confidence_stats': {
                'min': min(e.confidence for e in self.safety_events),
                'max': max(e.confidence for e in self.safety_events),
                'mean': np.mean([e.confidence for e in self.safety_events]),
                'std': np.std([e.confidence for e in self.safety_events])
            }
        }
        
        # Count safety levels
        for event in self.safety_events:
            level = event.safety_level.value
            stats['safety_levels'][level] = stats['safety_levels'].get(level, 0) + 1
            
            failure_type = event.failure_type
            stats['failure_types'][failure_type] = stats['failure_types'].get(failure_type, 0) + 1
            
            action = event.action_taken.value
            stats['actions_taken'][action] = stats['actions_taken'].get(action, 0) + 1
        
        return stats
    
    def save_safety_log(self, filepath: str = "safety_events.json"):
        """Save safety events to JSON file."""
        events_data = []
        for event in self.safety_events:
            event_dict = {
                'timestamp': event.timestamp,
                'safety_level': event.safety_level.value,
                'confidence': event.confidence,
                'prediction': event.prediction,
                'true_label': event.true_label,
                'failure_type': event.failure_type,
                'action_taken': event.action_taken.value,
                'metadata': event.metadata
            }
            events_data.append(event_dict)
        
        with open(filepath, 'w') as f:
            json.dump(events_data, f, indent=2)
        
        logger.info(f"Safety log saved to {filepath}")
    
    def reset_safety_counters(self):
        """Reset safety counters and events."""
        self.safety_events = []
        self.failure_count = 0
        self.last_action_time = time.time()
        logger.info("Safety counters reset")

class SafetyMonitor:
    """
    Continuous safety monitor for real-time failure detection.
    """
    
    def __init__(self, failsafe_handler: FailSafeHandler):
        """
        Initialize the safety monitor.
        
        Args:
            failsafe_handler: Fail-safe handler instance
        """
        self.failsafe_handler = failsafe_handler
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("SafetyMonitor initialized")
    
    def start_monitoring(self):
        """Start continuous safety monitoring."""
        self.monitoring_active = True
        logger.info("Safety monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous safety monitoring."""
        self.monitoring_active = False
        logger.info("Safety monitoring stopped")
    
    def process_prediction(self, prediction: np.ndarray, image: np.ndarray, 
                          true_label: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a model prediction and check safety.
        
        Args:
            prediction: Model prediction
            image: Input image
            true_label: True label (if available)
            
        Returns:
            Dictionary with safety check results and actions taken
        """
        confidence = np.max(prediction)
        
        # Check safety
        safety_level, fallback_action = self.failsafe_handler.check_prediction_safety(
            prediction, confidence, true_label
        )
        
        # Execute action if needed
        action_result = self.failsafe_handler.execute_fallback_action(
            fallback_action, prediction, image
        )
        
        return {
            'safety_level': safety_level.value,
            'confidence': confidence,
            'prediction': np.argmax(prediction),
            'fallback_action': fallback_action.value,
            'action_result': action_result,
            'timestamp': time.time()
        }

def main():
    """Main function for testing fail-safe handler."""
    # Create a dummy model prediction
    dummy_prediction = np.array([0.1, 0.8, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005])
    dummy_image = np.random.random((32, 32, 3))
    
    # Initialize fail-safe handler
    handler = FailSafeHandler()
    
    # Test safety check
    safety_level, action = handler.check_prediction_safety(dummy_prediction, 0.8)
    print(f"Safety Level: {safety_level.value}")
    print(f"Fallback Action: {action.value}")
    
    # Test with low confidence
    low_conf_prediction = np.array([0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.05, 0.02, 0.02, 0.01])
    safety_level, action = handler.check_prediction_safety(low_conf_prediction, 0.3)
    print(f"Low Confidence - Safety Level: {safety_level.value}")
    print(f"Low Confidence - Fallback Action: {action.value}")
    
    # Get statistics
    stats = handler.get_safety_statistics()
    print(f"Safety Statistics: {stats}")

if __name__ == "__main__":
    main()
