"""
Error Detection Network for Cities: Skylines 2 agent.
Identifies potential problems in the agent's understanding or decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

class ErrorDetectionNetwork(nn.Module):
    """Error Detection Network for identifying problems in agent's understanding or decisions."""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 error_types: int = 5,
                 use_world_model: bool = True,
                 device=None):
        """Initialize the Error Detection Network.
        
        Args:
            state_dim: Dimension of the state representation
            action_dim: Dimension of the action space
            hidden_dim: Dimension of hidden layers
            error_types: Number of error types to detect
            use_world_model: Whether to use world model predictions
            device: Computation device
        """
        super(ErrorDetectionNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.error_types = error_types
        self.use_world_model = use_world_model
        
        # Input dimensions depend on whether we use world model
        input_dim = state_dim
        if use_world_model:
            # We'll concatenate [state, predicted_next_state, actual_next_state, uncertainty]
            input_dim = state_dim * 3 + state_dim  # Last state_dim is for uncertainty
        
        # Error detection core network
        self.detection_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        ).to(self.device)
        
        # Error type classification head
        self.error_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, error_types),
            nn.Sigmoid()  # Multi-label classification (multiple error types can be present)
        ).to(self.device)
        
        # Error severity regression head
        self.severity_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Normalized severity between 0 and 1
        ).to(self.device)
        
        # Action inconsistency detector (detects if action doesn't match state)
        self.action_inconsistency_detector = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # State anomaly detector (detects if state is unusual/unseen)
        self.state_anomaly_detector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Define error type labels
        self.error_labels = [
            "perception_error",  # Error in perceiving the environment
            "planning_error",    # Error in planning actions
            "execution_error",   # Error in executing actions
            "prediction_error",  # Error in predicting outcomes
            "adaptation_error"   # Error in adapting to new situations
        ]
        
        logger.critical(f"Initialized Error Detection Network on device {self.device} with {error_types} error types")
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def detect_errors(self, 
                    state: torch.Tensor, 
                    action: torch.Tensor, 
                    next_state: Optional[torch.Tensor] = None,
                    predicted_next_state: Optional[torch.Tensor] = None,
                    uncertainty: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Detect errors in the agent's understanding or decisions.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Actual next state (if available)
            predicted_next_state: Predicted next state from world model (if available)
            uncertainty: Uncertainty in prediction (if available)
            
        Returns:
            Dict containing error types, severity, and anomaly scores
        """
        # Ensure inputs are on the correct device
        state = state.to(self.device)
        action = action.to(self.device)
        
        if next_state is not None:
            next_state = next_state.to(self.device)
        
        if predicted_next_state is not None:
            predicted_next_state = predicted_next_state.to(self.device)
            
        if uncertainty is not None:
            uncertainty = uncertainty.to(self.device)
        
        # Prepare input based on available data
        if self.use_world_model and predicted_next_state is not None and next_state is not None:
            # Full information available for error detection
            if uncertainty is None:
                # If uncertainty is not provided, create a dummy tensor
                uncertainty = torch.zeros_like(state).to(self.device)
                
            detector_input = torch.cat([state, predicted_next_state, next_state, uncertainty], dim=-1)
        else:
            # Use only current state for error detection
            detector_input = state
        
        # Detect errors
        features = self.detection_network(detector_input)
        error_types = self.error_classifier(features)
        error_severity = self.severity_regressor(features)
        
        # Check action inconsistency
        if action.dtype == torch.long:
            action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        else:
            action_one_hot = action
            
        action_inconsistency = self.action_inconsistency_detector(
            torch.cat([state, action_one_hot], dim=-1)
        )
        
        # Check state anomaly
        state_anomaly = self.state_anomaly_detector(state)
        
        # Create result dictionary
        result = {
            'error_types': error_types,
            'error_severity': error_severity,
            'action_inconsistency': action_inconsistency,
            'state_anomaly': state_anomaly
        }
        
        # Add prediction error if we have both predicted and actual next states
        if predicted_next_state is not None and next_state is not None:
            prediction_error = F.mse_loss(predicted_next_state, next_state, reduction='none').mean(dim=-1)
            result['prediction_error'] = prediction_error
        
        return result
    
    def get_error_explanation(self, error_scores: torch.Tensor, threshold: float = 0.5) -> List[str]:
        """Generate human-readable explanations of detected errors.
        
        Args:
            error_scores: Scores for each error type (output from error_classifier)
            threshold: Threshold for considering an error present
            
        Returns:
            List of error explanations
        """
        explanations = []
        
        # Convert to numpy for easier handling
        if isinstance(error_scores, torch.Tensor):
            scores = error_scores.detach().cpu().numpy()
        else:
            scores = error_scores
            
        # Generate explanations for each error type that exceeds threshold
        for i, score in enumerate(scores):
            if score >= threshold:
                if i < len(self.error_labels):
                    error_label = self.error_labels[i]
                    
                    if error_label == "perception_error":
                        explanations.append(
                            f"Perception error detected (confidence: {score:.2f}): "
                            "The agent may not correctly perceive the current state."
                        )
                    elif error_label == "planning_error":
                        explanations.append(
                            f"Planning error detected (confidence: {score:.2f}): "
                            "The agent's action planning may be suboptimal."
                        )
                    elif error_label == "execution_error":
                        explanations.append(
                            f"Execution error detected (confidence: {score:.2f}): "
                            "The agent may have issues executing the intended action."
                        )
                    elif error_label == "prediction_error":
                        explanations.append(
                            f"Prediction error detected (confidence: {score:.2f}): "
                            "The agent's prediction of future states may be inaccurate."
                        )
                    elif error_label == "adaptation_error":
                        explanations.append(
                            f"Adaptation error detected (confidence: {score:.2f}): "
                            "The agent may be struggling to adapt to new or changing situations."
                        )
                else:
                    explanations.append(f"Unknown error type {i} detected (confidence: {score:.2f})")
        
        return explanations
    
    def forward(self, state, action, next_state=None, predicted_next_state=None, uncertainty=None):
        """Forward pass for error detection.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Actual next state (if available)
            predicted_next_state: Predicted next state from world model (if available)
            uncertainty: Uncertainty in prediction (if available)
            
        Returns:
            Tuple of (error_types, error_severity, action_inconsistency, state_anomaly)
        """
        result = self.detect_errors(state, action, next_state, predicted_next_state, uncertainty)
        
        return (
            result['error_types'],
            result['error_severity'],
            result['action_inconsistency'],
            result['state_anomaly']
        )
    
    def compute_loss(self, 
                   state, 
                   action, 
                   next_state, 
                   predicted_next_state,
                   target_errors=None,
                   target_severity=None):
        """Compute loss for training the error detection network.
        
        Args:
            state: Batch of states
            action: Batch of actions
            next_state: Batch of actual next states
            predicted_next_state: Batch of predicted next states
            target_errors: Target error types (if available for supervised training)
            target_severity: Target error severity (if available for supervised training)
            
        Returns:
            Dict of losses
        """
        # Get error detection outputs
        error_detection_result = self.detect_errors(
            state, action, next_state, predicted_next_state
        )
        
        # Initialize losses
        losses = {}
        
        # If we have target labels (supervised learning case)
        if target_errors is not None:
            # Multi-label classification loss for error types
            error_type_loss = F.binary_cross_entropy(
                error_detection_result['error_types'], target_errors
            )
            losses['error_type_loss'] = error_type_loss
        
        if target_severity is not None:
            # Regression loss for error severity
            severity_loss = F.mse_loss(
                error_detection_result['error_severity'], target_severity
            )
            losses['severity_loss'] = severity_loss
        
        # If we don't have target labels (self-supervised case)
        if target_errors is None and predicted_next_state is not None and next_state is not None:
            # Calculate prediction error as a proxy for error detection
            prediction_error = F.mse_loss(predicted_next_state, next_state, reduction='none').mean(dim=-1, keepdim=True)
            
            # The error detection network should predict high error when prediction error is high
            self_supervised_loss = F.mse_loss(
                error_detection_result['error_severity'], 
                torch.sigmoid(prediction_error * 5.0)  # Scale and sigmoid for 0-1 range
            )
            losses['self_supervised_loss'] = self_supervised_loss
        
        # Total loss is the sum of all losses
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses 