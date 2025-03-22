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
            # We'll concatenate [state, action, next_state, predicted_next_state, uncertainty]
            input_dim = state_dim + action_dim + state_dim + state_dim + hidden_dim
        
        # Ensure input_dim is a power of 2 for better compatibility
        target_input_dim = 2048  # Explicitly set to match the expected size in detect_errors method
        
        # Error detection core network with adaptive input layer
        self.detection_network = nn.Sequential(
            nn.Linear(target_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        ).to(self.device)
        
        # Store the expected input dimension for proper padding in detect_errors
        self.expected_input_dim = target_input_dim
        
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
            state: Current state representation
            action: Action taken
            next_state: Actual next state (optional)
            predicted_next_state: Predicted next state (optional)
            uncertainty: Uncertainty in prediction (optional)
            
        Returns:
            Dictionary with error detection results
        """
        try:
            # Log input shapes for debugging
            logger.debug(f"Error Detection Input Shapes - state: {state.shape if hasattr(state, 'shape') else 'unknown'}, " +
                         f"action: {action.shape if hasattr(action, 'shape') else 'unknown'}")
            if next_state is not None:
                logger.debug(f"next_state shape: {next_state.shape}")
            if predicted_next_state is not None:
                logger.debug(f"predicted_next_state shape: {predicted_next_state.shape}")
            if uncertainty is not None:
                logger.debug(f"uncertainty shape: {uncertainty.shape}")
            
            # Check for NaN values in all input tensors and replace them
            if torch.is_tensor(state) and (torch.isnan(state).any() or torch.isinf(state).any()):
                logger.warning("NaN/Inf values detected in state tensor, replacing with zeros")
                state = torch.where(torch.isnan(state) | torch.isinf(state), torch.zeros_like(state), state)
                
            if torch.is_tensor(action) and (torch.isnan(action).any() or torch.isinf(action).any()):
                logger.warning("NaN/Inf values detected in action tensor, replacing with zeros")
                action = torch.where(torch.isnan(action) | torch.isinf(action), torch.zeros_like(action), action)
                
            if next_state is not None and torch.is_tensor(next_state) and (torch.isnan(next_state).any() or torch.isinf(next_state).any()):
                logger.warning("NaN/Inf values detected in next_state tensor, replacing with zeros")
                next_state = torch.where(torch.isnan(next_state) | torch.isinf(next_state), torch.zeros_like(next_state), next_state)
                
            if predicted_next_state is not None and torch.is_tensor(predicted_next_state) and (torch.isnan(predicted_next_state).any() or torch.isinf(predicted_next_state).any()):
                logger.warning("NaN/Inf values detected in predicted_next_state tensor, replacing with zeros")
                predicted_next_state = torch.where(torch.isnan(predicted_next_state) | torch.isinf(predicted_next_state), torch.zeros_like(predicted_next_state), predicted_next_state)
                
            if uncertainty is not None and torch.is_tensor(uncertainty) and (torch.isnan(uncertainty).any() or torch.isinf(uncertainty).any()):
                logger.warning("NaN/Inf values detected in uncertainty tensor, replacing with zeros")
                uncertainty = torch.where(torch.isnan(uncertainty) | torch.isinf(uncertainty), torch.zeros_like(uncertainty), uncertainty)
                
            # Ensure inputs are on the correct device
            state = state.to(self.device)
            action = action.to(self.device)
            
            if next_state is not None:
                next_state = next_state.to(self.device)
            
            if predicted_next_state is not None:
                predicted_next_state = predicted_next_state.to(self.device)
                
            if uncertainty is not None:
                uncertainty = uncertainty.to(self.device)
            
            # Handle feature dimensionality to match expected input size of the detection network
            # Combine all available information into a detector input
            detector_components = []
            
            # Process state features
            if self.state_dim != state.shape[-1]:
                logger.debug(f"State dimension mismatch: expected {self.state_dim}, got {state.shape[-1]} - adapting")
                if state.shape[-1] > self.state_dim:
                    # If input is larger, we can reduce it with a projection
                    state_features = state[:, :self.state_dim]
                else:
                    # If input is smaller, pad with zeros
                    padding = torch.zeros(state.size(0), self.state_dim - state.shape[-1], device=self.device)
                    state_features = torch.cat([state, padding], dim=1)
            else:
                state_features = state
                
            detector_components.append(state_features)
            
            # Process action features
            if isinstance(action, torch.Tensor) and action.dim() > 0:
                # For one-hot or continuous actions
                # First flatten action if it has more than one dimension
                if action.dim() > 1 and action.dtype == torch.long:
                    action = action.flatten()
                
                if action.dim() == 1 and action.dtype == torch.long:
                    # Convert scalar action to one-hot
                    one_hot_action = torch.zeros(1, self.action_dim, device=self.device)
                    # Ensure action is within valid range
                    action_idx = max(0, min(action.item(), self.action_dim - 1))
                    one_hot_action[0, action_idx] = 1.0
                    action_features = one_hot_action
                elif action.dim() == 2 and action.size(1) == 1 and action.dtype == torch.long:
                    one_hot_action = torch.zeros(action.size(0), self.action_dim, device=self.device)
                    action_indices = torch.clamp(action, 0, self.action_dim - 1)
                    one_hot_action.scatter_(1, action_indices, 1)
                    action_features = one_hot_action
                else:
                    # Ensure proper dimensions
                    if action.size(-1) != self.action_dim:
                        if action.size(-1) > self.action_dim:
                            action_features = action[:, :self.action_dim]
                        else:
                            # Pad with zeros
                            padding = torch.zeros(action.size(0), self.action_dim - action.size(-1), device=self.device)
                            action_features = torch.cat([action, padding], dim=1)
                    else:
                        action_features = action
            else:
                # For single scalar actions, convert to one-hot
                action_idx = action.item() if isinstance(action, torch.Tensor) else int(action)
                action_idx = max(0, min(action_idx, self.action_dim - 1))  # Clamp to valid range
                action_features = torch.zeros(1, self.action_dim, device=self.device)
                action_features[0, action_idx] = 1.0
                
            detector_components.append(action_features)
            
            # Add next state features if available
            if next_state is not None:
                if self.state_dim != next_state.shape[-1]:
                    if next_state.shape[-1] > self.state_dim:
                        next_state_features = next_state[:, :self.state_dim]
                    else:
                        padding = torch.zeros(next_state.size(0), self.state_dim - next_state.shape[-1], device=self.device)
                        next_state_features = torch.cat([next_state, padding], dim=1)
                else:
                    next_state_features = next_state
                    
                detector_components.append(next_state_features)
            else:
                # Add zeros if not available
                detector_components.append(torch.zeros_like(state_features))
                
            # Add predicted next state features if available
            if predicted_next_state is not None:
                if self.state_dim != predicted_next_state.shape[-1]:
                    if predicted_next_state.shape[-1] > self.state_dim:
                        pred_features = predicted_next_state[:, :self.state_dim]
                    else:
                        padding = torch.zeros(predicted_next_state.size(0), self.state_dim - predicted_next_state.shape[-1], device=self.device)
                        pred_features = torch.cat([predicted_next_state, padding], dim=1)
                else:
                    pred_features = predicted_next_state
                    
                detector_components.append(pred_features)
            else:
                detector_components.append(torch.zeros_like(state_features))
                
            # Add uncertainty information if available
            if uncertainty is not None:
                # Ensure proper dimensions
                if uncertainty.shape[-1] > self.hidden_dim:
                    uncertainty_features = uncertainty[:, :self.hidden_dim]
                else:
                    # Pad with zeros
                    padding = torch.zeros(uncertainty.size(0), self.hidden_dim - uncertainty.shape[-1], device=self.device)
                    uncertainty_features = torch.cat([uncertainty, padding], dim=1)
                    
                detector_components.append(uncertainty_features)
            else:
                detector_components.append(torch.zeros(state.size(0), self.hidden_dim, device=self.device))
            
            # Prepare input based on available data
            if self.use_world_model and predicted_next_state is not None and next_state is not None:
                # Full information available for error detection
                if uncertainty is None:
                    # If uncertainty is not provided, create a dummy tensor
                    uncertainty = torch.zeros_like(state).to(self.device)
                    
                detector_input = torch.cat(detector_components, dim=-1)
                logger.debug(f"Created detector_input from all components, shape: {detector_input.shape}")
            else:
                # Use only current state for error detection
                detector_input = torch.cat(detector_components[:1], dim=-1)
                logger.debug(f"Using only state for detector_input, shape: {detector_input.shape}")
            
            # Log expected input dimensions for the detection network
            detection_layers = [m for m in self.detection_network.modules() if isinstance(m, nn.Linear)]
            if detection_layers:
                first_layer = detection_layers[0]
                logger.debug(f"First detection layer expects input dim: {first_layer.in_features}, " + 
                           f"detector_input has dim: {detector_input.shape[-1]}")
                
                # Final safety check - if dimensions don't match, adapt the input
                if detector_input.shape[-1] != first_layer.in_features:
                    logger.warning(f"Dimension mismatch: detector_input has {detector_input.shape[-1]} features, " + 
                                 f"but first layer expects {first_layer.in_features}")
                    
                    # Use a simple projection or resize approach to match dimensions
                    if detector_input.shape[-1] > first_layer.in_features:
                        detector_input = detector_input[..., :first_layer.in_features]
                        logger.debug(f"Truncated detector_input to match network, new shape: {detector_input.shape}")
                    else:
                        # Pad with zeros to match expected dimension
                        padding = torch.zeros(detector_input.size(0), 
                                             first_layer.in_features - detector_input.shape[-1], 
                                             device=self.device)
                        detector_input = torch.cat([detector_input, padding], dim=1)
                        logger.debug(f"Padded detector_input to match network, new shape: {detector_input.shape}")
            
            # Detect errors
            try:
                features = self.detection_network(detector_input)
                error_types = self.error_classifier(features)
                error_severity = self.severity_regressor(features)
            except Exception as e:
                logger.error(f"Error in detection network forward pass: {e}")
                # Return fallback results and exit early
                batch_size = state.shape[0] if isinstance(state, torch.Tensor) and state.dim() > 0 else 1
                return self._create_fallback_detection_results(batch_size)
            
            # Check action inconsistency
            try:
                # Handle action for action_inconsistency_detector
                if action.dtype == torch.long:
                    # Flatten action if needed
                    if action.dim() > 1:
                        action_flat = action.flatten()
                    else:
                        action_flat = action
                        
                    # Ensure action indices are in range
                    action_flat = torch.clamp(action_flat, 0, self.action_dim - 1)
                    action_one_hot = F.one_hot(action_flat, num_classes=self.action_dim).float()
                else:
                    action_one_hot = action
                    
                # Make sure the first dimension is a batch dimension
                if action_one_hot.dim() == 1:
                    action_one_hot = action_one_hot.unsqueeze(0)
                
                # Ensure state and action_one_hot have same number of dimensions
                if state.dim() != action_one_hot.dim():
                    if state.dim() > action_one_hot.dim():
                        # Add dimensions to action_one_hot
                        for _ in range(state.dim() - action_one_hot.dim()):
                            action_one_hot = action_one_hot.unsqueeze(-1)
                    else:
                        # Add dimensions to state
                        for _ in range(action_one_hot.dim() - state.dim()):
                            state = state.unsqueeze(-1)
                
                # Make sure batch dimensions match
                if state.size(0) != action_one_hot.size(0):
                    # Try to broadcast
                    if state.size(0) == 1:
                        state = state.expand(action_one_hot.size(0), *state.shape[1:])
                    elif action_one_hot.size(0) == 1:
                        action_one_hot = action_one_hot.expand(state.size(0), *action_one_hot.shape[1:])
                    else:
                        logger.warning(f"Incompatible batch dimensions: state={state.size(0)}, action={action_one_hot.size(0)}")
                        # Just take first element as fallback
                        state = state[:1]
                        action_one_hot = action_one_hot[:1]
                
                # Now try to concatenate along last dimension
                state_action_combined = torch.cat([state, action_one_hot], dim=-1)
                action_inconsistency = self.action_inconsistency_detector(state_action_combined)
            
            except Exception as e:
                import traceback
                logger.error(f"Error in action inconsistency detection: {e}")
                logger.error(traceback.format_exc())
                # Fallback to zeros
                action_inconsistency = torch.zeros(state.shape[0] if state.dim() > 0 else 1, 1, device=self.device)
            
            # Check state anomaly
            try:
                state_anomaly = self.state_anomaly_detector(state)
            except Exception as e:
                logger.error(f"Error in state anomaly detection: {e}")
                # Fallback to zeros
                state_anomaly = torch.zeros(state.shape[0] if state.dim() > 0 else 1, 1, device=self.device)
            
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
            
        except Exception as e:
            import traceback
            logger.error(f"Error in error detection: {e}")
            logger.error(traceback.format_exc())
            # Determine batch size if possible
            batch_size = None
            if isinstance(state, torch.Tensor) and state.dim() > 0:
                batch_size = state.shape[0]
            return self._create_fallback_detection_results(batch_size)
    
    def _create_fallback_detection_results(self, batch_size=None) -> Dict[str, torch.Tensor]:
        """Create fallback detection results when error detection fails.
        
        Args:
            batch_size: Optional batch size for results, defaults to 1 if not provided
        """
        # Use provided batch_size or default to 1
        if batch_size is None:
            batch_size = 1
            
        # Ensure batch_size is an integer
        try:
            batch_size = int(batch_size)
        except (ValueError, TypeError):
            batch_size = 1
        
        # Create safe fallback results
        result = {
            'error_types': torch.zeros(batch_size, self.error_types, device=self.device),
            'error_severity': torch.zeros(batch_size, 1, device=self.device).fill_(0.5),  # Neutral severity
            'action_inconsistency': torch.zeros(batch_size, 1, device=self.device),
            'state_anomaly': torch.zeros(batch_size, 1, device=self.device),
            'prediction_error': torch.zeros(batch_size, device=self.device)
        }
        
        logger.warning("Returning fallback error detection results")
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
            
            # Handle batch dimension if present
            if len(scores.shape) > 1:
                # If we have a batch, take the first element for simplicity
                scores = scores[0]
        else:
            scores = error_scores
            
        # Generate explanations for each error type that exceeds threshold
        for i, score in enumerate(scores):
            # Use scalar comparison for each element
            if float(score) >= threshold:  # Convert to float for safe comparison
                if i < len(self.error_labels):
                    error_label = self.error_labels[i]
                    
                    if error_label == "perception_error":
                        explanations.append(
                            f"Perception error detected (confidence: {float(score):.2f}): "
                            "The agent may not correctly perceive the current state."
                        )
                    elif error_label == "planning_error":
                        explanations.append(
                            f"Planning error detected (confidence: {float(score):.2f}): "
                            "The agent's action planning may be suboptimal."
                        )
                    elif error_label == "execution_error":
                        explanations.append(
                            f"Execution error detected (confidence: {float(score):.2f}): "
                            "The agent may have issues executing the intended action."
                        )
                    elif error_label == "prediction_error":
                        explanations.append(
                            f"Prediction error detected (confidence: {float(score):.2f}): "
                            "The agent's prediction of future states may be inaccurate."
                        )
                    elif error_label == "adaptation_error":
                        explanations.append(
                            f"Adaptation error detected (confidence: {float(score):.2f}): "
                            "The agent may be struggling to adapt to new or changing situations."
                        )
                else:
                    explanations.append(f"Unknown error type {i} detected (confidence: {float(score):.2f})")
        
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