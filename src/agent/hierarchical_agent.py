"""
Hierarchical Agent for Cities: Skylines 2.
Integrates multiple specialized neural networks in a hierarchical architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque

from src.agent.memory_agent import MemoryAugmentedAgent
from src.memory.memory_augmented_network import MemoryAugmentedNetwork
from src.model.visual_understanding_network import VisualUnderstandingNetwork
from src.model.world_model import WorldModel
from src.model.error_detection_network import ErrorDetectionNetwork

logger = logging.getLogger(__name__)

class HierarchicalAgent(MemoryAugmentedAgent):
    """Hierarchical agent that integrates specialized neural networks."""
    
    def __init__(self,
                 policy_network,
                 observation_space,
                 action_space,
                 device: torch.device = None,
                 memory_size: int = 2000,
                 memory_use_prob: float = 0.8,
                 use_visual_network: bool = True,
                 use_world_model: bool = True,
                 use_error_detection: bool = True,
                 feature_dim: int = 512,
                 latent_dim: int = 256,
                 prediction_horizon: int = 5,
                 adaptive_memory_use: bool = True,
                 adaptive_memory_threshold: float = 0.7,
                 **kwargs):
        """Initialize the hierarchical agent.
        
        Args:
            policy_network: Policy network (MemoryAugmentedNetwork)
            observation_space: Observation space
            action_space: Action space
            device: Computation device
            memory_size: Maximum number of memories to store
            memory_use_prob: Probability of using memory during inference
            use_visual_network: Whether to use the visual understanding network
            use_world_model: Whether to use the world model
            use_error_detection: Whether to use the error detection network
            feature_dim: Dimension of visual features
            latent_dim: Dimension of latent space in world model
            prediction_horizon: Number of timesteps to predict in world model
            adaptive_memory_use: Whether to adapt memory usage based on errors
            adaptive_memory_threshold: Threshold for adaptive memory usage
            **kwargs: Additional arguments for MemoryAugmentedAgent
        """
        # Initialize the base memory-augmented agent
        # Pass only the parameters that the base class expects
        super().__init__(
            policy_network, 
            observation_space, 
            action_space, 
            device, 
            memory_size, 
            memory_use_prob,
            **kwargs
        )
        
        # Record configuration
        self.use_visual_network = use_visual_network
        self.use_world_model = use_world_model
        self.use_error_detection = use_error_detection
        
        # Store adaptive memory parameters as instance variables
        self.adaptive_memory_use = adaptive_memory_use
        self.adaptive_memory_threshold = adaptive_memory_threshold
        
        # Get device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up input/output dimensions
        if isinstance(observation_space, tuple):
            self.input_shape = observation_space
        else:
            self.input_shape = observation_space.shape
            
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
        else:
            self.action_dim = action_space
        
        # Initialize component networks
        
        # 1. Visual Understanding Network (for scene parsing from raw pixels)
        if use_visual_network:
            logger.critical("Initializing Visual Understanding Network")
            self.visual_network = VisualUnderstandingNetwork(
                input_shape=self.input_shape,
                feature_dim=feature_dim,
                device=self.device
            )
        else:
            self.visual_network = None
        
        # 2. Determine state dimension for world model
        # If we use visual network, it's the feature dimension
        # Otherwise, it's the flattened observation space
        if use_visual_network:
            state_dim = feature_dim
        else:
            state_dim = np.prod(self.input_shape)
            
        # 3. World Model (for predicting future states)
        if use_world_model:
            logger.critical("Initializing World Model")
            self.world_model = WorldModel(
                state_dim=state_dim,
                action_dim=self.action_dim,
                latent_dim=latent_dim,
                prediction_horizon=prediction_horizon,
                device=self.device
            )
        else:
            self.world_model = None
            
        # 4. Error Detection Network (for identifying problems)
        if use_error_detection:
            logger.critical("Initializing Error Detection Network")
            self.error_network = ErrorDetectionNetwork(
                state_dim=state_dim,
                action_dim=self.action_dim,
                use_world_model=use_world_model,
                device=self.device
            )
        else:
            self.error_network = None
            
        # Experience and prediction history
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.last_prediction = None
        self.last_uncertainty = None
        self.prediction_errors = deque(maxlen=100)
        
        # Performance tracking
        self.error_history = deque(maxlen=100)
        self.correction_history = deque(maxlen=100)
        
        # Initialize new instance variables for storing action info
        self.last_action_probs = None
        self.last_state_embedding = None
        self.last_used_memory = False
        self.last_error_detected = False
        self.last_error_info = None
        self.last_predicted_next_state = None
        
        logger.critical(f"Initialized hierarchical agent with components: "
                       f"Visual={use_visual_network}, World={use_world_model}, "
                       f"Error={use_error_detection}")
    
    def preprocess_observation(self, observation):
        """Preprocess the raw observation through specialized networks.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Processed observation (features)
        """
        # Ensure observation is a tensor
        if not isinstance(observation, torch.Tensor):
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device)
        
        # Handle frame stacking - if we have 12 channels (4 stacked frames of 3 channels each)
        # Use only the most recent frame for visual processing
        if self.use_visual_network and len(observation.shape) > 1 and observation.shape[0] == 12:
            # Extract the last frame (last 3 channels) from the stacked frames
            last_frame = observation[9:12]  # Get channels 9, 10, 11 (last frame in stack)
            
            with torch.no_grad():
                visual_features, _ = self.visual_network(last_frame)
                return visual_features
        # Apply Visual Understanding Network if enabled
        elif self.use_visual_network:
            with torch.no_grad():
                visual_features, _ = self.visual_network(observation)
                return visual_features
        else:
            # Otherwise, flatten the observation
            if len(observation.shape) > 1:
                return observation.reshape(-1)
            return observation
    
    def select_action(self, state, deterministic=False):
        """Select an action based on the current state with hierarchical processing.
        
        Args:
            state: Current state (raw observation)
            deterministic: Whether to select the action deterministically
            
        Returns:
            tuple: (action, log_prob, value) for compatibility with the trainer
        """
        logger.debug(f"Selecting action for state with shape {state.shape if hasattr(state, 'shape') else 'unknown'}")
        
        try:
            # 1. Process through Visual Understanding Network if enabled
            processed_state = self.preprocess_observation(state)
            
            # 2. Check for errors in the previous action if available
            error_detected = False
            error_info = None
            
            if self.use_error_detection and self.last_state is not None and self.last_action is not None:
                with torch.no_grad():
                    error_result = self.error_network.detect_errors(
                        self.last_state,
                        self.last_action,
                        processed_state,
                        self.last_prediction,
                        self.last_uncertainty
                    )
                    
                    # Check if any serious error was detected
                    error_severity = error_result['error_severity'].item()
                    error_detected = error_severity > self.adaptive_memory_threshold
                    
                    if error_detected:
                        error_types = error_result['error_types']
                        explanations = self.error_network.get_error_explanation(error_types)
                        error_info = {
                            'severity': error_severity,
                            'types': error_types.cpu().numpy(),
                            'explanations': explanations
                        }
                        logger.info(f"Error detected: {error_info}")
                        
                        # Adjust memory usage based on error detection
                        if self.adaptive_memory_use:
                            # Increase memory usage when errors are detected
                            self.memory_use_prob = min(1.0, self.memory_use_prob + 0.1)
                            logger.debug(f"Increased memory usage to {self.memory_use_prob}")
            
            # 3. Pass through the memory-augmented policy network
            # Ensure state is a tensor
            if not isinstance(processed_state, torch.Tensor):
                processed_state = torch.tensor(processed_state, device=self.device).float()
                
            if len(processed_state.shape) == 1:
                processed_state = processed_state.unsqueeze(0)
                
            # Decide whether to use memory for this decision
            use_memory = self.memory_enabled and (np.random.random() < self.memory_use_prob or error_detected)
            
            # Forward pass through policy network with memory
            with torch.no_grad():
                action_probs, value, next_hidden = self.policy(
                    processed_state, 
                    self.hidden_state, 
                    use_memory=use_memory
                )
                
                # Update hidden state for next time
                self.hidden_state = next_hidden
                
                # Apply action smoothing from parent class (if using)
                if len(self.action_history) > 0 and not deterministic:
                    action_probs = self._apply_action_smoothing(action_probs)
                
                # Create categorical distribution
                action_distribution = torch.distributions.Categorical(action_probs)
                
                # Choose action
                if deterministic:
                    action = torch.argmax(action_probs, dim=1)
                else:
                    action = action_distribution.sample()
                
                # Store action in history for future smoothing
                action_item = action.cpu().item()
                self._update_action_history(action_item)
                
                # Get log probability
                log_prob = action_distribution.log_prob(action)
                
                # Extract state embedding for potential memory storage
                state_embedding = self.policy.extract_state_embedding(processed_state, self.hidden_state)
                
                # Store info for experience processing
                self.last_state_embedding = state_embedding
            
            # 4. Use world model to predict the next state if enabled
            predicted_next_state = None
            uncertainty = None
            
            if self.use_world_model:
                with torch.no_grad():
                    predicted_next_state, uncertainty = self.world_model(processed_state, action)
                    
                    # Store for future error detection
                    self.last_prediction = predicted_next_state
                    self.last_uncertainty = uncertainty
            
            # Store current state and action for future error detection
            self.last_state = processed_state
            self.last_action = action
            
            # Store additional info as instance variables for process_experience
            self.last_action_probs = action_probs
            self.last_state_embedding = state_embedding
            self.last_used_memory = use_memory
            self.last_error_detected = error_detected
            self.last_error_info = error_info
            self.last_predicted_next_state = predicted_next_state
            self.last_uncertainty = uncertainty
            
            logger.debug(f"Returning action: {action.cpu().numpy().item()}")
            # Return the triple expected by the trainer
            return action.cpu().item(), log_prob.cpu().item(), value.cpu().item()
                
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a random action as fallback - using the format expected by the trainer
            random_action = torch.randint(0, self.action_dim, (1,), device=self.device).item()
            return random_action, 0.0, 0.0
    
    def process_experience(self, state, action, reward, next_state, done, info=None):
        """Process experience to potentially store in episodic memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            bool: Whether the experience was stored in memory
        """
        try:
            # Preprocess states
            processed_state = self.preprocess_observation(state)
            processed_next_state = self.preprocess_observation(next_state)
            
            # Update world model prediction error if applicable
            if self.use_world_model and self.last_prediction is not None:
                with torch.no_grad():
                    prediction_error = F.mse_loss(self.last_prediction, processed_next_state).item()
                    self.prediction_errors.append(prediction_error)
                    logger.debug(f"World model prediction error: {prediction_error:.4f}")
            
            # Store this experience's reward
            self.last_reward = reward
            
            # Get state embedding if not already available - use stored embedding if we have it
            state_embedding = self.last_state_embedding
            if state_embedding is None:
                if isinstance(processed_state, torch.Tensor):
                    state_tensor = processed_state
                else:
                    state_tensor = torch.tensor(processed_state, device=self.device).float()
                    
                state_embedding = self.policy.extract_state_embedding(state_tensor, self.hidden_state)
            
            # Calculate importance for memory storage
            importance = 0.5  # Default importance
            
            # Adjust importance based on:
            # 1. World model prediction error
            if self.use_world_model and len(self.prediction_errors) > 0:
                avg_error = sum(self.prediction_errors) / len(self.prediction_errors)
                current_error = self.prediction_errors[-1] if self.prediction_errors else 0
                # Higher importance if error is above average
                if current_error > avg_error:
                    importance += min(0.3, current_error / (avg_error + 1e-5) * 0.2)
            
            # 2. Error detection results
            if self.use_error_detection and self.last_error_detected:
                # Higher importance for states with errors
                if self.last_error_info and 'severity' in self.last_error_info:
                    severity = self.last_error_info['severity']
                    importance += min(0.3, severity)
            
            # 3. Reward signal
            if abs(reward) > 0.1:
                importance += min(0.2, abs(reward) * 0.1)
                
            # 4. Episode completion
            if done:
                importance += 0.2
            
            # Determine if we should store based on calculated importance
            should_store = importance > 0.6 or np.random.random() < importance * 0.7
                
            if should_store:
                # Prepare memory data with hierarchical components
                memory_data = {
                    'raw_state': state,
                    'processed_state': processed_state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'processed_next_state': processed_next_state,
                    'done': done,
                    'info': info,
                    'prediction_error': self.prediction_errors[-1] if self.prediction_errors else None,
                    'error_detected': self.last_error_detected
                }
                
                # Store in memory
                success = self.policy.store_memory(state_embedding, memory_data, importance)
                
                if success:
                    self.memory_stats["writes"] += 1
                    if importance > 0.7:
                        self.memory_stats["important_experiences"] += 1
                    
                    logger.info(f"Stored experience with importance {importance:.2f}")
                    return True
            
            # Add to short-term buffer regardless
            self.experience_buffer.append((state_embedding, {
                'raw_state': state,
                'processed_state': processed_state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'processed_next_state': processed_next_state,
                'done': done,
                'info': info
            }))
            
            return False
                
        except Exception as e:
            logger.error(f"Error processing experience: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_hierarchical_stats(self):
        """Get statistics for all hierarchical components.
        
        Returns:
            Dict of statistics
        """
        stats = {
            'memory_stats': self.get_memory_stats(),
            'prediction_error': sum(self.prediction_errors) / max(1, len(self.prediction_errors)) if self.prediction_errors else 0,
            'error_rate': sum(self.error_history) / max(1, len(self.error_history)) if self.error_history else 0,
            'correction_rate': sum(self.correction_history) / max(1, len(self.correction_history)) if self.correction_history else 0,
            'memory_use_prob': self.memory_use_prob
        }
        
        return stats
    
    def train_visual_network(self, observations, labels=None):
        """Train the visual understanding network.
        
        Args:
            observations: Batch of raw observations
            labels: Optional scene type labels for supervised training
            
        Returns:
            Dict of losses
        """
        if not self.use_visual_network:
            return {'loss': 0.0}
            
        # If no labels, use self-supervised learning approach
        # This is a placeholder for a proper self-supervised learning implementation
        # In practice, you would implement methods like contrastive learning, etc.
        
        visual_loss = torch.tensor(0.0).to(self.device)
        
        # Return losses
        return {'visual_loss': visual_loss.item()}
    
    def train_world_model(self, states, actions, next_states):
        """Train the world model.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            next_states: Batch of next states
            
        Returns:
            Dict of losses
        """
        if not self.use_world_model:
            return {'loss': 0.0}
            
        # Compute world model losses
        losses = self.world_model.compute_loss(states, actions, next_states)
        
        # Return losses
        return {k: v.item() for k, v in losses.items()}
    
    def train_error_network(self, states, actions, next_states, predicted_next_states=None):
        """Train the error detection network.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            next_states: Batch of next states
            predicted_next_states: Batch of predicted next states
            
        Returns:
            Dict of losses
        """
        if not self.use_error_detection:
            return {'loss': 0.0}
            
        # If predicted states not provided, generate them with world model
        if predicted_next_states is None and self.use_world_model:
            with torch.no_grad():
                predicted_next_states, _ = self.world_model(states, actions)
        
        # Compute error detection losses
        losses = self.error_network.compute_loss(
            states, actions, next_states, predicted_next_states
        )
        
        # Return losses
        return {k: v.item() for k, v in losses.items()}
    
    def reset(self):
        """Reset the agent's state."""
        super().reset()
        
        # Reset prediction histories
        self.last_state = None
        self.last_action = None
        self.last_prediction = None
        self.last_uncertainty = None
        self.prediction_errors.clear()
        
        # Reset performance tracking
        self.error_history.clear()
        self.correction_history.clear() 