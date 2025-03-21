"""
Autonomous reward system for Cities: Skylines 2 environment.

This module provides a reward system that learns to provide rewards
without explicit game metrics, based on visual changes and predictions.
"""

import numpy as np
import torch
import torch.nn.functional as F
import logging
import time
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import threading

from src.environment.rewards.metrics import DensityEstimator, TemporalAssociationMemory
from src.environment.rewards.analyzers import VisualChangeAnalyzer, VisualFeatureExtractor
from src.environment.rewards.calibration import RewardNormalizer, RewardScaler, RewardShaper
from src.environment.rewards.world_model import WorldModelCNN, ExperienceBuffer, WorldModelTrainer

# Configure logger
logger = logging.getLogger(__name__)

class ActionOutcomeTracker:
    """Tracks outcomes of different actions to guide exploration."""
    
    def __init__(self, action_dim: int = 12, memory_size: int = 1000, decay_rate: float = 0.95):
        """Initialize action outcome tracker.
        
        Args:
            action_dim: Number of action dimensions
            memory_size: Size of outcome memory per action
            decay_rate: Rate at which old values decay
        """
        self.action_dim = action_dim
        self.memory_size = memory_size
        self.decay_rate = decay_rate
        
        # Outcome tracking per action
        self.action_outcomes = {}
        self.action_counts = np.zeros(action_dim)
        self.success_counts = np.zeros(action_dim)
        
    def update_action_tracking(self, action_idx: int, outcome: float) -> None:
        """Update tracking for an action based on outcome.
        
        Args:
            action_idx: Index of action taken
            outcome: Outcome/reward value for action
        """
        # Initialize if first time seeing this action
        if action_idx not in self.action_outcomes:
            self.action_outcomes[action_idx] = deque(maxlen=self.memory_size)
            
        # Store outcome
        self.action_outcomes[action_idx].append(outcome)
        
        # Update counts
        self.action_counts[action_idx] += 1
        if outcome > 0:
            self.success_counts[action_idx] += 1
            
    def get_action_preference(self, action_idx: int) -> float:
        """Get preference score for an action.
        
        Args:
            action_idx: Index of action
            
        Returns:
            float: Preference score (higher = more preferred)
        """
        # If no data yet, return neutral preference
        if action_idx not in self.action_outcomes or len(self.action_outcomes[action_idx]) == 0:
            return 0.5
            
        # Calculate average outcome
        avg_outcome = np.mean(self.action_outcomes[action_idx])
        
        # Calculate success rate
        if self.action_counts[action_idx] > 0:
            success_rate = self.success_counts[action_idx] / self.action_counts[action_idx]
        else:
            success_rate = 0.0
            
        # Calculate recency-weighted average
        recency_weighted = 0.0
        total_weight = 0.0
        
        for i, outcome in enumerate(reversed(self.action_outcomes[action_idx])):
            weight = self.decay_rate ** i
            recency_weighted += outcome * weight
            total_weight += weight
            
        if total_weight > 0:
            recency_weighted /= total_weight
            
        # Combine metrics into preference score
        preference = 0.4 * (avg_outcome + 0.5) + 0.3 * success_rate + 0.3 * (recency_weighted + 0.5)
        
        # Ensure in [0, 1] range
        preference = max(0.0, min(1.0, preference))
        
        return preference
        
    def get_all_preferences(self) -> np.ndarray:
        """Get preference scores for all actions.
        
        Returns:
            np.ndarray: Array of preference scores
        """
        preferences = np.zeros(self.action_dim)
        
        for i in range(self.action_dim):
            preferences[i] = self.get_action_preference(i)
            
        return preferences
        
    def get_exploration_bonus(self, action_idx: int) -> float:
        """Get exploration bonus for an action based on visit count.
        
        Args:
            action_idx: Index of action
            
        Returns:
            float: Exploration bonus (higher for less visited actions)
        """
        # If no action has been taken yet, provide uniform bonus
        if np.sum(self.action_counts) == 0:
            return 1.0
            
        # Compute visit count ratio
        total_count = np.sum(self.action_counts)
        action_ratio = self.action_counts[action_idx] / total_count
        
        # Convert to bonus (lower ratio = higher bonus)
        bonus = 1.0 - action_ratio
        
        # Scale bonus
        scaled_bonus = min(1.0, bonus * self.action_dim)
        
        return scaled_bonus
        
    def save_state(self, path: str) -> None:
        """Save tracker state to disk.
        
        Args:
            path: Path to save state
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare state for saving
            state = {
                'action_dim': self.action_dim,
                'memory_size': self.memory_size,
                'decay_rate': self.decay_rate,
                'action_outcomes': {k: list(v) for k, v in self.action_outcomes.items()},
                'action_counts': self.action_counts.tolist(),
                'success_counts': self.success_counts.tolist()
            }
            
            # Save using json
            with open(path, 'w') as f:
                json.dump(state, f)
                
            logger.info(f"Saved action outcome tracker state")
        except Exception as e:
            logger.error(f"Error saving action outcome tracker: {e}")
            
    def load_state(self, path: str) -> None:
        """Load tracker state from disk.
        
        Args:
            path: Path to load state from
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Action outcome tracker state file not found: {path}")
                return
                
            # Load from json
            with open(path, 'r') as f:
                state = json.load(f)
                
            # Restore state
            self.action_dim = state['action_dim']
            self.memory_size = state['memory_size']
            self.decay_rate = state['decay_rate']
            
            # Restore outcomes (converting to deque)
            self.action_outcomes = {}
            for k, v in state['action_outcomes'].items():
                self.action_outcomes[int(k)] = deque(v, maxlen=self.memory_size)
                
            # Restore counts
            self.action_counts = np.array(state['action_counts'])
            self.success_counts = np.array(state['success_counts'])
            
            logger.info(f"Loaded action outcome tracker state")
        except Exception as e:
            logger.error(f"Error loading action outcome tracker: {e}")


class AutonomousRewardSystem:
    """Learns to provide rewards without explicit game metrics."""
    
    def __init__(self, 
                config: Optional[Dict[str, Any]] = None,
                hardware_accelerated: bool = True,
                checkpoints_dir: str = "checkpoints/reward_system",
                feature_dim: int = 512,
                action_dim: int = 12):
        """Initialize autonomous reward system.
        
        Args:
            config: Configuration dict
            hardware_accelerated: Use GPU if available
            checkpoints_dir: Directory for saving/loading checkpoints
            feature_dim: Dimension of feature vectors
            action_dim: Dimension of action space
        """
        self.config = config or {}
        self.checkpoints_dir = checkpoints_dir
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # Ensure checkpoints directory exists
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Initialize device
        if hardware_accelerated and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("Using CUDA for reward system")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for reward system")
            
        # Initialize components
        self._init_components()
        
        # Tracking variables
        self.reward_history = deque(maxlen=1000)
        self.cumulative_reward = 0.0
        self.last_world_model_update = 0
        self.last_frame = None
        self.last_action = None
        self.last_state_embedding = None
        self.last_visual_features = None
        self.last_error = 0.0
        self.steps_since_reset = 0
        
        # State variables
        self.is_stable = False
        self.stability_threshold = self.config.get("stability_threshold", 100)
        self.training_interval = self.config.get("world_model_training_interval", 200)
        self.world_model_batch_size = self.config.get("world_model_batch_size", 64)
        self.world_model_steps = self.config.get("world_model_steps", 10)
        
        # Reward component weights
        self.reward_weights = {
            'prediction_error': self.config.get("prediction_error_weight", 0.3),
            'visual_change': self.config.get("visual_change_weight", 0.3),
            'density': self.config.get("density_weight", 0.2),
            'association': self.config.get("association_weight", 0.2)
        }
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.reward_weights.values())
        if weight_sum > 0:
            self.reward_weights = {k: v / weight_sum for k, v in self.reward_weights.items()}
            
        # Background training thread
        self.training_thread = None
        self.stop_training = False
        
        logger.info(f"Initialized autonomous reward system with weights: {self.reward_weights}")
        
    def _init_components(self) -> None:
        """Initialize reward system components."""
        try:
            # Create feature extractor
            self.feature_extractor = VisualFeatureExtractor(
                config={
                    'edge_threshold': 100,
                    'histogram_bins': 32
                }
            )
            
            # Create visual change analyzer
            self.visual_change_analyzer = VisualChangeAnalyzer(
                config={
                    'min_change_threshold': 0.05,
                    'max_change_threshold': 0.8,
                    'edge_threshold': 100,
                    'histogram_bins': 32
                }
            )
            
            # Create density estimator
            self.density_estimator = DensityEstimator(
                feature_dim=self.feature_dim,
                history_size=1000,
                device=self.device
            )
            
            # Create association memory
            self.association_memory = TemporalAssociationMemory(
                feature_dim=self.feature_dim,
                history_size=1000,
                device=self.device
            )
            
            # Create world model
            self.world_model = WorldModelCNN(
                input_channels=3,
                action_dim=self.action_dim,
                embedding_dim=self.feature_dim,
                device=self.device
            )
            
            # Create experience buffer
            self.experience_buffer = ExperienceBuffer(
                capacity=10000
            )
            
            # Create world model trainer
            self.world_model_trainer = WorldModelTrainer(
                world_model=self.world_model,
                experience_buffer=self.experience_buffer,
                device=self.device
            )
            
            # Create reward calibration
            self.reward_normalizer = RewardNormalizer(
                window_size=1000,
                update_freq=50
            )
            self.reward_scaler = RewardScaler(
                target_min=-1.0,
                target_max=1.0
            )
            
            # Create action tracker
            self.action_tracker = ActionOutcomeTracker(
                action_dim=self.action_dim,
                memory_size=1000
            )
            
            logger.info("Reward system components initialized")
        except Exception as e:
            logger.error(f"Error initializing reward system components: {e}")
            raise
        
    def save_state(self, save_dir: Optional[str] = None) -> None:
        """Save reward system state to disk.
        
        Args:
            save_dir: Directory to save state (uses checkpoints_dir if None)
        """
        try:
            save_dir = save_dir or self.checkpoints_dir
            os.makedirs(save_dir, exist_ok=True)
            
            # Save world model components
            self.world_model_trainer.save_all(
                model_path=os.path.join(save_dir, "world_model.pt"),
                buffer_path=os.path.join(save_dir, "experience_buffer.pkl"),
                trainer_path=os.path.join(save_dir, "world_model_trainer.pt")
            )
            
            # Save density estimator
            self.density_estimator.save_state(
                path=os.path.join(save_dir, "density_estimator.pkl")
            )
            
            # Save association memory
            self.association_memory.save_state(
                path=os.path.join(save_dir, "association_memory.pkl")
            )
            
            # Save visual change analyzer
            self.visual_change_analyzer.save_state(
                path=os.path.join(save_dir, "visual_change_analyzer.pkl")
            )
            
            # Save action tracker
            self.action_tracker.save_state(
                path=os.path.join(save_dir, "action_tracker.json")
            )
            
            # Save metrics and state
            state = {
                'reward_weights': self.reward_weights,
                'cumulative_reward': self.cumulative_reward,
                'steps_since_reset': self.steps_since_reset,
                'is_stable': self.is_stable,
                'last_error': self.last_error,
                'reward_history': list(self.reward_history),
                'config': self.config
            }
            
            with open(os.path.join(save_dir, "reward_system_state.json"), 'w') as f:
                json.dump(state, f, indent=2)
                
            logger.info(f"Saved reward system state to {save_dir}")
        except Exception as e:
            logger.error(f"Error saving reward system state: {e}")
            
    def load_state(self, load_dir: Optional[str] = None) -> None:
        """Load reward system state from disk.
        
        Args:
            load_dir: Directory to load state from (uses checkpoints_dir if None)
        """
        try:
            load_dir = load_dir or self.checkpoints_dir
            
            if not os.path.exists(load_dir):
                logger.warning(f"Reward system checkpoint directory not found: {load_dir}")
                return
                
            # Load world model components
            self.world_model_trainer.load_all(
                model_path=os.path.join(load_dir, "world_model.pt"),
                buffer_path=os.path.join(load_dir, "experience_buffer.pkl"),
                trainer_path=os.path.join(load_dir, "world_model_trainer.pt")
            )
            
            # Load density estimator
            self.density_estimator.load_state(
                path=os.path.join(load_dir, "density_estimator.pkl")
            )
            
            # Load association memory
            self.association_memory.load_state(
                path=os.path.join(load_dir, "association_memory.pkl")
            )
            
            # Load visual change analyzer
            self.visual_change_analyzer.load_state(
                path=os.path.join(load_dir, "visual_change_analyzer.pkl")
            )
            
            # Load action tracker
            self.action_tracker.load_state(
                path=os.path.join(load_dir, "action_tracker.json")
            )
            
            # Load metrics and state
            if os.path.exists(os.path.join(load_dir, "reward_system_state.json")):
                with open(os.path.join(load_dir, "reward_system_state.json"), 'r') as f:
                    state = json.load(f)
                    
                self.reward_weights = state.get('reward_weights', self.reward_weights)
                self.cumulative_reward = state.get('cumulative_reward', 0.0)
                self.steps_since_reset = state.get('steps_since_reset', 0)
                self.is_stable = state.get('is_stable', False)
                self.last_error = state.get('last_error', 0.0)
                
                if 'reward_history' in state:
                    self.reward_history = deque(state['reward_history'], maxlen=1000)
                    
                if 'config' in state:
                    # Only update config values that exist in current config
                    for k, v in state['config'].items():
                        if k in self.config:
                            self.config[k] = v
                            
            logger.info(f"Loaded reward system state from {load_dir}")
        except Exception as e:
            logger.error(f"Error loading reward system state: {e}")
            
    def compute_reward(self, 
                      current_frame: torch.Tensor,
                      action_idx: int,
                      next_frame: torch.Tensor) -> float:
        """Compute reward based on visual changes between frames.
        
        Args:
            current_frame: Current frame observation (PyTorch tensor)
            action_idx: Index of action taken
            next_frame: Next frame observation (PyTorch tensor)
            
        Returns:
            float: Computed reward
        """
        try:
            # Convert PyTorch tensors to numpy arrays for OpenCV operations
            if isinstance(current_frame, torch.Tensor):
                current_frame_np = current_frame.detach().cpu().numpy()
                # If tensor is in CHW format, convert to HWC for OpenCV
                if current_frame_np.shape[0] == 3:  # CHW format
                    current_frame_np = np.transpose(current_frame_np, (1, 2, 0))
            else:
                current_frame_np = current_frame
                
            if isinstance(next_frame, torch.Tensor):
                next_frame_np = next_frame.detach().cpu().numpy()
                # If tensor is in CHW format, convert to HWC for OpenCV
                if next_frame_np.shape[0] == 3:  # CHW format
                    next_frame_np = np.transpose(next_frame_np, (1, 2, 0))
            else:
                next_frame_np = next_frame
            
            # Create tensor versions for PyTorch operations
            current_frame_tensor = torch.tensor(current_frame_np, dtype=torch.float32).unsqueeze(0)
            # Convert action_idx to a tensor with correct shape
            action_tensor = torch.zeros(self.action_dim, dtype=torch.float32)
            action_tensor[action_idx] = 1.0  # One-hot encoding
            action_tensor = action_tensor.unsqueeze(0)  # Add batch dimension
            next_frame_tensor = torch.tensor(next_frame_np, dtype=torch.float32).unsqueeze(0)
            
            # Ensure frames are in correct format (B, C, H, W)
            if current_frame_tensor.dim() == 4 and current_frame_tensor.shape[1] != 3:
                # Move channel dimension to correct position if needed
                current_frame_tensor = current_frame_tensor.permute(0, 3, 1, 2)
            if next_frame_tensor.dim() == 4 and next_frame_tensor.shape[1] != 3:
                next_frame_tensor = next_frame_tensor.permute(0, 3, 1, 2)
            
            # Get state embedding
            with torch.no_grad():
                current_state_embedding = self.world_model.encode(current_frame_tensor)
                
            # Extract visual features - use numpy arrays for OpenCV compatibility
            current_features = self.feature_extractor.extract_features(current_frame_np)
            next_features = self.feature_extractor.extract_features(next_frame_np)
            
            # Compute feature differences
            feature_diff = self.feature_extractor.compute_feature_difference(
                current_features, next_features)
                
            # Compute visual change score
            visual_change_score, change_metrics = self.visual_change_analyzer.get_visual_change_score(
                current_frame_np, next_frame_np)
                
            # Add to experience buffer for world model training
            self.experience_buffer.add(
                current_frame_np, action_idx, next_frame_np, 0.0)  # Reward will be updated later
                
            # Compute prediction error component
            prediction_error = self._compute_prediction_error_reward(
                current_state_embedding, action_tensor, next_frame_tensor)
                
            # Compute visual change component
            visual_change_reward = self._compute_visual_change_reward(
                visual_change_score, change_metrics)
                
            # Compute density component
            density_reward = self._compute_density_reward(current_state_embedding)
                
            # Compute association component
            association_reward = self._compute_association_reward(
                current_state_embedding, change_metrics)
                
            # Combine reward components
            reward_components = {
                'prediction_error': prediction_error,
                'visual_change': visual_change_reward,
                'density': density_reward,
                'association': association_reward
            }
            
            # Weighted sum of components
            reward = sum(self.reward_weights[k] * v for k, v in reward_components.items() 
                        if k in self.reward_weights)
                
            # Apply normalization and scaling
            normalized_reward = self.reward_normalizer.normalize(reward)
            scaled_reward = self.reward_scaler.percentile_scale(normalized_reward)
            
            # Update tracking variables
            self.reward_history.append(scaled_reward)
            self.cumulative_reward += scaled_reward
            self.steps_since_reset += 1
            
            # Update action tracker - action_idx is already a scalar
            self.action_tracker.update_action_tracking(action_idx, scaled_reward)
            
            # Update association memory
            self.association_memory.store(current_state_embedding.cpu(), scaled_reward)
            
            # Store for next iteration
            self.last_frame = current_frame_np
            self.last_action = action_idx
            self.last_state_embedding = current_state_embedding
            self.last_visual_features = current_features
            
            # Check if it's time to update world model
            if (self.steps_since_reset - self.last_world_model_update) >= self.training_interval:
                self._update_world_model()
                
            # Check stability
            self._check_stability()
            
            # Log details periodically
            if self.steps_since_reset % 100 == 0:
                self._log_reward_metrics(reward_components, scaled_reward)
                
            return scaled_reward
            
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            return 0.0
            
    def _compute_prediction_error_reward(self, current_state, action_tensor, next_state):
        """Compute reward based on prediction error.
        
        Args:
            current_state: Current state tensor
            action_tensor: Action tensor
            next_state: Next state tensor
            
        Returns:
            float: Prediction error reward
        """
        try:
            # Ensure we have the right shapes for prediction
            if isinstance(current_state, torch.Tensor) and current_state.dim() < 2:
                current_state = current_state.unsqueeze(0)  # Add batch dimension
                
            if isinstance(action_tensor, torch.Tensor) and action_tensor.dim() < 2:
                action_tensor = action_tensor.unsqueeze(0)  # Add batch dimension
                
            if isinstance(next_state, torch.Tensor) and next_state.dim() < 2:
                next_state = next_state.unsqueeze(0)  # Add batch dimension
                
            # Get world model's prediction
            with torch.no_grad():
                prediction = self.world_model.predict(current_state, action_tensor)
                
            # If prediction and next_state have incompatible shapes, reshape them
            if prediction.shape != next_state.shape:
                # If dimensions differ completely, log error and return 0
                if prediction.dim() != next_state.dim():
                    logger.error(f"Incompatible tensor dimensions: prediction {prediction.shape} vs next_state {next_state.shape}")
                    return 0.0
                    
                # Otherwise try to align dimensions
                if prediction.shape[0] != next_state.shape[0]:
                    if prediction.shape[0] == 1:
                        prediction = prediction.expand(next_state.shape[0], *prediction.shape[1:])
                    elif next_state.shape[0] == 1:
                        next_state = next_state.expand(prediction.shape[0], *next_state.shape[1:])
                    else:
                        logger.error(f"Cannot reconcile batch dimensions: {prediction.shape[0]} vs {next_state.shape[0]}")
                        return 0.0
                
                # Adjust feature dimensions if needed by padding or truncating
                if prediction.dim() > 1 and next_state.dim() > 1:
                    min_feat_dim = min(prediction.shape[1], next_state.shape[1])
                    prediction = prediction[:, :min_feat_dim]
                    next_state = next_state[:, :min_feat_dim]
                    
            # Compute prediction error (MSE)
            prediction_error = F.mse_loss(prediction, next_state).item()
            
            # Convert to reward (higher error = lower reward)
            reward = 1.0 - min(prediction_error / self.error_normalization_factor, 1.0)
            
            return reward
        except Exception as e:
            logger.error(f"Error computing prediction error reward: {e}")
            return 0.0
            
    def _compute_visual_change_reward(self, 
                                     change_score: float,
                                     change_metrics: Dict[str, float]) -> float:
        """Compute reward component based on visual change.
        
        Args:
            change_score: Visual change score
            change_metrics: Detailed change metrics
            
        Returns:
            float: Visual change reward
        """
        try:
            # Add metrics to visual change analyzer
            if self.last_frame is not None:
                # Use stored reward as outcome for analyzer
                avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0
                self.visual_change_analyzer.update_with_outcome(change_metrics, avg_reward)
                
            # Compute reward based on change significance
            # Moderate changes are better than no change or extreme changes
            # Using a bell curve centered around 0.5
            optimal_change = 0.5
            reward = 1.0 - 2.0 * abs(change_score - optimal_change)
            
            # Ensure non-negative reward
            reward = max(0.0, reward)
            
            return reward
            
        except Exception as e:
            logger.error(f"Error computing visual change reward: {e}")
            return 0.0
            
    def _compute_density_reward(self, state_embedding: torch.Tensor) -> float:
        """Compute reward component based on state density.
        
        Args:
            state_embedding: Encoded state embedding
            
        Returns:
            float: Density reward
        """
        try:
            # Get novelty score
            novelty = self.density_estimator.compute_novelty(state_embedding)
            
            # Update density estimator
            self.density_estimator.update(state_embedding)
            
            # Novelty as reward (encourage exploration)
            reward = novelty
            
            return reward
            
        except Exception as e:
            logger.error(f"Error computing density reward: {e}")
            return 0.0
            
    def _compute_association_reward(self, 
                                   state_embedding: torch.Tensor,
                                   change_metrics: Dict[str, float]) -> float:
        """Compute reward component based on temporal associations.
        
        Args:
            state_embedding: Encoded state embedding
            change_metrics: Visual change metrics
            
        Returns:
            float: Association reward
        """
        try:
            # Query association memory for expected outcome
            expected_outcome = self.association_memory.query(state_embedding)
            
            # Compute association strength with visual changes
            if self.last_state_embedding is not None:
                # Get predicted outcome from visual change
                predicted_outcome = self.visual_change_analyzer.predict_outcome(change_metrics)
                
                # Compute agreement between predictions
                error = abs(expected_outcome - predicted_outcome)
                agreement = max(0.0, 1.0 - error)
                
                # Reward proportional to agreement
                reward = agreement
            else:
                # No previous state for comparison
                reward = 0.0
                
            return reward
            
        except Exception as e:
            logger.error(f"Error computing association reward: {e}")
            return 0.0
            
    def _update_world_model(self) -> None:
        """Update world model with collected experiences."""
        try:
            # Check if enough experiences
            if len(self.experience_buffer) < self.world_model_batch_size:
                logger.debug(f"Not enough experiences to train world model: {len(self.experience_buffer)}/{self.world_model_batch_size}")
                return
                
            # Update world model
            logger.info(f"Training world model ({len(self.experience_buffer)} experiences available)")
            metrics = self.world_model_trainer.train_batch(self.world_model_steps)
            
            # Update timestamp
            self.last_world_model_update = self.steps_since_reset
            
            logger.info(f"World model training: loss={metrics['total_loss']:.6f}")
            
        except Exception as e:
            logger.error(f"Error updating world model: {e}")
            
    def _check_stability(self) -> None:
        """Check if reward system has stabilized."""
        try:
            # Check if enough steps
            if self.steps_since_reset < self.stability_threshold:
                return
                
            # Check reward variance
            if len(self.reward_history) > self.stability_threshold // 2:
                recent_rewards = list(self.reward_history)[-self.stability_threshold//2:]
                reward_std = np.std(recent_rewards)
                
                # Stable if standard deviation is low
                stable_threshold = 0.3  # Adjust as needed
                self.is_stable = reward_std < stable_threshold
                
                if self.is_stable and self.steps_since_reset % 500 == 0:
                    logger.info(f"Reward system is stable: std={reward_std:.4f}")
                    
        except Exception as e:
            logger.error(f"Error checking stability: {e}")
            
    def _log_reward_metrics(self, 
                           components: Dict[str, float],
                           total_reward: float) -> None:
        """Log reward metrics periodically.
        
        Args:
            components: Reward components
            total_reward: Total reward
        """
        # Calculate statistics
        avg_reward = np.mean(self.reward_history) if self.reward_history else 0.0
        min_reward = min(self.reward_history) if self.reward_history else 0.0
        max_reward = max(self.reward_history) if self.reward_history else 0.0
        std_reward = np.std(self.reward_history) if len(self.reward_history) > 1 else 0.0
        
        # Log metrics
        logger.info(f"Reward metrics [step {self.steps_since_reset}]: "
                   f"total={total_reward:.4f}, avg={avg_reward:.4f}, "
                   f"min={min_reward:.4f}, max={max_reward:.4f}, std={std_reward:.4f}")
        
        # Log components
        logger.debug(f"Reward components: " + 
                    ", ".join([f"{k}={v:.4f}" for k, v in components.items()]))
                    
    def start_background_training(self, 
                                 interval_seconds: int = 60,
                                 checkpoint_interval: int = 10) -> None:
        """Start background thread for periodic training and checkpointing.
        
        Args:
            interval_seconds: Seconds between training cycles
            checkpoint_interval: Number of cycles between checkpoints
        """
        if self.training_thread is not None and self.training_thread.is_alive():
            logger.warning("Background training already running")
            return
            
        self.stop_training = False
        
        def training_loop():
            """Background training loop function."""
            cycles = 0
            while not self.stop_training:
                try:
                    # Train world model if enough experiences
                    if len(self.experience_buffer) >= self.world_model_batch_size:
                        logger.info("Running background world model training")
                        self.world_model_trainer.train_batch(self.world_model_steps * 5)
                        
                    # Save checkpoint periodically
                    cycles += 1
                    if cycles % checkpoint_interval == 0:
                        logger.info("Saving reward system checkpoint")
                        self.save_state()
                        
                    # Sleep until next cycle
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Error in background training: {e}")
                    time.sleep(interval_seconds)
                    
        # Start thread
        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()
        logger.info(f"Started background training thread (interval: {interval_seconds}s)")
        
    def stop_background_training(self) -> None:
        """Stop background training thread and save checkpoint."""
        if self.training_thread is None or not self.training_thread.is_alive():
            return
            
        logger.info("Stopping background training thread")
        self.stop_training = True
        self.training_thread.join(timeout=10)
        
        # Save final checkpoint
        self.save_state()
        logger.info("Background training stopped and checkpoint saved")
        
    def reset(self) -> None:
        """Reset reward system for a new episode."""
        # Reset tracking variables
        self.reward_history.clear()
        self.cumulative_reward = 0.0
        self.steps_since_reset = 0
        self.last_world_model_update = 0
        self.last_frame = None
        self.last_action = None
        self.last_state_embedding = None
        self.last_visual_features = None
        self.last_error = 0.0
        
        # Reset reward shaper
        self.reward_shaper.reset()
        
        logger.info("Reset reward system for new episode")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current reward system metrics.
        
        Returns:
            Dict: Current metrics
        """
        # Calculate reward statistics
        reward_stats = {
            'mean': np.mean(self.reward_history) if self.reward_history else 0.0,
            'std': np.std(self.reward_history) if len(self.reward_history) > 1 else 0.0,
            'min': min(self.reward_history) if self.reward_history else 0.0,
            'max': max(self.reward_history) if self.reward_history else 0.0,
            'count': len(self.reward_history),
            'cumulative': self.cumulative_reward
        }
        
        # Get world model metrics
        world_model_metrics = self.world_model_trainer.get_training_metrics()
        
        # Combine metrics
        metrics = {
            'reward': reward_stats,
            'world_model': world_model_metrics,
            'steps': self.steps_since_reset,
            'is_stable': self.is_stable,
            'experience_buffer_size': len(self.experience_buffer),
            'last_prediction_error': self.last_error
        }
        
        return metrics 