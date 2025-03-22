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
import math

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
                action_dim: int = 137):
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
            # Visual feature extractor
            self.feature_extractor = VisualFeatureExtractor()
            
            # Visual change analyzer
            self.visual_change_analyzer = VisualChangeAnalyzer()
            
            # Reward normalization and scaling
            self.reward_normalizer = RewardNormalizer(
                window_size=1000,
                clip_range=10.0,
                epsilon=1e-8,
                update_freq=100
            )
            self.reward_scaler = RewardScaler(
                target_min=-1.0,
                target_max=1.0,
                window_size=1000
            )
            self.reward_shaper = RewardShaper()
            
            # State density estimator
            self.density_estimator = DensityEstimator(
                feature_dim=self.feature_dim,
                history_size=1000,
                device=self.device
            )
            
            # State novelty memory
            self.association_memory = TemporalAssociationMemory(
                feature_dim=self.feature_dim,
                history_size=1000,
                device=self.device
            )
            
            # World model for prediction
            self.world_model = WorldModelCNN(
                action_dim=self.action_dim, 
                device=self.device
            )
            self.world_model.to(self.device)
            
            # Experience buffer
            self.experience_buffer = ExperienceBuffer(capacity=10000)
            
            # World model trainer
            self.world_model_trainer = WorldModelTrainer(
                world_model=self.world_model,
                experience_buffer=self.experience_buffer,
                device=self.device
            )
            
            # Action outcome tracker
            self.action_tracker = ActionOutcomeTracker(
                action_dim=self.action_dim,  # Now this is already 137
                memory_size=1000
            )
            
            # Component weights
            self.reward_weights = {
                'prediction_error': 0.25,
                'visual_change': 0.35,
                'density': 0.20,
                'association': 0.20
            }
            
            # Training parameters
            self.training_interval = 100  # Steps between world model updates
            self.steps_since_reset = 0
            self.last_world_model_update = 0
            
            # Error normalization factor for prediction error scaling
            self.error_normalization_factor = 1.0
            
            # Background training thread
            self.bg_training_thread = None
            self.stop_bg_training = threading.Event()
            
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
            
    def compute_reward(self, current_frame: torch.Tensor, next_frame: torch.Tensor,
                       action: int, done: bool = False, info: dict = None) -> float:
        """
        Compute a reward based on frame transitions.
        
        Args:
            current_frame: Current state frame
            next_frame: Next state frame
            action: Action taken
            done: Whether the episode is done (default: False)
            info: Additional information (default: None)
            
        Returns:
            float: Computed reward
        """
        # Validate input frames
        if current_frame is None or next_frame is None:
            logger.warning("Missing input frames for reward computation")
            return 0.0
            
        # Handle case where info is None
        if info is None:
            info = {}
            
        # Check for NaN/Inf values in input frames
        if torch.is_tensor(current_frame) and (torch.isnan(current_frame).any() or torch.isinf(current_frame).any()):
            logger.warning("NaN/Inf values detected in current_frame, replacing with zeros")
            current_frame = torch.zeros_like(current_frame)
            
        if torch.is_tensor(next_frame) and (torch.isnan(next_frame).any() or torch.isinf(next_frame).any()):
            logger.warning("NaN/Inf values detected in next_frame, replacing with zeros")
            next_frame = torch.zeros_like(next_frame)
        
        # For autonomous rewards, multiple components can be used
        components = {}
        
        try:
            # Component 1: Temporal difference (visual change reward)
            if self.reward_weights['visual_change'] > 0:
                # Check if frames are usable for difference calculation
                if torch.is_tensor(current_frame) and torch.is_tensor(next_frame) and current_frame.shape == next_frame.shape:
                    frame_diff = torch.abs(next_frame - current_frame).mean().item()
                    # Normalize and scale with smoothing constant
                    visual_change = min(frame_diff * 10.0, 1.0)
                    components['visual_change'] = visual_change
                else:
                    logger.warning("Frames not compatible for difference calculation")
                    components['visual_change'] = 0.0
            
            # Component 2: Prediction error (if model is available)
            if self.reward_weights['prediction_error'] > 0 and hasattr(self, 'prediction_model') and self.prediction_model is not None:
                try:
                    # Predict next frame from current frame and action
                    predicted_next_frame = self.prediction_model(current_frame, action)
                    
                    # Compute prediction error
                    prediction_error = torch.abs(predicted_next_frame - next_frame).mean().item()
                    
                    # Normalize prediction error
                    prediction_error = min(prediction_error * 5.0, 1.0)
                    components['prediction_error'] = prediction_error
                except Exception as e:
                    logger.error(f"Error computing prediction error: {e}")
                    components['prediction_error'] = 0.0
            
            # Component 3: Exploration bonus (visit novel states)
            if self.reward_weights['density'] > 0:
                # Check if we can compute state hash
                if torch.is_tensor(next_frame):
                    # Compute state fingerprint (low-dimensional hash)
                    state_hash = self._compute_state_hash(next_frame)
                    
                    # Check if state is novel
                    if state_hash not in self.visited_states:
                        # High reward for novel states
                        self.visited_states.add(state_hash)
                        exploration_bonus = 1.0
                    else:
                        # No reward for revisited states
                        exploration_bonus = 0.0
                    
                    components['density'] = exploration_bonus
                else:
                    logger.warning("Next frame not suitable for state hash calculation")
                    components['density'] = 0.0
            
            # Component 4: Action variety reward
            if self.reward_weights['association'] > 0:
                # Update action history
                self.action_history.append(action)
                if len(self.action_history) > self.action_history_size:
                    self.action_history.pop(0)
                
                # Compute variety in recent actions
                unique_actions = len(set(self.action_history))
                variety_ratio = unique_actions / min(len(self.action_history), self.num_actions)
                
                # Scale variety ratio
                action_variety = variety_ratio
                components['association'] = action_variety
            
            # Combine components with their weights
            reward = 0.0
            for component, value in components.items():
                # Check for NaN/Inf in component values
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    logger.warning(f"NaN/Inf detected in reward component '{component}', using zero")
                    value = 0.0
                    
                reward += self.reward_weights[component] * value
            
            # Check if final reward is NaN/Inf
            if math.isnan(reward) or math.isinf(reward):
                logger.warning("NaN/Inf in final computed reward, using zero")
                reward = 0.0
                
            # Add to reward history and update cumulative reward
            self.reward_history.append(reward)
            self.cumulative_reward += reward
            
            # Log metrics periodically
            self.reward_counter += 1
            if self.reward_counter % 100 == 0:
                avg_reward = sum(self.reward_history[-100:]) / min(100, len(self.reward_history))
                logger.info(f"Average reward (last 100): {avg_reward:.4f}, Cumulative: {self.cumulative_reward:.4f}")
                logger.debug(f"Reward components: {components}")
            
            # Clip reward to reasonable range
            reward = max(min(reward, self.max_reward), self.min_reward)
            
            return reward
            
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            logger.error("Using default reward of 0.0")
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
            # Get device from input tensors
            device = next_state.device if isinstance(next_state, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Ensure we have the right shapes for prediction
            if isinstance(current_state, torch.Tensor) and current_state.dim() < 2:
                current_state = current_state.unsqueeze(0)  # Add batch dimension
                
            if isinstance(action_tensor, torch.Tensor) and action_tensor.dim() < 2:
                action_tensor = action_tensor.unsqueeze(0)  # Add batch dimension
                
            if isinstance(next_state, torch.Tensor) and next_state.dim() < 2:
                next_state = next_state.unsqueeze(0)  # Add batch dimension
                
            # Ensure all tensors are on the same device
            current_state = current_state.to(device)
            action_tensor = action_tensor.to(device)
            next_state = next_state.to(device)
            
            # Get world model's prediction
            with torch.no_grad():
                # First encode the current state (if it's not already an embedding)
                if current_state.dim() > 2:  # If it's an image tensor (batch, channels, height, width)
                    current_embedding = self.world_model.encode(current_state)
                else:
                    current_embedding = current_state
                
                # Predict the next state embedding
                next_embedding = self.world_model.predict_next(current_embedding, action_tensor)
                
                # Ensure next_embedding is on the correct device
                next_embedding = next_embedding.to(device)
                
                # Decode the embedding to get the predicted next frame
                if next_state.dim() > 2:  # If expected output is an image
                    prediction = self.world_model.decode(next_embedding)
                    # Ensure prediction is on the correct device
                    prediction = prediction.to(device)
                else:  # If expected output is an embedding
                    prediction = next_embedding
            
            # If prediction and next_state still have incompatible shapes, log and handle
            if prediction.shape != next_state.shape:
                logger.error(f"Incompatible tensor dimensions: prediction {prediction.shape} vs next_state {next_state.shape}")
                
                # If the dimensions are completely different, we may need different comparison approach
                if prediction.dim() != next_state.dim():
                    # If next_state is an image but prediction is an embedding, 
                    # we should compare embeddings instead
                    if next_state.dim() > 2 and prediction.dim() == 2:
                        next_embedding = self.world_model.encode(next_state).to(device)
                        error = F.mse_loss(next_embedding, next_embedding).item()
                        reward = 1.0 - min(error / self.error_normalization_factor, 1.0)
                        return reward
                    else:
                        return 0.0  # Can't compare these formats
            
            # Normal case - compute prediction error (MSE)
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