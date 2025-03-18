"""
Autonomous reward system for learning game objectives through raw pixel observations.
Implements intrinsic motivation mechanisms without explicit game knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
import time
from collections import deque
from src.config.hardware_config import HardwareConfig
import cv2
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)

class WorldModelCNN(nn.Module):
    """Neural network for predicting next frame and encoding observations into feature space."""
    
    def __init__(self, config: HardwareConfig):
        """Initialize the world model.
        
        Args:
            config (HardwareConfig): Hardware configuration
        """
        super(WorldModelCNN, self).__init__()
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # Encoder (shared between predictor and feature extractor)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        ).to(self.device, dtype=self.dtype)
        
        # Feature representation (for novelty detection)
        self.feature_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(59904, 512),  # Updated from 64 * 9 * 9 to match actual flattened dimension
            nn.ReLU()
        ).to(self.device, dtype=self.dtype)
        
        # World model (predicts next frame)
        self.frame_predictor = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=8, stride=4, padding=2, output_padding=0)
        ).to(self.device, dtype=self.dtype)
        
        # Action embedding
        self.action_embedder = nn.Sequential(
            nn.Linear(1, 64),  # Assumes action is a single discrete value
            nn.ReLU()
        ).to(self.device, dtype=self.dtype)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
    def encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a frame into feature representation.
        
        Args:
            frame (torch.Tensor): Current frame [C, H, W]
            
        Returns:
            torch.Tensor: Feature representation
        """
        x = frame.to(self.device, dtype=self.dtype)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        features = self.encoder(x)
        return self.feature_layer(features)
    
    def predict_next_frame(self, frames: List[torch.Tensor], action: int) -> torch.Tensor:
        """Predict the next frame based on previous frames and action.
        
        Args:
            frames (List[torch.Tensor]): List of previous frames [C, H, W]
            action (int): Action taken
            
        Returns:
            torch.Tensor: Predicted next frame
        """
        # Check if we have frames
        if not frames:
            # Return a zero tensor with default size if no frames available
            return torch.zeros(3, 240, 320, device=self.device, dtype=self.dtype)
            
        # Use the most recent frame
        frame = frames[-1].to(self.device, dtype=self.dtype)
        
        # Store original dimensions for later resizing
        original_shape = frame.shape
        
        if len(frame.shape) == 3:
            frame = frame.unsqueeze(0)  # Add batch dimension
        
        # Encode frame
        encoded = self.encoder(frame)
        
        # Embed action (we'll ignore it for now as it complicates architecture)
        # In a full implementation, we would condition the prediction on the action
        
        # Predict next frame
        predicted = self.frame_predictor(encoded)
        
        # Ensure output has the same dimensions as input
        if predicted.shape[-2:] != frame.shape[-2:]:
            predicted = F.interpolate(
                predicted, 
                size=frame.shape[-2:],  # Match height and width of input
                mode='bilinear', 
                align_corners=False
            )
            
        # Restore original dimensions (remove batch if necessary)
        if len(original_shape) == 3:
            predicted = predicted.squeeze(0)
            
        return predicted
    
    def update(self, previous_frame: torch.Tensor, action: int, current_frame: torch.Tensor):
        """Update the world model based on observed transition.
        
        Args:
            previous_frame (torch.Tensor): Previous frame [C, H, W]
            action (int): Action taken
            current_frame (torch.Tensor): Current frame [C, H, W]
        """
        prev = previous_frame.to(self.device, dtype=self.dtype)
        curr = current_frame.to(self.device, dtype=self.dtype)
        
        if len(prev.shape) == 3:
            prev = prev.unsqueeze(0)  # Add batch dimension
        if len(curr.shape) == 3:
            curr = curr.unsqueeze(0)  # Add batch dimension
        
        # Predict next frame
        encoded = self.encoder(prev)
        predicted = self.frame_predictor(encoded)
        
        # Compute loss (MSE between predicted and actual)
        loss = F.mse_loss(predicted, curr)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class StateDensityEstimator:
    """Tracks density of visited states in feature space for novelty detection."""
    
    def __init__(self, feature_dim=512, memory_size=10000):
        """Initialize state density estimator.
        
        Args:
            feature_dim (int): Dimension of feature vectors
            memory_size (int): Maximum number of states to keep in memory
        """
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.state_memory = np.zeros((memory_size, feature_dim), dtype=np.float32)
        self.memory_index = 0
        self.filled = 0
        
    def compute_novelty(self, state_embedding: torch.Tensor) -> float:
        """Compute novelty score based on distance to nearest neighbors.
        
        Args:
            state_embedding (torch.Tensor): Feature embedding of current state
            
        Returns:
            float: Novelty score (higher = more novel)
        """
        if self.filled == 0:
            # First state is always novel
            return 1.0
        
        # Convert to numpy array
        state_np = state_embedding.detach().cpu().numpy()
        
        # Compute distances to all states in memory
        if len(state_np.shape) > 1:
            state_np = state_np.squeeze()
            
        # Use only filled portion of memory
        filled_memory = self.state_memory[:self.filled]
        distances = np.linalg.norm(filled_memory - state_np, axis=1)
        
        # Use average distance to k nearest neighbors as novelty measure
        k = min(10, self.filled)
        nearest_distances = np.partition(distances, k)[:k]
        novelty_score = np.mean(nearest_distances)
        
        # Normalize novelty score (0 to 1)
        return min(1.0, novelty_score / 10.0)  # Adjust scaling factor as needed
    
    def update(self, state_embedding: torch.Tensor):
        """Add state to memory.
        
        Args:
            state_embedding (torch.Tensor): Feature embedding of current state
        """
        state_np = state_embedding.detach().cpu().numpy()
        if len(state_np.shape) > 1:
            state_np = state_np.squeeze()
            
        # Store state in memory
        self.state_memory[self.memory_index] = state_np
        
        # Update index and filled count
        self.memory_index = (self.memory_index + 1) % self.memory_size
        self.filled = min(self.filled + 1, self.memory_size)


class TemporalAssociationMemory:
    """Tracks associations between actions, states, and outcomes over time."""
    
    def __init__(self, config: HardwareConfig, history_length=100):
        """Initialize temporal association memory.
        
        Args:
            config (HardwareConfig): Hardware configuration
            history_length (int): Number of recent frames to keep for analysis
        """
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # Store recent observations, actions, and outcomes
        self.frame_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)
        self.visual_change_history = deque(maxlen=history_length)
        
        # For detecting UI elements 
        self.color_clusters = {
            'red': np.array([0, 0, 255], dtype=np.float32),  # BGR format
            'green': np.array([0, 255, 0], dtype=np.float32),
            'blue': np.array([255, 0, 0], dtype=np.float32),
            'yellow': np.array([0, 255, 255], dtype=np.float32),
            'white': np.array([255, 255, 255], dtype=np.float32)
        }
        
        # Initialize counters for color associations
        self.color_outcome_associations = {
            'red': {'positive': 0, 'negative': 0},
            'green': {'positive': 0, 'negative': 0},
            'blue': {'positive': 0, 'negative': 0},
            'yellow': {'positive': 0, 'negative': 0},
            'white': {'positive': 0, 'negative': 0}
        }
        
    def update(self, frame: torch.Tensor, action: int, next_frame: torch.Tensor):
        """Update temporal memory with new observation.
        
        Args:
            frame (torch.Tensor): Current frame
            action (int): Action taken
            next_frame (torch.Tensor): Next frame
        """
        # Store frame and action
        frame_np = frame.detach().cpu().numpy()
        next_frame_np = next_frame.detach().cpu().numpy()
        
        self.frame_history.append(frame_np)
        self.action_history.append(action)
        
        # Compute visual change
        if len(self.frame_history) > 1:
            visual_change = self._compute_visual_change(frame_np, next_frame_np)
            self.visual_change_history.append(visual_change)
            
            # Update color associations
            self._update_color_associations(frame_np, visual_change)
    
    def _compute_visual_change(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute visual change between frames.
        
        Args:
            frame1 (np.ndarray): First frame
            frame2 (np.ndarray): Second frame
            
        Returns:
            float: Visual change score (-1 to 1, positive = improvement)
        """
        # Convert to grayscale
        current_gray = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        previous_gray = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Compute structural similarity
        from skimage.metrics import structural_similarity
        score, diff = structural_similarity(previous_gray, current_gray, full=True, data_range=1.0)
        
        # Convert diff to numpy array (0-1 range)
        diff_image = (1.0 - diff) 
        
        # Compute visual change magnitude
        change_magnitude = np.mean(np.abs(diff_image))
        
        # Get predicted outcome based on visual patterns
        if len(self.frame_history) > 100:  # Only start using the analyzer after some experience
            # Get the predicted outcome for this visual change
            predicted_outcome = self.visual_change_analyzer.predict_outcome(diff_image)
            
            # Blend prediction with base magnitude (initially rely more on magnitude)
            blend_factor = min(len(self.frame_history) / 10000.0, 0.8)  # Up to 80% from prediction
            change_score = (1 - blend_factor) * change_magnitude * 5.0 + blend_factor * predicted_outcome
        else:
            change_score = change_magnitude * 5.0
        
        # Store the visual change for later analysis
        if len(self.visual_change_history) >= 5:  # Wait for some outcomes to be available
            # Use the average outcome over next few steps as the "true" outcome
            past_outcomes = np.mean([o for o in list(self.visual_change_history)[-5:]])
            # Update the analyzer with this pattern->outcome pair
            self.visual_change_analyzer.update_association(diff_image, past_outcomes)
        
        # Store the current change score to correlate with future outcomes
        self.visual_change_history.append(change_score)
        
        # Return signed change value in range [-1, 1]
        return np.clip(change_score, -1.0, 1.0)
    
    def _update_color_associations(self, frame: np.ndarray, change_score: float):
        """Update associations between colors and outcomes.
        
        Args:
            frame (np.ndarray): Frame to analyze
            change_score (float): Change score (positive = good outcome)
        """
        # Convert to BGR for color analysis
        if len(frame.shape) == 3:
            frame_bgr = frame.transpose(1, 2, 0)
        else:
            return  # Can't analyze colors without RGB channels
        
        # Count pixels of each color
        for color_name, color_value in self.color_clusters.items():
            # Create color mask
            lower = color_value * 0.8  # Allow some variation
            upper = color_value * 1.2
            mask = cv2.inRange(frame_bgr, lower, upper)
            color_pixels = np.sum(mask > 0)
            
            # If significant presence of this color
            if color_pixels > (frame_bgr.shape[0] * frame_bgr.shape[1] * 0.01):  # At least 1% of pixels
                # Update association based on outcome
                if change_score > 0.05:  # Significant positive change
                    self.color_outcome_associations[color_name]['positive'] += 1
                elif change_score < -0.05:  # Significant negative change
                    self.color_outcome_associations[color_name]['negative'] += 1
    
    def evaluate_stability(self, current_frame: torch.Tensor, previous_frames: List[torch.Tensor]) -> float:
        """Evaluate stability of game state over time.
        
        Args:
            current_frame (torch.Tensor): Current frame
            previous_frames (List[torch.Tensor]): Previous frames
            
        Returns:
            float: Stability score (higher = more stable with positive indicators)
        """
        if len(previous_frames) < 5:  # Need sufficient history
            return 0.0
            
        # Extract learned color preferences
        color_preferences = {}
        for color_name, counts in self.color_outcome_associations.items():
            total = counts['positive'] + counts['negative']
            if total > 0:
                preference = (counts['positive'] - counts['negative']) / total
                color_preferences[color_name] = preference
        
        # Apply learned preferences to current frame
        current_np = current_frame.detach().cpu().numpy()
        if len(current_np.shape) == 3:
            current_bgr = current_np.transpose(1, 2, 0)
        else:
            return 0.0
            
        weighted_score = 0.0
        total_weight = 0.0
        
        # For each color we've learned about
        for color_name, preference in color_preferences.items():
            color_value = self.color_clusters[color_name]
            
            # Create color mask
            lower = color_value * 0.8
            upper = color_value * 1.2
            mask = cv2.inRange(current_bgr, lower, upper)
            color_pixels = np.sum(mask > 0) / (current_bgr.shape[0] * current_bgr.shape[1])
            
            # Weight by preference and prevalence
            weighted_score += preference * color_pixels
            total_weight += abs(preference) * color_pixels
        
        # Normalize score
        if total_weight > 0:
            stability_score = weighted_score / total_weight
        else:
            stability_score = 0.0
            
        return stability_score


class ActionOutcomeTracker:
    """Tracks outcomes of actions to guide exploration strategy."""
    
    def __init__(self, num_actions=20):
        """Initialize action outcome tracker.
        
        Args:
            num_actions (int): Number of possible actions
        """
        self.num_actions = num_actions
        self.action_counts = np.zeros(num_actions, dtype=np.int32)
        self.action_outcomes = np.zeros(num_actions, dtype=np.float32)
        
    def update(self, action: int, visual_change: float, stability_score: float):
        """Update action outcome statistics.
        
        Args:
            action (int): Action taken
            visual_change (float): Visual change score
            stability_score (float): Stability score
        """
        if action >= self.num_actions:
            return
            
        # Combine outcomes into single score
        outcome = 0.7 * visual_change + 0.3 * stability_score
        
        # Update running average of outcomes for this action
        self.action_counts[action] += 1
        self.action_outcomes[action] += (outcome - self.action_outcomes[action]) / self.action_counts[action]
    
    def get_action_preference(self, action: int) -> float:
        """Get learned preference for an action.
        
        Args:
            action (int): Action index
            
        Returns:
            float: Preference score for action (-1 to 1)
        """
        if action >= self.num_actions or self.action_counts[action] == 0:
            return 0.0
            
        return self.action_outcomes[action]


class VisualChangeAnalyzer:
    """Analyzes and learns correlations between visual changes and outcomes."""
    
    def __init__(self, feature_dim=64, memory_size=1000):
        """Initialize visual change analyzer.
        
        Args:
            feature_dim (int): Feature dimension for pattern encoding
            memory_size (int): Size of pattern memory
        """
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        
        # Memory for storing pattern->outcome associations
        self.pattern_features = np.zeros((memory_size, feature_dim), dtype=np.float32)
        self.pattern_outcomes = np.zeros(memory_size, dtype=np.float32)
        self.memory_index = 0
        self.memory_filled = 0
        
        # CNN for pattern encoding (we'll use a simple one for now)
        self.pattern_encoder = None
        self.initialized = False
        
        # Statistics for normalization
        self.outcome_mean = 0.0
        self.outcome_std = 1.0
        self.update_counter = 0
        
    def initialize_encoder(self, input_shape):
        """Initialize the pattern encoder with correct input shape.
        
        Args:
            input_shape (tuple): Shape of the difference images
        """
        import torch.nn as nn
        
        # Create a simple CNN for encoding difference patterns
        self.pattern_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.initialized = True
        
    def encode_pattern(self, diff_image: np.ndarray) -> np.ndarray:
        """Encode a difference image into feature space.
        
        Args:
            diff_image (np.ndarray): Difference image
            
        Returns:
            np.ndarray: Feature vector
        """
        import torch
        
        # Ensure encoder is initialized
        if not self.initialized:
            h, w = diff_image.shape
            self.initialize_encoder((1, h, w))
            
        # Convert to tensor
        diff_tensor = torch.from_numpy(diff_image).float().unsqueeze(0).unsqueeze(0)
        
        # Encode pattern
        with torch.no_grad():
            features = self.pattern_encoder(diff_tensor)
            
        return features.numpy().squeeze()
        
    def update_association(self, diff_image: np.ndarray, future_outcome: float):
        """Update association between pattern and outcome.
        
        Args:
            diff_image (np.ndarray): Difference image
            future_outcome (float): Observed outcome after this change
        """
        # Encode pattern
        feature_vector = self.encode_pattern(diff_image)
        
        # Store in memory
        self.pattern_features[self.memory_index] = feature_vector
        self.pattern_outcomes[self.memory_index] = future_outcome
        
        # Update index and filled count
        self.memory_index = (self.memory_index + 1) % self.memory_size
        self.memory_filled = min(self.memory_filled + 1, self.memory_size)
        
        # Update statistics for normalization
        self.update_counter += 1
        if self.memory_filled > 10:  # Only update after we have some data
            self.outcome_mean = np.mean(self.pattern_outcomes[:self.memory_filled])
            self.outcome_std = np.std(self.pattern_outcomes[:self.memory_filled]) + 1e-6
        
    def predict_outcome(self, diff_image: np.ndarray) -> float:
        """Predict future outcome from a visual change pattern.
        
        Args:
            diff_image (np.ndarray): Difference image
            
        Returns:
            float: Predicted outcome (-1 to 1, positive = good change)
        """
        if self.memory_filled < 10:  # Not enough data to make predictions
            return 0.0
            
        # Encode pattern
        feature_vector = self.encode_pattern(diff_image)
        
        # Find similar patterns in memory
        similarities = np.zeros(self.memory_filled)
        for i in range(self.memory_filled):
            similarities[i] = np.dot(feature_vector, self.pattern_features[i]) / (
                np.linalg.norm(feature_vector) * np.linalg.norm(self.pattern_features[i]) + 1e-8)
        
        # Get top k similar patterns
        k = min(5, self.memory_filled)
        top_indices = np.argsort(similarities)[-k:]
        
        # Weight by similarity
        total_similarity = np.sum(similarities[top_indices])
        if total_similarity > 0:
            weighted_outcome = np.sum(similarities[top_indices] * self.pattern_outcomes[top_indices]) / total_similarity
            
            # Normalize to -1 to 1 range
            if self.update_counter > 100:  # Only normalize after sufficient updates
                normalized_outcome = (weighted_outcome - self.outcome_mean) / self.outcome_std
                return np.clip(normalized_outcome, -1.0, 1.0)
            else:
                return np.clip(weighted_outcome / 2.0, -1.0, 1.0)  # Simple normalization early on
        else:
            return 0.0


class AutonomousRewardSystem:
    """Autonomous reward system that learns from intrinsic motivation and visual patterns."""
    
    def __init__(self, config: HardwareConfig):
        """Initialize autonomous reward system.
        
        Args:
            config (HardwareConfig): Hardware and training configuration
        """
        self.config = config
        self.device = config.get_device()
        self.dtype = torch.float32  # Add the missing dtype attribute
        
        # World model for prediction and feature extraction
        self.world_model = WorldModelCNN(config)
        
        # State density estimator for novelty detection
        self.state_archive = StateDensityEstimator()
        
        # Temporal association memory
        self.temporal_memory = TemporalAssociationMemory(config)
        
        # Action outcome tracker
        self.action_tracker = ActionOutcomeTracker()
        
        # Visual change analyzer
        self.visual_change_analyzer = VisualChangeAnalyzer()
        
        # Frame history
        self.frame_history = deque(maxlen=10)
        self.outcome_history = deque(maxlen=10)
        
        # Training phase weights (adjust emphasis over time)
        self.training_phase = 0
        self.phase_weights = [
            # Initial phase: heavy exploration
            {
                'curiosity': 0.5,
                'novelty': 0.4,
                'visual_change': 0.1,
                'stability': 0.0
            },
            # Middle phase: balanced learning
            {
                'curiosity': 0.3,
                'novelty': 0.3,
                'visual_change': 0.2,
                'stability': 0.2
            },
            # Late phase: exploitation of learned patterns
            {
                'curiosity': 0.1,
                'novelty': 0.1,
                'visual_change': 0.4,
                'stability': 0.4
            }
        ]
        
        self.current_weights = self.phase_weights[0]
        self.phase_transition_steps = [10000, 50000]  # When to transition phases
        self.total_steps = 0
        
        # For logging
        self.reward_components = {
            'curiosity': [],
            'novelty': [],
            'visual_change': [],
            'stability': []
        }
        
    def compute_reward(self, current_frame: torch.Tensor, action_taken: int) -> float:
        """Compute reward based on intrinsic motivation and learned patterns.
        
        Args:
            current_frame (torch.Tensor): Current frame
            action_taken (int): Action that was taken
            
        Returns:
            float: Computed reward
        """
        # Update step count and possibly training phase
        self.total_steps += 1
        self._update_training_phase()
        
        # Ensure the frame is correctly shaped and on the right device
        if current_frame is None:
            # In case of a missing frame, use a zero frame
            height, width = getattr(self.config, 'resolution', (240, 320))
            current_frame = torch.zeros(3, height, width, device=self.device, dtype=self.dtype)
        else:
            # Move to correct device
            current_frame = current_frame.to(self.device, dtype=self.dtype)
            
            # Ensure dimensions match expected resolution
            expected_height, expected_width = getattr(self.config, 'resolution', (240, 320))
            if current_frame.shape[-2:] != (expected_height, expected_width):
                # Reshape frame to match expected dimensions
                current_frame = F.interpolate(
                    current_frame.unsqueeze(0) if current_frame.dim() == 3 else current_frame,
                    size=(expected_height, expected_width),
                    mode='bilinear',
                    align_corners=False
                )
                current_frame = current_frame.squeeze(0) if current_frame.dim() == 4 else current_frame
        
        # Store frame in history
        self.frame_history.append(current_frame)
        
        # If we don't have enough history, return neutral reward
        if len(self.frame_history) < 2:
            return 0.0
        
        # Get previous frame
        previous_frame = self.frame_history[-2]
        
        # 1. Prediction-based curiosity reward
        predicted_frame = self.world_model.predict_next_frame(list(self.frame_history)[:-1], action_taken)
        prediction_error = self._compute_prediction_error(predicted_frame, current_frame)
        curiosity_reward = self._normalize_curiosity(prediction_error)
        
        # 2. Novelty detection reward
        frame_embedding = self.world_model.encode_frame(current_frame)
        novelty_score = self.state_archive.compute_novelty(frame_embedding)
        
        # 3. Visual change detection
        visual_change = self._detect_visual_changes(current_frame, previous_frame)
        
        # 4. Temporal stability evaluation
        stability_score = self.temporal_memory.evaluate_stability(current_frame, list(self.frame_history))
        
        # Update models and memories
        self.world_model.update(previous_frame, action_taken, current_frame)
        self.state_archive.update(frame_embedding)
        self.temporal_memory.update(previous_frame, action_taken, current_frame)
        self.action_tracker.update(action_taken, visual_change, stability_score)
        
        # Combine rewards using current phase weights
        reward = (
            self.current_weights['curiosity'] * curiosity_reward + 
            self.current_weights['novelty'] * novelty_score + 
            self.current_weights['visual_change'] * visual_change + 
            self.current_weights['stability'] * stability_score
        )
        
        # Store reward components for logging
        self.reward_components['curiosity'].append(curiosity_reward)
        self.reward_components['novelty'].append(novelty_score)
        self.reward_components['visual_change'].append(visual_change)
        self.reward_components['stability'].append(stability_score)
        
        # Log periodically
        if self.total_steps % 100 == 0:
            self._log_rewards()
        
        return reward
    
    def _compute_prediction_error(self, predicted: torch.Tensor, actual: torch.Tensor) -> float:
        """Compute prediction error between predicted and actual frames.
        
        Args:
            predicted (torch.Tensor): Predicted frame
            actual (torch.Tensor): Actual frame
            
        Returns:
            float: Prediction error (MSE)
        """
        # Move tensors to the same device if needed
        if predicted.device != actual.device:
            predicted = predicted.to(actual.device)
            
        # Ensure both tensors have same batch dimension
        if len(predicted.shape) != len(actual.shape):
            if len(predicted.shape) < len(actual.shape):
                predicted = predicted.unsqueeze(0)
            else:
                actual = actual.unsqueeze(0)
        
        # Check for dimension mismatch and resize if needed
        if predicted.shape != actual.shape:
            # Get target shape (use actual frame dimensions)
            _, c, h, w = actual.shape if len(actual.shape) == 4 else (1, *actual.shape)
            
            # Resize predicted to match actual dimensions
            predicted = F.interpolate(
                predicted.view(-1, *predicted.shape[-3:]) if len(predicted.shape) > 3 else predicted.unsqueeze(0),
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )
            
            # Restore original batch dimension if needed
            if len(actual.shape) == 3:
                predicted = predicted.squeeze(0)
                
        # Compute mean squared error
        mse = F.mse_loss(predicted, actual).item()
        return mse
    
    def _normalize_curiosity(self, prediction_error: float) -> float:
        """Normalize curiosity reward to prevent exploitation of unpredictable states.
        
        Args:
            prediction_error (float): Raw prediction error
            
        Returns:
            float: Normalized curiosity reward
        """
        # Apply sigmoid-like normalization to prevent excessive rewards for chaos
        normalized = 2.0 / (1.0 + np.exp(-10.0 * prediction_error)) - 1.0
        
        # Decay curiosity as training progresses
        phase_discount = 1.0 - (self.total_steps / 100000.0)  # Adjust for your training schedule
        phase_discount = max(0.1, min(1.0, phase_discount))
        
        return normalized * phase_discount
    
    def _detect_visual_changes(self, current_frame: torch.Tensor, previous_frame: torch.Tensor) -> float:
        """Detect visual changes between frames to reward progress.
        
        Args:
            current_frame (torch.Tensor): Current frame
            previous_frame (torch.Tensor): Previous frame
            
        Returns:
            float: Visual change score (positive = improvement)
        """
        # Convert to numpy for OpenCV
        current_np = current_frame.detach().cpu().numpy()
        previous_np = previous_frame.detach().cpu().numpy()
        
        # Ensure correct format
        if len(current_np.shape) == 3:
            current_np = current_np.transpose(1, 2, 0)
            previous_np = previous_np.transpose(1, 2, 0)
            
        # Convert to grayscale
        current_gray = cv2.cvtColor(current_np, cv2.COLOR_RGB2GRAY)
        previous_gray = cv2.cvtColor(previous_np, cv2.COLOR_RGB2GRAY)
        
        # Compute structural similarity
        from skimage.metrics import structural_similarity
        score, diff = structural_similarity(previous_gray, current_gray, full=True, data_range=1.0)
        
        # Convert diff to numpy array (0-1 range)
        diff_image = (1.0 - diff) 
        
        # Compute visual change magnitude
        change_magnitude = np.mean(np.abs(diff_image))
        
        # Get predicted outcome based on visual patterns
        if self.total_steps > 100:  # Only start using the analyzer after some experience
            # Get the predicted outcome for this visual change
            predicted_outcome = self.visual_change_analyzer.predict_outcome(diff_image)
            
            # Blend prediction with base magnitude (initially rely more on magnitude)
            blend_factor = min(self.total_steps / 10000.0, 0.8)  # Up to 80% from prediction
            change_score = (1 - blend_factor) * change_magnitude * 5.0 + blend_factor * predicted_outcome
        else:
            change_score = change_magnitude * 5.0
        
        # Store the visual change for later analysis
        if len(self.outcome_history) >= 5:  # Wait for some outcomes to be available
            # Use the average outcome over next few steps as the "true" outcome
            past_outcomes = np.mean([o for o in list(self.outcome_history)[-5:]])
            # Update the analyzer with this pattern->outcome pair
            self.visual_change_analyzer.update_association(diff_image, past_outcomes)
        
        # Store the current change score to correlate with future outcomes
        self.outcome_history.append(change_score)
        
        # Return signed change value in range [-1, 1]
        return np.clip(change_score, -1.0, 1.0)
    
    def _update_training_phase(self):
        """Update training phase based on steps."""
        for i, threshold in enumerate(self.phase_transition_steps):
            if self.total_steps >= threshold and self.training_phase == i:
                self.training_phase = i + 1
                self.current_weights = self.phase_weights[i + 1]
                logger.info(f"Transitioning to training phase {i + 1}")
    
    def _log_rewards(self):
        """Log reward statistics."""
        # Compute means for each component
        means = {k: np.mean(v[-100:]) for k, v in self.reward_components.items()}
        
        logger.info(f"Step {self.total_steps}, "
                   f"Phase: {self.training_phase}, "
                   f"Curiosity: {means['curiosity']:.4f}, "
                   f"Novelty: {means['novelty']:.4f}, "
                   f"Visual Change: {means['visual_change']:.4f}, "
                   f"Stability: {means['stability']:.4f}")
                   
    def calculate_menu_penalty(self, in_menu: bool, consecutive_menu_steps: int = 0) -> float:
        """Calculate penalty for getting stuck in menus.
        
        Args:
            in_menu (bool): Whether the agent is currently in a menu
            consecutive_menu_steps (int): Number of consecutive steps spent in menus
            
        Returns:
            float: Penalty value (negative reward)
        """
        if not in_menu:
            return 0.0
            
        # Apply escalating penalty based on how long the agent has been stuck
        # This encourages the agent to learn to exit menus quickly
        base_penalty = -0.2
        escalation_factor = min(consecutive_menu_steps * 0.1, 1.0)
        
        return base_penalty * (1.0 + escalation_factor) 