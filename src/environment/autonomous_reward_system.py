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
        # Calculate the output size from encoder dynamically
        with torch.no_grad():
            # Get resolution from config
            width, height = getattr(config, 'resolution', (1920, 1080))
            # Scale down to match the expected input resolution for the model
            model_width, model_height = 320, 240  # Typical processing resolution
            # Create a dummy input to compute output shape - MUST use the same dtype as the encoder
            dummy_input = torch.zeros(1, 3, model_height, model_width, 
                                     device=self.device, dtype=self.dtype)
            encoder_output = self.encoder(dummy_input)
            flattened_size = encoder_output.numel() // encoder_output.size(0)
            
        self.feature_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),  # Dynamically sized input
            nn.ReLU()
        ).to(self.device, dtype=self.dtype)
        
        # Action embedding layer - maps discrete action index to embedding
        self.action_embedding = nn.Embedding(1000, 64).to(self.device, dtype=self.dtype)  # Support up to 1000 actions
        
        # Action fusion layer - combines action embedding with visual features
        self.action_fusion = nn.Sequential(
            nn.Linear(64, encoder_output.size(1) * encoder_output.size(2) * encoder_output.size(3)),
            nn.ReLU()
        ).to(self.device, dtype=self.dtype)
        
        # World model (predicts next frame)
        self.frame_predictor = nn.Sequential(
            nn.Conv2d(64 + 1, 64, kernel_size=3, padding=1),  # +1 channel for action influence
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=8, stride=4, padding=2, output_padding=0)
        ).to(self.device, dtype=self.dtype)
        
        # Store encoder output shape for reshaping
        self.encoder_shape = encoder_output.shape[1:]
        
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
        
        # Embed and incorporate action
        action_tensor = torch.tensor([action], device=self.device).long()
        action_embedding = self.action_embedding(action_tensor)
        
        # Reshape action embedding to match spatial dimensions of encoded frame
        action_spatial = self.action_fusion(action_embedding).view(-1, *self.encoder_shape)
        
        # Create an action influence channel
        batch_size, channels, height, width = encoded.shape
        action_channel = torch.zeros((batch_size, 1, height, width), device=self.device, dtype=self.dtype)
        
        # Apply action influence
        for b in range(batch_size):
            action_channel[b, 0] = action_spatial[b, 0].view(height, width)
        
        # Concatenate features with action channel
        combined_features = torch.cat([encoded, action_channel], dim=1)
        
        # Predict next frame using the combined features
        predicted = self.frame_predictor(combined_features)
        
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
        
        # Encode previous frame
        encoded = self.encoder(prev)
        
        # Embed and incorporate action
        action_tensor = torch.tensor([action], device=self.device).long()
        action_embedding = self.action_embedding(action_tensor)
        
        # Reshape action embedding to match spatial dimensions of encoded frame
        action_spatial = self.action_fusion(action_embedding).view(-1, *self.encoder_shape)
        
        # Create an action influence channel
        batch_size, channels, height, width = encoded.shape
        action_channel = torch.zeros((batch_size, 1, height, width), device=self.device, dtype=self.dtype)
        
        # Apply action influence
        for b in range(batch_size):
            action_channel[b, 0] = action_spatial[b, 0].view(height, width)
        
        # Concatenate features with action channel
        combined_features = torch.cat([encoded, action_channel], dim=1)
        
        # Predict next frame
        predicted = self.frame_predictor(combined_features)
        
        # Ensure tensors have the same shape before computing loss
        if predicted.shape != curr.shape:
            predicted = F.interpolate(
                predicted, 
                size=(curr.shape[2], curr.shape[3]),
                mode='bilinear',
                align_corners=False
            )
        
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
        
        # If only one state in memory, just return the distance to that state
        if self.filled == 1:
            return min(1.0, distances[0] / 10.0)
        
        # Use average distance to k nearest neighbors as novelty measure
        k = min(10, self.filled)  # Ensure k is not larger than number of states
        if k <= 2:  # If we have 2 or fewer states in memory
            return min(1.0, np.mean(distances) / 10.0)
            
        # Get k nearest neighbors - ensure k is at least 3 to avoid np.partition errors
        k = max(3, k)
        if k >= len(distances):
            return min(1.0, np.mean(distances) / 10.0)
            
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
        
        # Initialize the visual change analyzer
        self.visual_change_analyzer = VisualChangeAnalyzer()
        
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
        # Convert tensors to numpy arrays if needed
        if torch.is_tensor(frame1):
            frame1 = frame1.cpu().numpy()
        if torch.is_tensor(frame2):
            frame2 = frame2.cpu().numpy()
            
        # Ensure correct channel ordering (from PyTorch's CHW to OpenCV's HWC)
        if frame1.shape[0] == 3:  # If channels are first
            frame1 = np.transpose(frame1, (1, 2, 0))
        if frame2.shape[0] == 3:  # If channels are first
            frame2 = np.transpose(frame2, (1, 2, 0))
            
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
    """Analyzes visual changes and learns to associate them with outcomes."""
    
    def __init__(self, memory_size=1000):
        """Initialize visual change analyzer.
        
        Args:
            memory_size (int): Number of patterns to store
        """
        self.pattern_memory = []
        self.outcome_memory = []
        self.memory_size = memory_size
        
    def update_association(self, pattern: np.ndarray, outcome: float):
        """Update association between pattern and outcome.
        
        Args:
            pattern (np.ndarray): Visual change pattern
            outcome (float): Observed outcome
        """
        # Downsample pattern for memory efficiency
        h, w = pattern.shape
        downsampled = cv2.resize(pattern, (w//4, h//4))
        flattened = downsampled.flatten()
        
        # Store pattern and outcome
        self.pattern_memory.append(flattened)
        self.outcome_memory.append(outcome)
        
        # Limit memory size
        if len(self.pattern_memory) > self.memory_size:
            self.pattern_memory.pop(0)
            self.outcome_memory.pop(0)
    
    def predict_outcome(self, pattern: np.ndarray) -> float:
        """Predict outcome for pattern.
        
        Args:
            pattern (np.ndarray): Visual change pattern
            
        Returns:
            float: Predicted outcome
        """
        if not self.pattern_memory:
            return 0.0
            
        # Downsample pattern
        h, w = pattern.shape
        downsampled = cv2.resize(pattern, (w//4, h//4))
        flattened = downsampled.flatten()
        
        # Find k nearest neighbors
        k = min(5, len(self.pattern_memory))
        distances = []
        
        for stored_pattern in self.pattern_memory:
            # Ensure dimensions match
            if len(stored_pattern) != len(flattened):
                continue
                
            # Compute distance
            distance = np.linalg.norm(stored_pattern - flattened)
            distances.append(distance)
            
        if not distances:
            return 0.0
            
        # Find k smallest distances
        indices = np.argsort(distances)[:k]
        
        # Weight by inverse distance
        weights = [1.0 / (distances[i] + 1e-6) for i in indices]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 0.0
            
        # Weighted average of outcomes
        prediction = sum(weights[i] * self.outcome_memory[indices[i]] for i in range(k)) / total_weight
        
        return prediction


class AutonomousRewardSystem:
    """Autonomous reward system that learns to provide rewards without explicit game metrics."""
    
    def __init__(self, config: HardwareConfig):
        """Initialize the autonomous reward system.
        
        Args:
            config (HardwareConfig): Hardware configuration
        """
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # World model for prediction and novelty detection
        self.world_model = WorldModelCNN(config)
        
        # State density estimator for novelty detection
        self.density_estimator = StateDensityEstimator(feature_dim=512)
        
        # Temporal association memory for action-outcome associations
        self.association_memory = TemporalAssociationMemory(config)
        
        # Track history for predicting outcomes
        self.frame_history = []
        self.max_history_length = 5
        
        # Feature extractor for state representations
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 9 * 7, 512),  # Assuming input resolution is 320x240
            nn.ReLU()
        ).to(self.device, dtype=self.dtype)
        
        # Adaptive normalization parameters
        self.reward_scale = 1.0
        self.reward_shift = 0.0
        self.reward_history = []
        self.history_size = 1000
        
        # Default max menu penalty
        self.max_menu_penalty = -20.0
        
    def compute_reward(self, prev_frame: torch.Tensor, action_idx: int, curr_frame: torch.Tensor) -> float:
        """Compute reward for the current state-action-next_state transition.
        
        Args:
            prev_frame (torch.Tensor): Previous frame (before action)
            action_idx (int): Index of the action taken
            curr_frame (torch.Tensor): Current frame (after action)
            
        Returns:
            float: Computed reward
        """
        # Calculate multiple reward components
        
        # 1. Prediction error reward (novelty)
        prediction_reward = self._prediction_error_reward(prev_frame, action_idx, curr_frame)
        
        # 2. Visual change reward (structural changes)
        visual_change_reward = self._visual_change_reward(prev_frame, curr_frame)
        
        # 3. State density reward (exploration)
        state_embedding = self._get_state_embedding(curr_frame)
        density_reward = self._density_reward(state_embedding)
        
        # 4. Temporal association reward (learned preferences)
        association_reward = self._association_reward(prev_frame, action_idx, curr_frame)
        
        # Combine reward components with weights
        reward = (
            0.3 * prediction_reward +    # 30% weight for prediction error
            0.4 * visual_change_reward + # 40% weight for visual changes
            0.1 * density_reward +       # 10% weight for exploration
            0.2 * association_reward     # 20% weight for learned preferences
        )
        
        # Apply adaptive normalization
        reward = self._normalize_reward(reward)
        
        # Update components with new data
        self._update(prev_frame, action_idx, curr_frame, reward)
        
        return reward
        
    def _prediction_error_reward(self, prev_frame: torch.Tensor, action_idx: int, curr_frame: torch.Tensor) -> float:
        """Compute reward component based on prediction error.
        
        Args:
            prev_frame (torch.Tensor): Previous frame
            action_idx (int): Action index
            curr_frame (torch.Tensor): Current frame
            
        Returns:
            float: Prediction error reward
        """
        # Update frame history
        self._update_frame_history(prev_frame)
        
        # Get prediction of current frame based on previous frames and action
        predicted_frame = self.world_model.predict_next_frame(self.frame_history, action_idx)
        
        # Compute prediction error
        if predicted_frame is not None and curr_frame is not None:
            if isinstance(predicted_frame, torch.Tensor) and isinstance(curr_frame, torch.Tensor):
                # Ensure frames are on the same device
                if predicted_frame.device != curr_frame.device:
                    predicted_frame = predicted_frame.to(curr_frame.device)
                
                # Compute L2 error
                error = torch.mean((predicted_frame - curr_frame) ** 2).item()
                
                # Scale error to reasonable range - novelty is both good and bad
                # Small errors (gradual change) -> positive reward
                # Large errors (unexpected change) -> negative reward
                # Tuned thresholds based on typical prediction errors
                if error < 0.05:
                    # Small errors are generally good (progress)
                    return max(0.0, 0.5 - error * 10.0)  # +0.5 to 0.0 as error increases
                else:
                    # Large errors can be disruptive/bad (failures, menus, etc.)
                    return -min(1.0, (error - 0.05) * 5.0)  # 0.0 to -1.0 as error increases
        
        # Default to small positive reward if prediction unavailable
        return 0.01
    
    def _visual_change_reward(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> float:
        """Compute reward component based on visual change.
        
        Args:
            prev_frame (torch.Tensor): Previous frame
            curr_frame (torch.Tensor): Current frame
            
        Returns:
            float: Visual change reward
        """
        # Convert to numpy for OpenCV
        prev_np = prev_frame.detach().cpu().numpy()
        curr_np = curr_frame.detach().cpu().numpy()
        
        # Ensure correct format
        if len(prev_np.shape) == 3:
            prev_np = prev_np.transpose(1, 2, 0)
            curr_np = curr_np.transpose(1, 2, 0)
            
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_np, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_np, cv2.COLOR_RGB2GRAY)
        
        # Compute structural similarity
        from skimage.metrics import structural_similarity
        score, diff = structural_similarity(prev_gray, curr_gray, full=True, data_range=1.0)
        
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
        if len(self.reward_history) >= 5:  # Wait for some outcomes to be available
            # Use the average outcome over next few steps as the "true" outcome
            past_outcomes = np.mean([o for o in list(self.reward_history)[-5:]])
            # Update the analyzer with this pattern->outcome pair
            self.visual_change_analyzer.update_association(diff_image, past_outcomes)
        
        # Store the current change score to correlate with future outcomes
        self.reward_history.append(change_score)
        
        # Return signed change value in range [-1, 1]
        return np.clip(change_score, -1.0, 1.0)
    
    def _get_state_embedding(self, frame: torch.Tensor) -> torch.Tensor:
        """Get state embedding from frame."""
        return self.feature_extractor(frame)
    
    def _density_reward(self, state_embedding: torch.Tensor) -> float:
        """Compute reward component based on state density."""
        return self.density_estimator.compute_novelty(state_embedding)
    
    def _association_reward(self, prev_frame: torch.Tensor, action_idx: int, curr_frame: torch.Tensor) -> float:
        """Compute reward component based on temporal association."""
        return self.association_memory.evaluate_stability(curr_frame, [prev_frame])
    
    def _normalize_reward(self, reward: float) -> float:
        """Apply adaptive normalization to reward."""
        # Update reward history
        self.reward_history.append(reward)
        
        # Compute moving average of rewards
        if len(self.reward_history) > self.history_size:
            self.reward_history.pop(0)
        average_reward = np.mean(self.reward_history)
        
        # Scale and shift reward
        scaled_reward = self.reward_scale * (reward - average_reward) + self.reward_shift
        
        return scaled_reward
    
    def _update(self, prev_frame: torch.Tensor, action_idx: int, curr_frame: torch.Tensor, reward: float):
        """Update components with new data."""
        self._update_frame_history(prev_frame)
        self.density_estimator.update(self._get_state_embedding(curr_frame))
        self.association_memory.update(prev_frame, action_idx, curr_frame)
        
    def _update_frame_history(self, frame: torch.Tensor):
        """Update frame history."""
        self.frame_history.append(frame.detach().cpu().numpy())
        if len(self.frame_history) > self.max_history_length:
            self.frame_history.pop(0)
        
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
        base_penalty = -5.0  # Increased from -0.2 to -5.0 for stronger discouragement
        # Escalate much faster based on consecutive steps
        escalation_factor = min(consecutive_menu_steps * 0.5, 10.0)  # Faster escalation, up to 10x
        
        return base_penalty * (1.0 + escalation_factor) 