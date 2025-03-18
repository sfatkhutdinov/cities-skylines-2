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


class DensityEstimator:
    """Tracks density of visited states in feature space for novelty detection."""
    
    def __init__(self, feature_dim=512, history_size=2000, device=None):
        """Initialize state density estimator.
        
        Args:
            feature_dim (int): Dimension of feature vectors
            history_size (int): Maximum number of states to keep in memory
            device: Device for computations
        """
        self.feature_dim = feature_dim
        self.history_size = history_size
        self.device = device
        self.state_memory = []
        self.mean_vector = None
        self.std_vector = None
        
    def compute_novelty(self, state_embedding: torch.Tensor) -> float:
        """Compute novelty score based on distance to nearest neighbors.
        
        Args:
            state_embedding (torch.Tensor): Feature embedding of current state
            
        Returns:
            float: Novelty score (higher = more novel)
        """
        if not self.state_memory:
            # First state is always novel
            return 1.0
        
        # Convert to device if needed
        if self.device is not None:
            state_embedding = state_embedding.to(self.device)
        
        # Normalize embedding if we have statistics
        if self.mean_vector is not None:
            state_embedding = (state_embedding - self.mean_vector) / (self.std_vector + 1e-8)
            
        # Compute distances to all states in memory
        distances = []
        for stored_state in self.state_memory:
            # Using dot product as a similarity measure (higher = more similar)
            # Convert to distance where higher = more different
            similarity = torch.nn.functional.cosine_similarity(state_embedding, stored_state, dim=0)
            distance = 1.0 - similarity.item()
            distances.append(distance)
            
        # Use average distance to k nearest neighbors as novelty measure
        k = min(5, len(distances))
        distances.sort()  # Sort in ascending order
        novelty = sum(distances[:k]) / k
        
        # Normalize to 0-1 range
        novelty = min(1.0, max(0.0, novelty))
        
        return novelty
        
    def update(self, state_embedding: torch.Tensor) -> None:
        """Update memory with new state embedding.
        
        Args:
            state_embedding (torch.Tensor): Feature embedding of current state
        """
        # Convert to device if needed
        if self.device is not None:
            state_embedding = state_embedding.to(self.device)
            
        # Add to memory
        self.state_memory.append(state_embedding.detach())
        
        # Keep memory size within limit
        if len(self.state_memory) > self.history_size:
            self.state_memory.pop(0)
            
        # Update statistics periodically
        if len(self.state_memory) % 50 == 0:
            self._update_statistics()
            
    def _update_statistics(self) -> None:
        """Update mean and standard deviation of stored embeddings."""
        if not self.state_memory:
            return
            
        # Stack all embeddings
        all_embeddings = torch.stack(self.state_memory)
        
        # Compute mean and std along batch dimension
        self.mean_vector = torch.mean(all_embeddings, dim=0)
        self.std_vector = torch.std(all_embeddings, dim=0)


class TemporalAssociationMemory:
    """Tracks associations between actions, states, and outcomes over time."""
    
    def __init__(self, feature_dim: int = 512, history_size: int = 1000, device=None):
        """Initialize temporal association memory.
        
        Args:
            feature_dim: Dimension of feature embeddings
            history_size: Maximum number of experiences to store
            device: Torch device for computations
        """
        self.feature_dim = feature_dim
        self.history_size = history_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Memory buffers for experiences
        self.state_buffer = [] 
        self.action_buffer = []
        self.next_state_buffer = []
        self.reward_buffer = []
        
        # Learned associations
        self.action_effect_model = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),  # +1 for action index
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.action_effect_model.parameters(), lr=0.001)
        
    def update(self, state_features: torch.Tensor, action_idx: int, next_state_features: torch.Tensor, reward: float = 0.0):
        """Update memory with new experience.
        
        Args:
            state_features: Features of current state
            action_idx: Action taken
            next_state_features: Features of next state
            reward: Observed reward (optional)
        """
        # Ensure tensors are detached from computation graph
        if state_features is not None:
            state_features = state_features.detach().cpu()
        if next_state_features is not None:
            next_state_features = next_state_features.detach().cpu()
            
        # Store experience
        self.state_buffer.append(state_features)
        self.action_buffer.append(action_idx)
        self.next_state_buffer.append(next_state_features)
        self.reward_buffer.append(reward)
        
        # Maintain buffer size limit
        if len(self.state_buffer) > self.history_size:
            self.state_buffer.pop(0)
            self.action_buffer.pop(0)
            self.next_state_buffer.pop(0)
            self.reward_buffer.pop(0)
            
        # Learn from experiences periodically
        if len(self.state_buffer) % 50 == 0 and len(self.state_buffer) >= 100:
            self._learn_associations()
    
    def _learn_associations(self):
        """Learn to predict state transitions based on collected experiences."""
        if len(self.state_buffer) < 100:
            return
            
        # Sample batch of experiences
        batch_size = min(64, len(self.state_buffer))
        indices = np.random.choice(len(self.state_buffer), batch_size, replace=False)
        
        # Prepare batch
        states = torch.stack([self.state_buffer[i] for i in indices]).to(self.device)
        actions = torch.tensor([self.action_buffer[i] for i in indices], 
                              dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack([self.next_state_buffer[i] for i in indices]).to(self.device)
        
        # Combine state and action
        state_action = torch.cat([states, actions], dim=1)
        
        # Predict next state
        predicted_delta = self.action_effect_model(state_action)
        predicted_next_state = states + predicted_delta
        
        # Compute loss
        loss = F.mse_loss(predicted_next_state, next_states)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def compute_association(self, state_features: torch.Tensor, action_idx: int, next_state_features: torch.Tensor) -> torch.Tensor:
        """Compute how expected/consistent a transition is based on learned associations.
        
        Args:
            state_features: Features of current state
            action_idx: Action taken
            next_state_features: Features of next state
            
        Returns:
            torch.Tensor: Association score (higher = more expected/consistent)
        """
        with torch.no_grad():
            # Combine state and action
            state_action = torch.cat([
                state_features, 
                torch.tensor([[float(action_idx)]], device=state_features.device)
            ], dim=1)
            
            # Predict next state
            predicted_delta = self.action_effect_model(state_action)
            predicted_next_state = state_features + predicted_delta
            
            # Compute prediction error
            prediction_error = F.mse_loss(predicted_next_state, next_state_features)
            
            # Convert to association score (inverse of error)
            # Normalize to 0-1 range with exponential scaling
            association_score = torch.exp(-5.0 * prediction_error)
            
            return association_score


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
    
    def __init__(self, config: HardwareConfig, history_length: int = 100):
        """Initialize the autonomous reward system.
        
        Args:
            config: Hardware configuration
            history_length: Length of history to maintain
        """
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # Initialize world model for prediction
        self.world_model = WorldModelCNN(config).to(self.device)
        
        # Initialize temporal memory
        self.frame_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)
        self.visual_change_history = deque(maxlen=history_length)
        
        # Feature embedding and state density tracking
        self.feature_extractor = self.world_model.encoder
        self.density_estimator = DensityEstimator(
            feature_dim=512,  # Output size of feature extractor
            history_size=2000,
            device=self.device
        )
        
        # Temporal association memory
        self.association_memory = TemporalAssociationMemory(
            feature_dim=512,
            history_size=1000,
            device=self.device
        )
        
        # Intrinsic motivation components
        self.reward_history = []
        self.reward_scale = 1.0
        self.reward_shift = 0.0
        self.max_menu_penalty = -10.0
        self.max_history_length = history_length
        
        # Visual change analyzer for pattern recognition
        self.visual_change_analyzer = VisualChangeAnalyzer(memory_size=1000)
        
    def compute_reward(self, prev_frame: np.ndarray, action_idx: int, curr_frame: np.ndarray) -> float:
        """Compute reward based on visual changes without using domain knowledge.
        
        This method only looks at raw visual changes between frames and uses
        intrinsic motivation approaches without any game-specific knowledge.
        
        Args:
            prev_frame: Previous frame
            action_idx: Index of the action taken
            curr_frame: Current frame after action
            
        Returns:
            float: Computed reward
        """
        # Input validation
        if prev_frame is None or curr_frame is None:
            return 0.0
            
        try:
            # Ensure we're using the right device and dtype
            device = self.device
            dtype = self.dtype
            
            # Convert frames to tensors if they aren't already
            if not isinstance(prev_frame, torch.Tensor):
                prev_frame = torch.from_numpy(prev_frame).to(device=device, dtype=dtype)
            if not isinstance(curr_frame, torch.Tensor):
                curr_frame = torch.from_numpy(curr_frame).to(device=device, dtype=dtype)
                
            # Normalize if needed
            if prev_frame.max() > 1.0:
                prev_frame = prev_frame / 255.0
            if curr_frame.max() > 1.0:
                curr_frame = curr_frame / 255.0
                
            # Ensure correct shape
            if len(prev_frame.shape) == 3:
                prev_frame = prev_frame.unsqueeze(0)
            if len(curr_frame.shape) == 3:
                curr_frame = curr_frame.unsqueeze(0)
                
            # Compute reward components
            prediction_error_reward = self._prediction_error_reward(prev_frame, action_idx, curr_frame)
            visual_change_reward = self._visual_change_reward(prev_frame, curr_frame)
            density_reward = self._density_reward(curr_frame)
            association_reward = self._association_reward(prev_frame, action_idx, curr_frame)
            
            # Combine reward components
            total_reward = (
                0.4 * prediction_error_reward +
                0.3 * visual_change_reward +
                0.2 * density_reward +
                0.1 * association_reward
            )
            
            # Update internal state
            self._update_state(prev_frame, action_idx, curr_frame, total_reward)
            
            # Apply adaptive normalization
            normalized_reward = self._normalize_reward(total_reward)
            
            return normalized_reward
            
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            return 0.0
    
    def _prediction_error_reward(self, prev_frame: torch.Tensor, action_idx: int, curr_frame: torch.Tensor) -> float:
        """Compute reward based on world model prediction error.
        
        This rewards the agent for exploring states that are hard to predict,
        encouraging exploration of novel situations.
        
        Args:
            prev_frame: Previous frame
            action_idx: Action index
            curr_frame: Current frame
            
        Returns:
            float: Prediction error reward
        """
        try:
            # Predict next frame using world model
            predicted_frame = self.world_model.predict_next_frame(prev_frame, action_idx)
            
            # Calculate prediction error
            prediction_error = F.mse_loss(predicted_frame, curr_frame)
            
            # Scale prediction error to a reward
            # Higher error = higher reward (encourages exploration)
            # But not too high (clip to avoid pursuing completely random states)
            reward = torch.clamp(prediction_error, 0.0, 1.0).item()
            
            return reward
        except Exception as e:
            logger.error(f"Error computing prediction error reward: {e}")
            return 0.0
    
    def _visual_change_reward(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> float:
        """Compute reward based on visual change between frames.
        
        This rewards the agent for actions that cause visual changes,
        encouraging interaction with the environment.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            
        Returns:
            float: Visual change reward
        """
        try:
            # Convert to numpy for visual change calculation if needed
            if isinstance(prev_frame, torch.Tensor):
                prev_frame_np = prev_frame.squeeze().cpu().numpy()
            else:
                prev_frame_np = prev_frame
                
            if isinstance(curr_frame, torch.Tensor):
                curr_frame_np = curr_frame.squeeze().cpu().numpy()
            else:
                curr_frame_np = curr_frame
                
            # Calculate visual change using our analyzer
            diff_pattern = cv2.absdiff(
                prev_frame_np.transpose(1, 2, 0) if prev_frame_np.ndim == 3 else prev_frame_np, 
                curr_frame_np.transpose(1, 2, 0) if curr_frame_np.ndim == 3 else curr_frame_np
            )
            
            # Compute mean pixel change
            mean_diff = np.mean(diff_pattern)
            
            # Use visual change analyzer to predict outcome from this pattern
            predicted_outcome = self.visual_change_analyzer.predict_outcome(diff_pattern)
            
            # Update association memory
            self.visual_change_analyzer.update_association(diff_pattern, mean_diff)
            
            # Combine raw change and predicted outcome
            reward = 0.7 * mean_diff + 0.3 * predicted_outcome
            
            return float(reward)
        except Exception as e:
            logger.error(f"Error computing visual change reward: {e}")
            return 0.0
    
    def _density_reward(self, curr_frame: torch.Tensor) -> float:
        """Compute reward based on state density estimation.
        
        This rewards the agent for visiting rare states, encouraging exploration.
        
        Args:
            curr_frame: Current frame
            
        Returns:
            float: Density reward
        """
        try:
            # Extract features from frame
            with torch.no_grad():
                features = self.world_model.encode_frame(curr_frame)
                
            # Compute density (novelty) of this state
            density = self.density_estimator.compute_novelty(features)
            
            # Update density estimator
            self.density_estimator.update(features)
            
            # Lower density = higher reward (encourage exploration of rare states)
            reward = 1.0 - torch.clamp(density, 0.0, 1.0).item()
            
            return reward
        except Exception as e:
            logger.error(f"Error computing density reward: {e}")
            return 0.0
    
    def _association_reward(self, prev_frame: torch.Tensor, action_idx: int, curr_frame: torch.Tensor) -> float:
        """Compute reward based on temporal association memory.
        
        This rewards the agent for learning action-outcome associations.
        
        Args:
            prev_frame: Previous frame
            action_idx: Action index
            curr_frame: Current frame
            
        Returns:
            float: Association reward
        """
        try:
            # Extract features from frames
            with torch.no_grad():
                prev_features = self.world_model.encode_frame(prev_frame)
                curr_features = self.world_model.encode_frame(curr_frame)
                
            # Compute association score
            score = self.association_memory.compute_association(prev_features, action_idx, curr_features)
            
            # Update association memory
            self.association_memory.update(prev_features, action_idx, curr_features)
            
            return score.item()
        except Exception as e:
            logger.error(f"Error computing association reward: {e}")
            return 0.0
    
    def _update_state(self, prev_frame: torch.Tensor, action_idx: int, curr_frame: torch.Tensor, reward: float) -> None:
        """Update internal state with new experience.
        
        Args:
            prev_frame: Previous frame
            action_idx: Action index
            curr_frame: Current frame
            reward: Computed reward
        """
        # Store reward in history for normalization
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:  # Keep limited history
            self.reward_history.pop(0)
            
        # Update frame history
        self.frame_history.append(curr_frame.detach().cpu())
        if len(self.frame_history) > self.max_history_length:
            self.frame_history.pop(0)
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward based on recent history.
        
        Args:
            reward: Raw reward value
            
        Returns:
            float: Normalized reward
        """
        if not self.reward_history:
            return reward
            
        # Compute mean and std of recent rewards
        mean_reward = np.mean(self.reward_history)
        std_reward = np.std(self.reward_history) + 1e-8  # Avoid division by zero
        
        # Normalize reward (z-score normalization)
        normalized = (reward - mean_reward) / std_reward
        
        # Clip to avoid extreme values
        return float(np.clip(normalized, -5.0, 5.0))

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
        score, diff = ssim(previous_gray, current_gray, full=True, data_range=1.0)
        
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
    
    def evaluate_stability(self, current_frame: torch.Tensor, previous_frames: List[torch.Tensor]) -> float:
        """Evaluate stability and novelty of game state using unsupervised learning.
        
        This method uses learned feature representations to evaluate state novelty and stability.
        
        Args:
            current_frame: Current frame
            previous_frames: Previous frames
            
        Returns:
            float: Stability/novelty score
        """
        if not previous_frames:
            return 0.0
            
        try:
            # Extract features from current and previous frames
            with torch.no_grad():
                current_features = self.world_model.encode_frame(current_frame)
                prev_features = [self.world_model.encode_frame(f) for f in previous_frames[-5:]]
            
            # Compute novelty (distance from previous states)
            novelty_scores = []
            for prev_feat in prev_features:
                dist = F.mse_loss(current_features, prev_feat)
                novelty_scores.append(dist.item())
            
            # Average novelty (higher = more different from previous states)
            avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
            
            # Scale to reasonable reward range (-0.5 to 0.5)
            # Some novelty is good (exploration), too much is bad (random/unstable)
            if avg_novelty < 0.01:  # Too similar = stuck
                score = -0.2  # Slight negative
            elif avg_novelty < 0.1:  # Good range of novelty
                score = 0.3  # Positive reward
            else:  # Too much change = unstable
                score = -0.4  # More negative
                
            return score
            
        except Exception as e:
            logger.error(f"Error computing stability: {e}")
            return 0.0 