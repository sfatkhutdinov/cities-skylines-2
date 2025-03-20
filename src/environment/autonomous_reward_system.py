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
from scipy.spatial import KDTree
import os
from src.environment.visual_change_analyzer import VisualChangeAnalyzer
from src.environment.causal_learning import CausalUnderstandingModule, ActionSequenceMemory

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
        # Alias state_memory as memory for compatibility
        self.memory = self.state_memory
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
            
        # Update the memory alias
        self.memory = self.state_memory
            
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
        
    def save_state(self, path: str) -> None:
        """Save the density estimator state to disk.
        
        Args:
            path: Path to save the state
        """
        state_dict = {
            'feature_dim': self.feature_dim,
            'history_size': self.history_size,
            'state_memory': [state.cpu() for state in self.state_memory],
            'mean_vector': self.mean_vector.cpu() if self.mean_vector is not None else None,
            'std_vector': self.std_vector.cpu() if self.std_vector is not None else None
        }
        torch.save(state_dict, path)
        logger.info(f"Saved density estimator state with {len(self.state_memory)} memory entries")
        
    def load_state(self, path: str) -> None:
        """Load the density estimator state from disk.
        
        Args:
            path: Path to load the state from
        """
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.feature_dim = state_dict['feature_dim']
            self.history_size = state_dict['history_size']
            
            # Load state memory and move to device if needed
            if self.device is not None:
                self.state_memory = [state.to(self.device) for state in state_dict['state_memory']]
            else:
                self.state_memory = state_dict['state_memory']
                
            # Update the memory alias
            self.memory = self.state_memory
                
            # Load other components
            if state_dict['mean_vector'] is not None:
                self.mean_vector = state_dict['mean_vector']
                if self.device is not None:
                    self.mean_vector = self.mean_vector.to(self.device)
                    
            if state_dict['std_vector'] is not None:
                self.std_vector = state_dict['std_vector']
                if self.device is not None:
                    self.std_vector = self.std_vector.to(self.device)
                    
            logger.info(f"Loaded density estimator state with {len(self.state_memory)} memory entries")
        except Exception as e:
            logger.error(f"Error loading density estimator state: {e}")
            # Initialize with empty memory
            self.state_memory = []
            self.memory = self.state_memory
            self.mean_vector = None
            self.std_vector = None


class TemporalAssociationMemory:
    """Memory for learning associations between features and outcomes over time."""
    
    def __init__(self, feature_dim=512, history_size=1000, device=None):
        """Initialize temporal association memory.
        
        Args:
            feature_dim (int): Dimension of feature vectors
            history_size (int): Maximum number of memory entries
            device: The device to use
        """
        self.feature_dim = feature_dim
        self.history_size = history_size
        self.device = device
        
        # Memory storage
        self.feature_memory = []
        self.outcome_memory = []
        
        # Aliases for compatibility with visual change analyzer interface
        self.pattern_memory = self.feature_memory
        self.outcomes_memory = self.outcome_memory
        
        # For optimized retrieval
        self.kdtree = None
        self.last_update_size = 0
        
    def store(self, feature: torch.Tensor, outcome: float) -> None:
        """Store feature-outcome pair in memory.
        
        Args:
            feature (torch.Tensor): Feature embedding
            outcome (float): Associated outcome (reward)
        """
        # Ensure feature is detached from computation graph and on CPU for storage
        feature_cpu = feature.detach().cpu()
        
        # Store feature and outcome
        self.feature_memory.append(feature_cpu)
        self.outcome_memory.append(outcome)
        
        # Update aliases
        self.pattern_memory = self.feature_memory
        self.outcomes_memory = self.outcome_memory
        
        # Keep memory within size limit
        if len(self.feature_memory) > self.history_size:
            self.feature_memory.pop(0)
            self.outcome_memory.pop(0)
            
            # Update aliases again after removal
            self.pattern_memory = self.feature_memory
            self.outcomes_memory = self.outcome_memory
            
        # Flag that we need to rebuild kd-tree for efficient retrieval
        self.kdtree = None
        
    def query(self, feature: torch.Tensor, k: int = 5) -> float:
        """Query memory for expected outcome given feature.
        
        Args:
            feature (torch.Tensor): Feature to query
            k (int): Number of nearest neighbors to consider
            
        Returns:
            float: Expected outcome (weighted average of k nearest neighbors)
        """
        if not self.feature_memory:
            return 0.0
            
        # Ensure feature is on CPU
        feature_np = feature.detach().cpu().numpy().flatten()
        
        # Check if we need to rebuild KD-tree
        if self.kdtree is None or len(self.feature_memory) != self.last_update_size:
            # Convert stored features to numpy array
            try:
                feature_array = np.vstack([f.numpy().flatten() for f in self.feature_memory])
                self.kdtree = KDTree(feature_array)
                self.last_update_size = len(self.feature_memory)
            except:
                # Fall back to linear search if KD-tree fails
                return self._linear_search(feature, k)
        
        # Find k nearest neighbors
        try:
            # Query KD-tree
            distances, indices = self.kdtree.query(feature_np.reshape(1, -1), k=min(k, len(self.feature_memory)))
            distances = distances[0]
            indices = indices[0]
            
            # Calculate weights based on inverse distance
            weights = 1.0 / (distances + 1e-6)
            total_weight = np.sum(weights)
            
            if total_weight == 0:
                return 0.0
                
            # Calculate weighted average of outcomes
            weighted_sum = sum(weights[i] * self.outcome_memory[indices[i]] for i in range(len(indices)))
            predicted_outcome = weighted_sum / total_weight
            
            return predicted_outcome
        except:
            # Fall back to linear search if KD-tree query fails
            return self._linear_search(feature, k)
            
    def _linear_search(self, feature: torch.Tensor, k: int) -> float:
        """Linear search fallback for when KD-tree is unavailable.
        
        Args:
            feature (torch.Tensor): Query feature
            k (int): Number of nearest neighbors
            
        Returns:
            float: Predicted outcome
        """
        # Ensure feature is on CPU
        feature_cpu = feature.detach().cpu()
        
        # Calculate distances to all stored features
        distances = []
        for stored_feature in self.feature_memory:
            distance = torch.nn.functional.mse_loss(feature_cpu, stored_feature).item()
            distances.append(distance)
            
        if not distances:
            return 0.0
            
        # Find k nearest neighbors
        k = min(k, len(distances))
        indices = np.argsort(distances)[:k]
        
        # Calculate weights based on inverse distance
        weights = [1.0 / (distances[i] + 1e-6) for i in indices]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 0.0
            
        # Calculate weighted average of outcomes
        weighted_sum = sum(weights[i] * self.outcome_memory[indices[i]] for i in range(len(indices)))
        predicted_outcome = weighted_sum / total_weight
        
        return predicted_outcome
        
    def save_state(self, path: str) -> None:
        """Save the temporal association memory state to disk.
        
        Args:
            path: Path to save the state
        """
        # Store all the necessary components
        state_dict = {
            'feature_dim': self.feature_dim,
            'history_size': self.history_size,
            'feature_memory': self.feature_memory,
            'outcome_memory': self.outcome_memory
        }
        torch.save(state_dict, path)
        logger.info(f"Saved temporal association memory with {len(self.feature_memory)} entries")
        
    def load_state(self, path: str) -> None:
        """Load the temporal association memory state from disk.
        
        Args:
            path: Path to load the state from
        """
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.feature_dim = state_dict['feature_dim']
            self.history_size = state_dict['history_size']
            self.feature_memory = state_dict['feature_memory']
            self.outcome_memory = state_dict['outcome_memory']
            
            # Update aliases
            self.pattern_memory = self.feature_memory
            self.outcomes_memory = self.outcome_memory
            
            # Reset KD-tree to be rebuilt on next query
            self.kdtree = None
            self.last_update_size = 0
            
            logger.info(f"Loaded temporal association memory with {len(self.feature_memory)} entries")
        except Exception as e:
            logger.error(f"Failed to load temporal association memory: {str(e)}")
            # Initialize fresh if loading fails
            self.feature_memory = []
            self.outcome_memory = []
            
            # Update aliases
            self.pattern_memory = self.feature_memory
            self.outcomes_memory = self.outcome_memory
            
            self.kdtree = None
            self.last_update_size = 0


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
        
        # Initialize enhanced causal understanding module
        self.causal_module = CausalUnderstandingModule(
            config=config,
            feature_dim=512,  # Match the feature extractor dimension
            device=self.device
        )
        
        # Feature history for causal analysis
        self.feature_history = deque(maxlen=history_length)
        self.outcome_history = deque(maxlen=history_length)
        
        # Causal rewards weighting
        self.use_causal_prediction = False  # Start with False, enable after collecting data
        self.causal_weight = 0.2  # Weight for causal component in reward calculation
        self.training_steps = 0  # Track how many steps we've trained
    
    def save_state(self, path_prefix: str):
        """Save the state of the reward system for later resumption.
        
        Args:
            path_prefix (str): Prefix path to save the state files
        """
        try:
            # Save density estimator state if available
            if hasattr(self, 'density_estimator') and self.density_estimator is not None:
                density_path = f"{path_prefix}_density.pkl"
                self.density_estimator.save_state(density_path)
                logger.info(f"Saved density estimator state with {len(self.density_estimator.state_memory)} memory entries")
                
            # Save temporal association memory if available
            if hasattr(self, 'association_memory') and self.association_memory is not None:
                temp_assoc_path = f"{path_prefix}_temp_assoc.pkl"
                self.association_memory.save_state(temp_assoc_path)
                logger.info(f"Saved temporal association memory with {len(self.association_memory.feature_memory)} entries")
                
            # Save visual change analyzer if available
            if hasattr(self, 'visual_change_analyzer') and self.visual_change_analyzer is not None:
                visual_change_path = f"{path_prefix}_visual_change.pkl"
                self.visual_change_analyzer.save_state(visual_change_path)
                logger.info(f"Saved visual change analyzer state with {len(self.visual_change_analyzer.pattern_memory)} patterns")
                
        except Exception as e:
            logger.error(f"Failed to save reward system state: {str(e)}")
        
    def load_state(self, path_prefix: str):
        """Load the state of the reward system from saved files.
        
        Args:
            path_prefix (str): Prefix path where state files are saved
        """
        try:
            # Load density estimator state if available
            density_path = f"{path_prefix}_density.pkl"
            if os.path.exists(density_path) and hasattr(self, 'density_estimator') and self.density_estimator is not None:
                self.density_estimator.load_state(density_path)
                logger.info(f"Loaded density estimator state with {len(self.density_estimator.state_memory)} memory entries")
                
            # Load temporal association memory if available
            temp_assoc_path = f"{path_prefix}_temp_assoc.pkl"
            if os.path.exists(temp_assoc_path) and hasattr(self, 'association_memory') and self.association_memory is not None:
                self.association_memory.load_state(temp_assoc_path)
                logger.info(f"Loaded temporal association memory with {len(self.association_memory.feature_memory)} entries")
                
            # Load visual change analyzer if available
            visual_change_path = f"{path_prefix}_visual_change.pkl"
            if os.path.exists(visual_change_path) and hasattr(self, 'visual_change_analyzer') and self.visual_change_analyzer is not None:
                self.visual_change_analyzer.load_state(visual_change_path)
                logger.info(f"Loaded visual change analyzer state with {len(self.visual_change_analyzer.pattern_memory)} patterns")
                
        except Exception as e:
            logger.error(f"Failed to load reward system state: {str(e)}")
            # Initialize with default values if loading fails
            self._initialize_default_values()
    
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
            
            # Extract features for causal analysis
            with torch.no_grad():
                prev_features = self.world_model.encode_frame(prev_frame)
                curr_features = self.world_model.encode_frame(curr_frame)
                
            # Compute causal reward if enabled
            causal_reward = 0.0
            if self.use_causal_prediction and len(self.action_history) > 5:
                # Query expected outcome from causal module
                expected_outcome, confidence = self.causal_module.predict_action_outcome(
                    prev_features, action_idx
                )
                
                # Calculate the accuracy of causal prediction
                actual_outcome = (prediction_error_reward + visual_change_reward + density_reward + association_reward) / 4.0
                prediction_error = abs(expected_outcome - actual_outcome)
                
                # Reward agent more when outcomes align with causal predictions (encourage predictable behavior)
                causal_reward = 0.5 * (1.0 - min(1.0, prediction_error)) * confidence
                
                # Log causal prediction accuracy
                logger.debug(f"Causal prediction: expected={expected_outcome:.3f}, actual={actual_outcome:.3f}, confidence={confidence:.3f}")
            
            # Combine reward components
            total_reward = (
                0.3 * prediction_error_reward +
                0.25 * visual_change_reward +
                0.15 * density_reward +
                0.1 * association_reward +
                self.causal_weight * causal_reward
            )
            
            # Update internal state, including causal module
            self._update_state(prev_frame, action_idx, curr_frame, total_reward, prev_features, curr_features)
            
            # Apply adaptive normalization
            normalized_reward = self._normalize_reward(total_reward)
            
            # After 1000 steps, start using causal predictions
            self.training_steps += 1
            if self.training_steps == 1000:
                self.use_causal_prediction = True
                logger.info("Enabled causal predictions for reward calculation")
                
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
            # Handle detached tensors
            if isinstance(prev_frame, torch.Tensor) and prev_frame.requires_grad:
                prev_frame = prev_frame.detach()
            if isinstance(curr_frame, torch.Tensor) and curr_frame.requires_grad:
                curr_frame = curr_frame.detach()
            
            # Predict next frame using world model - ensure we pass a list of frames
            predicted_frame = self.world_model.predict_next_frame([prev_frame], action_idx)
            
            # Calculate prediction error - ensure we're working with valid tensors
            # Explicitly check tensor shapes to avoid boolean tensor comparison
            if predicted_frame.shape != curr_frame.shape:
                # Resize to match if needed
                if len(curr_frame.shape) > len(predicted_frame.shape):
                    curr_frame = curr_frame.squeeze(0)
                elif len(predicted_frame.shape) > len(curr_frame.shape):
                    predicted_frame = predicted_frame.squeeze(0)
                    
                # If shapes still don't match, resize using interpolate
                if tuple(predicted_frame.shape) != tuple(curr_frame.shape):
                    predicted_frame = F.interpolate(
                        predicted_frame.unsqueeze(0),
                        size=curr_frame.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
            
            # Calculate MSE error
            prediction_error = F.mse_loss(predicted_frame, curr_frame)
            
            # Convert scalar tensor to float explicitly
            if isinstance(prediction_error, torch.Tensor):
                # Handle tensors with multiple values by taking the mean
                if prediction_error.numel() > 1:
                    prediction_error = prediction_error.mean()
                
                # Detach and convert to float
                if prediction_error.requires_grad:
                    prediction_error = prediction_error.detach()
                prediction_error = float(prediction_error.cpu().item())
            
            # Scale prediction error to a reward
            # Higher error = higher reward (encourages exploration)
            # But not too high (clip to avoid pursuing completely random states)
            reward = min(max(prediction_error, 0.0), 1.0)
            
            return float(reward)
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
                # Use detach to handle requires_grad tensors
                prev_frame_np = prev_frame.detach().squeeze().cpu().numpy()
            else:
                prev_frame_np = prev_frame
                
            if isinstance(curr_frame, torch.Tensor):
                # Use detach to handle requires_grad tensors
                curr_frame_np = curr_frame.detach().squeeze().cpu().numpy()
            else:
                curr_frame_np = curr_frame
                
            # Skip processing if either frame is None or empty
            if prev_frame_np is None or curr_frame_np is None:
                return 0.0
                
            if prev_frame_np.size == 0 or curr_frame_np.size == 0:
                return 0.0
                
            # Make sure shapes match for the diff calculation
            # If prev_frame has shape [C, H, W], transpose to [H, W, C]
            if prev_frame_np.ndim == 3 and prev_frame_np.shape[0] <= 3:
                prev_frame_np = prev_frame_np.transpose(1, 2, 0)
            # Same for curr_frame
            if curr_frame_np.ndim == 3 and curr_frame_np.shape[0] <= 3:
                curr_frame_np = curr_frame_np.transpose(1, 2, 0)
            
            # Ensure both frames have the same shape
            if prev_frame_np.shape != curr_frame_np.shape:
                # Resize to match
                if prev_frame_np.ndim == 3 and curr_frame_np.ndim == 3:
                    # Both are 3D, resize to match curr_frame
                    h, w = curr_frame_np.shape[0], curr_frame_np.shape[1]
                    prev_frame_np = cv2.resize(prev_frame_np, (w, h), interpolation=cv2.INTER_LINEAR)
                elif prev_frame_np.ndim == 2 and curr_frame_np.ndim == 2:
                    # Both are 2D, resize to match curr_frame
                    h, w = curr_frame_np.shape[0], curr_frame_np.shape[1]
                    prev_frame_np = cv2.resize(prev_frame_np, (w, h), interpolation=cv2.INTER_LINEAR)
                elif prev_frame_np.ndim != curr_frame_np.ndim:
                    # Convert one to match the other's dimensionality
                    if prev_frame_np.ndim == 2 and curr_frame_np.ndim == 3:
                        prev_frame_np = cv2.cvtColor(prev_frame_np, cv2.COLOR_GRAY2BGR)
                        h, w = curr_frame_np.shape[0], curr_frame_np.shape[1]
                        prev_frame_np = cv2.resize(prev_frame_np, (w, h), interpolation=cv2.INTER_LINEAR)
                    elif prev_frame_np.ndim == 3 and curr_frame_np.ndim == 2:
                        prev_frame_np = cv2.cvtColor(prev_frame_np, cv2.COLOR_BGR2GRAY)
                        h, w = curr_frame_np.shape[0], curr_frame_np.shape[1]
                        prev_frame_np = cv2.resize(prev_frame_np, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Reshape diff_pattern to 2D if needed
            if hasattr(self.visual_change_analyzer, 'input_shape'):
                # Get expected shape from the analyzer
                expected_shape = self.visual_change_analyzer.input_shape
                
                # Calculate visual change
                diff_pattern = cv2.absdiff(prev_frame_np, curr_frame_np)
                
                # Resize diff_pattern to match expected input shape for the analyzer
                if diff_pattern.shape != expected_shape and len(expected_shape) == 2:
                    # Convert to grayscale if needed
                    if diff_pattern.ndim == 3:
                        diff_pattern = cv2.cvtColor(diff_pattern, cv2.COLOR_BGR2GRAY)
                    # Resize to match expected dimensions
                    diff_pattern = cv2.resize(diff_pattern, (expected_shape[1], expected_shape[0]))
            else:
                # Just calculate the diff without reshaping
                diff_pattern = cv2.absdiff(prev_frame_np, curr_frame_np)
            
            # Compute mean pixel change
            mean_diff = float(np.mean(diff_pattern))
            
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
            # Skip if frame is invalid
            if curr_frame is None:
                return 0.0
                
            # Handle detached tensors
            if isinstance(curr_frame, torch.Tensor) and curr_frame.requires_grad:
                curr_frame = curr_frame.detach()
                
            # Extract features from frame
            with torch.no_grad():
                features = self.world_model.encode_frame(curr_frame)
                
                # Ensure features is flattened to 1D for density estimation
                if features.dim() > 1:
                    # If features has shape [batch, features], take the first batch
                    if features.dim() == 2 and features.size(0) > 1:
                        features = features[0]
                    # Flatten to 1D
                    features = features.flatten()
                
            # Compute density (novelty) of this state
            density = self.density_estimator.compute_novelty(features)
            
            # Update density estimator
            self.density_estimator.update(features)
            
            # Make sure density is a scalar
            if isinstance(density, torch.Tensor):
                # If tensor has multiple elements, take mean
                if density.numel() > 1:
                    density = density.mean().item()
                else:
                    density = density.item()
            
            # Ensure density is a valid float
            density = float(density)
            
            # Clamp density to [0, 1] range
            density = max(0.0, min(1.0, density))
            
            # Lower density = higher reward (encourage exploration of rare states)
            reward = 1.0 - density
            
            return float(reward)
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
            # Skip if frames are invalid
            if prev_frame is None or curr_frame is None:
                return 0.0
                
            # Handle detached tensors
            if isinstance(prev_frame, torch.Tensor) and prev_frame.requires_grad:
                prev_frame = prev_frame.detach()
            if isinstance(curr_frame, torch.Tensor) and curr_frame.requires_grad:
                curr_frame = curr_frame.detach()
                
            # Extract features from frames
            with torch.no_grad():
                prev_features = self.world_model.encode_frame(prev_frame)
                curr_features = self.world_model.encode_frame(curr_frame)
                
                # Ensure features are flattened
                if prev_features.dim() > 1:
                    if prev_features.dim() == 2 and prev_features.size(0) > 1:
                        prev_features = prev_features[0]
                    prev_features = prev_features.flatten()
                
                if curr_features.dim() > 1:
                    if curr_features.dim() == 2 and curr_features.size(0) > 1:
                        curr_features = curr_features[0]
                    curr_features = curr_features.flatten()
                
            # Compute association score
            score = self.association_memory.query(prev_features, k=5)
            
            # Update association memory
            self.association_memory.store(prev_features, float(score))
            
            # Ensure score is a scalar
            if isinstance(score, torch.Tensor):
                # If tensor has multiple elements, take mean
                if score.numel() > 1:
                    score = score.mean().item()
                else:
                    score = score.item()
                    
            return float(score)
        except Exception as e:
            logger.error(f"Error computing association reward: {e}")
            return 0.0
    
    def _update_state(self, prev_frame: torch.Tensor, action_idx: int, curr_frame: torch.Tensor, reward: float, 
                     prev_features: torch.Tensor = None, curr_features: torch.Tensor = None) -> None:
        """Update internal state with new experience.
        
        Args:
            prev_frame: Previous frame
            action_idx: Action index
            curr_frame: Current frame
            reward: Computed reward
            prev_features: Feature embedding of previous frame (optional)
            curr_features: Feature embedding of current frame (optional)
        """
        try:
            # Skip if frames are invalid
            if curr_frame is None:
                return
                
            # Store reward in history for normalization
            self.reward_history.append(float(reward))
            if len(self.reward_history) > 100:  # Keep limited history
                self.reward_history.pop(0)
                
            # Update frame history - ensure we detach and use CPU tensors to avoid memory issues
            if isinstance(curr_frame, torch.Tensor):
                # Force detach to prevent gradient leakage
                self.frame_history.append(curr_frame.detach().cpu())
            else:
                self.frame_history.append(curr_frame)
                
            # Update action history
            self.action_history.append(action_idx)
            
            # Update feature history for causal analysis
            if curr_features is not None:
                self.feature_history.append(curr_features.detach())
                self.outcome_history.append(reward)
                
                # Update causal understanding module
                self.causal_module.update(curr_features, action_idx, reward)
                
            # Maintain limited history
            if len(self.frame_history) > self.max_history_length:
                self.frame_history.pop(0)
                
            if len(self.action_history) > self.max_history_length:
                self.action_history.pop(0)
                
        except Exception as e:
            logger.error(f"Error updating agent state: {e}")
            # Continue without updating state
    
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

    def _initialize_default_values(self):
        """Initialize default values if state loading fails."""
        # Initialize memory components with default values
        if hasattr(self, 'density_estimator'):
            self.density_estimator.state_memory = []
            self.density_estimator.distances = []
            
        if hasattr(self, 'association_memory'):
            self.association_memory.feature_memory = []
            self.association_memory.outcomes_memory = []
            
        if hasattr(self, 'visual_change_analyzer'):
            self.visual_change_analyzer.pattern_memory = []
            self.visual_change_analyzer.outcomes_memory = []
            
        # Reset history
        self.reward_history = deque(maxlen=self.max_history_length)
        
        logger.info("Initialized reward system with default values")

    def get_action_recommendation(self, curr_frame: torch.Tensor, available_actions: List[int]) -> Tuple[int, float]:
        """Get a recommended action based on causal understanding.
        
        Args:
            curr_frame: Current frame
            available_actions: List of available action indices
            
        Returns:
            Tuple[int, float]: Recommended action and confidence
        """
        if not available_actions:
            return 0, 0.0
            
        try:
            # Extract features from current frame
            with torch.no_grad():
                curr_features = self.world_model.encode_frame(curr_frame)
                
            # Get recommendation from causal module
            recommended_action = self.causal_module.get_best_action(curr_features, available_actions)
            
            # Get confidence
            confidence = 0.0
            if recommended_action is not None:
                expected_outcome, conf = self.causal_module.predict_action_outcome(curr_features, recommended_action)
                confidence = conf
                
            return recommended_action, confidence
            
        except Exception as e:
            logger.error(f"Error getting action recommendation: {e}")
            return available_actions[0], 0.0
            
    def analyze_causality(self) -> Dict[str, Any]:
        """Analyze causal relationships in recent history.
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Collect data from history
            features = list(self.feature_history)
            actions = list(self.action_history)
            outcomes = list(self.outcome_history)
            
            # Get current state if available
            current_state = self.feature_history[-1] if self.feature_history else None
            
            # Run analysis
            if len(features) >= 5 and len(actions) >= 5 and current_state is not None:
                analysis = self.causal_module.enhance_temporal_causality(
                    features, actions, outcomes, current_state
                )
                return analysis
            else:
                return {"causal_strength": 0.0, "significant_actions": []}
                
        except Exception as e:
            logger.error(f"Error analyzing causality: {e}")
            return {"causal_strength": 0.0, "significant_actions": [], "error": str(e)} 