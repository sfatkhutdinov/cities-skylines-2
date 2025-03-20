"""
World model for Cities: Skylines 2 environment.

This module provides a predictive world model that learns to predict
state transitions based on actions, supporting the reward system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)

class WorldModelCNN(nn.Module):
    """Neural network model that predicts next frame and encodes observations."""
    
    def __init__(self, 
                 input_channels: int = 3,
                 embedding_dim: int = 512,
                 action_dim: int = 12,
                 device: Optional[torch.device] = None):
        """Initialize world model CNN.
        
        Args:
            input_channels: Number of input channels (3 for RGB)
            embedding_dim: Dimension of encoded state representation
            action_dim: Dimension of action space
            device: Compute device to use
        """
        super(WorldModelCNN, self).__init__()
        
        self.input_channels = input_channels
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define encoder network (frame -> embedding)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embedding_dim),  # Assuming 84x84 input
            nn.ReLU()
        )
        
        # Action embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # State transition model (embedding + action -> next embedding)
        self.transition = nn.Sequential(
            nn.Linear(embedding_dim + 64, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Prediction head for next frame
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=8, stride=4),
            nn.Sigmoid()  # Output normalized [0,1] pixel values
        )
        
        # Move model to device
        self.to(self.device)
        
    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a frame into a latent embedding.
        
        Args:
            frame: Input frame [B, C, H, W]
            
        Returns:
            torch.Tensor: Encoded state embedding
        """
        # Ensure frame is on the correct device
        frame = frame.to(self.device)
        
        # Normalize pixel values to [0, 1]
        if frame.max() > 1.0:
            frame = frame / 255.0
            
        # Encode
        embedding = self.encoder(frame)
        return embedding
        
    def predict_next(self, 
                    embedding: torch.Tensor, 
                    action: torch.Tensor) -> torch.Tensor:
        """Predict next state embedding based on current state and action.
        
        Args:
            embedding: Current state embedding
            action: Action tensor
            
        Returns:
            torch.Tensor: Predicted next state embedding
        """
        # Ensure inputs are on the correct device
        embedding = embedding.to(self.device)
        action = action.to(self.device)
        
        # Embed action
        action_emb = self.action_embedding(action)
        
        # Concatenate state and action
        combined = torch.cat([embedding, action_emb], dim=-1)
        
        # Predict next state
        next_embedding = self.transition(combined)
        
        return next_embedding
        
    def decode(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode embedding back to frame.
        
        Args:
            embedding: State embedding
            
        Returns:
            torch.Tensor: Reconstructed frame
        """
        # Ensure embedding is on the correct device
        embedding = embedding.to(self.device)
        
        # Decode
        frame = self.decoder(embedding)
        
        return frame
        
    def forward(self, 
               frame: torch.Tensor, 
               action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: predict next frame and encode current.
        
        Args:
            frame: Current frame
            action: Action tensor
            
        Returns:
            Tuple: (predicted next frame, encoded state)
        """
        # Encode current frame
        embedding = self.encode(frame)
        
        # Predict next state
        next_embedding = self.predict_next(embedding, action)
        
        # Decode next frame
        predicted_next_frame = self.decode(next_embedding)
        
        return predicted_next_frame, embedding
        
    def compute_prediction_error(self, 
                               predicted_frame: torch.Tensor, 
                               actual_frame: torch.Tensor) -> torch.Tensor:
        """Compute prediction error between predicted and actual next frames.
        
        Args:
            predicted_frame: Predicted next frame
            actual_frame: Actual observed next frame
            
        Returns:
            torch.Tensor: Mean squared error
        """
        # Ensure frames are on the correct device
        predicted_frame = predicted_frame.to(self.device)
        actual_frame = actual_frame.to(self.device)
        
        # Normalize actual frame if needed
        if actual_frame.max() > 1.0:
            actual_frame = actual_frame / 255.0
            
        # Compute MSE
        mse = F.mse_loss(predicted_frame, actual_frame)
        
        return mse
        
    def update(self, 
              current_frame: torch.Tensor,
              action: torch.Tensor,
              next_frame: torch.Tensor,
              optimizer: torch.optim.Optimizer,
              lambda_reconstruction: float = 1.0) -> Dict[str, float]:
        """Update model based on observed transition.
        
        Args:
            current_frame: Current frame
            action: Action taken
            next_frame: Next frame observed
            optimizer: Optimizer to use
            lambda_reconstruction: Weight for reconstruction loss
            
        Returns:
            Dict: Loss metrics
        """
        # Ensure inputs are on the correct device
        current_frame = current_frame.to(self.device)
        action = action.to(self.device)
        next_frame = next_frame.to(self.device)
        
        # Normalize frames if needed
        if current_frame.max() > 1.0:
            current_frame = current_frame / 255.0
        if next_frame.max() > 1.0:
            next_frame = next_frame / 255.0
            
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predicted_next, embedding = self(current_frame, action)
        
        # Compute losses
        reconstruction_loss = F.mse_loss(predicted_next, next_frame)
        
        # Combined loss
        loss = lambda_reconstruction * reconstruction_loss
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Return metrics
        return {
            'total_loss': loss.item(),
            'reconstruction_loss': reconstruction_loss.item()
        }
        
    def save_model(self, path: str) -> None:
        """Save model to disk.
        
        Args:
            path: Path to save model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': {
                    'input_channels': self.input_channels,
                    'embedding_dim': self.embedding_dim,
                    'action_dim': self.action_dim
                }
            }, path)
            
            logger.info(f"Saved world model to {path}")
        except Exception as e:
            logger.error(f"Error saving world model: {e}")
            
    def load_model(self, path: str) -> None:
        """Load model from disk.
        
        Args:
            path: Path to load model from
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"World model file not found: {path}")
                return
                
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Check if config matches
            config = checkpoint.get('config', {})
            if (config.get('input_channels') != self.input_channels or
                config.get('embedding_dim') != self.embedding_dim or
                config.get('action_dim') != self.action_dim):
                logger.warning("Model configuration mismatch. Attempting to load anyway.")
                
            # Load state dict
            self.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Loaded world model from {path}")
        except Exception as e:
            logger.error(f"Error loading world model: {e}")


class ExperienceBuffer:
    """Buffer for storing transition experiences for world model training."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize experience buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def add(self, 
           current_frame: np.ndarray,
           action: np.ndarray,
           next_frame: np.ndarray,
           reward: float) -> None:
        """Add transition to buffer.
        
        Args:
            current_frame: Current frame
            action: Action taken
            next_frame: Next frame observed
            reward: Reward received
        """
        # Store transition
        self.buffer.append((current_frame, action, next_frame, reward))
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample batch of transitions from buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple: (current_frames, actions, next_frames, rewards)
        """
        if len(self.buffer) < batch_size:
            # If buffer doesn't have enough samples, return all
            batch_size = len(self.buffer)
            
        # Sample random indices
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        
        # Extract transitions
        transitions = [self.buffer[i] for i in indices]
        
        # Unpack transitions
        current_frames, actions, next_frames, rewards = zip(*transitions)
        
        # Convert to numpy arrays
        current_frames = np.array(current_frames)
        actions = np.array(actions)
        next_frames = np.array(next_frames)
        rewards = np.array(rewards)
        
        return current_frames, actions, next_frames, rewards
        
    def __len__(self) -> int:
        """Get current buffer size.
        
        Returns:
            int: Number of transitions in buffer
        """
        return len(self.buffer)
        
    def save_buffer(self, path: str) -> None:
        """Save buffer to disk.
        
        Args:
            path: Path to save buffer
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save buffer using pickle
            with open(path, 'wb') as f:
                pickle.dump({
                    'capacity': self.capacity,
                    'buffer': list(self.buffer)
                }, f)
                
            logger.info(f"Saved experience buffer with {len(self.buffer)} transitions")
        except Exception as e:
            logger.error(f"Error saving experience buffer: {e}")
            
    def load_buffer(self, path: str) -> None:
        """Load buffer from disk.
        
        Args:
            path: Path to load buffer from
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Experience buffer file not found: {path}")
                return
                
            # Load buffer using pickle
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
            # Restore data
            self.capacity = data['capacity']
            self.buffer = deque(data['buffer'], maxlen=self.capacity)
            
            logger.info(f"Loaded experience buffer with {len(self.buffer)} transitions")
        except Exception as e:
            logger.error(f"Error loading experience buffer: {e}")


class WorldModelTrainer:
    """Trainer for the world model."""
    
    def __init__(self, 
                world_model: WorldModelCNN,
                experience_buffer: ExperienceBuffer,
                learning_rate: float = 0.001,
                batch_size: int = 64,
                device: Optional[torch.device] = None):
        """Initialize world model trainer.
        
        Args:
            world_model: World model to train
            experience_buffer: Buffer of transitions
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            device: Compute device to use
        """
        self.world_model = world_model
        self.experience_buffer = experience_buffer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device if device is not None else world_model.device
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(world_model.parameters(), lr=learning_rate)
        
        # Training metrics
        self.train_steps = 0
        self.total_loss_history = []
        self.reconstruction_loss_history = []
        
    def train_step(self) -> Dict[str, float]:
        """Perform one training step.
        
        Returns:
            Dict: Loss metrics
        """
        # Check if buffer has enough samples
        if len(self.experience_buffer) < self.batch_size:
            logger.warning("Not enough samples in buffer for training")
            return {
                'total_loss': 0.0,
                'reconstruction_loss': 0.0
            }
            
        # Sample batch
        current_frames, actions, next_frames, _ = self.experience_buffer.sample(self.batch_size)
        
        # Convert to tensors
        current_frames = torch.tensor(current_frames, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        next_frames = torch.tensor(next_frames, dtype=torch.float32).to(self.device)
        
        # Update model
        loss_metrics = self.world_model.update(
            current_frames, actions, next_frames, self.optimizer)
            
        # Record metrics
        self.train_steps += 1
        self.total_loss_history.append(loss_metrics['total_loss'])
        self.reconstruction_loss_history.append(loss_metrics['reconstruction_loss'])
        
        # Truncate history to last 1000 steps
        if len(self.total_loss_history) > 1000:
            self.total_loss_history = self.total_loss_history[-1000:]
            self.reconstruction_loss_history = self.reconstruction_loss_history[-1000:]
            
        return loss_metrics
        
    def train_batch(self, num_steps: int = 10) -> Dict[str, float]:
        """Train model for multiple steps.
        
        Args:
            num_steps: Number of training steps
            
        Returns:
            Dict: Average loss metrics
        """
        total_metrics = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0
        }
        
        for _ in range(num_steps):
            metrics = self.train_step()
            
            for key in total_metrics:
                total_metrics[key] += metrics[key]
                
        # Compute averages
        for key in total_metrics:
            total_metrics[key] /= num_steps
            
        logger.info(f"Trained world model for {num_steps} steps. "
                   f"Avg loss: {total_metrics['total_loss']:.6f}")
        
        return total_metrics
        
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get training metrics summary.
        
        Returns:
            Dict: Summary metrics
        """
        if not self.total_loss_history:
            return {
                'train_steps': 0,
                'avg_total_loss': 0.0,
                'avg_reconstruction_loss': 0.0,
                'min_total_loss': 0.0,
                'max_total_loss': 0.0
            }
            
        metrics = {
            'train_steps': self.train_steps,
            'avg_total_loss': np.mean(self.total_loss_history),
            'avg_reconstruction_loss': np.mean(self.reconstruction_loss_history),
            'min_total_loss': min(self.total_loss_history),
            'max_total_loss': max(self.total_loss_history)
        }
        
        return metrics
        
    def save_trainer_state(self, path: str) -> None:
        """Save trainer state to disk.
        
        Args:
            path: Path to save state
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save optimizer state and metrics
            trainer_state = {
                'optimizer_state_dict': self.optimizer.state_dict(),
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'train_steps': self.train_steps,
                'total_loss_history': self.total_loss_history,
                'reconstruction_loss_history': self.reconstruction_loss_history
            }
            
            # Save
            torch.save(trainer_state, path)
            
            logger.info(f"Saved world model trainer state after {self.train_steps} steps")
        except Exception as e:
            logger.error(f"Error saving world model trainer state: {e}")
            
    def load_trainer_state(self, path: str) -> None:
        """Load trainer state from disk.
        
        Args:
            path: Path to load state from
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"World model trainer state file not found: {path}")
                return
                
            # Load state
            trainer_state = torch.load(path, map_location=self.device)
            
            # Restore state
            self.optimizer.load_state_dict(trainer_state['optimizer_state_dict'])
            self.learning_rate = trainer_state['learning_rate']
            self.batch_size = trainer_state['batch_size']
            self.train_steps = trainer_state['train_steps']
            self.total_loss_history = trainer_state['total_loss_history']
            self.reconstruction_loss_history = trainer_state['reconstruction_loss_history']
            
            logger.info(f"Loaded world model trainer state with {self.train_steps} steps")
        except Exception as e:
            logger.error(f"Error loading world model trainer state: {e}")
            
    def save_all(self, 
               model_path: str,
               buffer_path: str,
               trainer_path: str) -> None:
        """Save all components (model, buffer, trainer).
        
        Args:
            model_path: Path to save world model
            buffer_path: Path to save experience buffer
            trainer_path: Path to save trainer state
        """
        self.world_model.save_model(model_path)
        self.experience_buffer.save_buffer(buffer_path)
        self.save_trainer_state(trainer_path)
        
        logger.info("Saved all world model components")
        
    def load_all(self, 
               model_path: str,
               buffer_path: str,
               trainer_path: str) -> None:
        """Load all components (model, buffer, trainer).
        
        Args:
            model_path: Path to load world model from
            buffer_path: Path to load experience buffer from
            trainer_path: Path to load trainer state from
        """
        self.world_model.load_model(model_path)
        self.experience_buffer.load_buffer(buffer_path)
        self.load_trainer_state(trainer_path)
        
        logger.info("Loaded all world model components") 