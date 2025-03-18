"""
Intrinsic Curiosity Module (ICM) for exploration in Cities: Skylines 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict

class FeatureEncoder(nn.Module):
    """Encodes observations into latent feature space."""
    
    def __init__(self, config):
        super(FeatureEncoder, self).__init__()
        self.config = config
        self.device = config.get_device()
        
        # Get expected dimensions
        height, width = getattr(config, 'resolution', (240, 320))
        
        # CNN for encoding features
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        
        # Calculate the output size from CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, height, width, device=self.device)
            dummy_output = self.cnn(dummy_input)
            self.feature_size = dummy_output.shape[1]
            
        # Feature embedding layer
        self.fc = nn.Linear(self.feature_size, 256)
        
    def forward(self, x):
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Handle batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # Extract features
        features = self.cnn(x)
        return F.leaky_relu(self.fc(features))

class ForwardModel(nn.Module):
    """Predicts next state features given current state features and action."""
    
    def __init__(self, feature_size=256, action_dim=51):
        super(ForwardModel, self).__init__()
        
        # Action embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.LeakyReLU()
        )
        
        # Forward dynamics model
        self.forward_model = nn.Sequential(
            nn.Linear(feature_size + 128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, feature_size)
        )
        
    def forward(self, features, action_one_hot):
        # Embed action
        action_emb = self.action_embedding(action_one_hot)
        
        # Concatenate state features and action embedding
        combined = torch.cat([features, action_emb], dim=1)
        
        # Predict next state features
        predicted_next_features = self.forward_model(combined)
        return predicted_next_features

class InverseDynamicsModel(nn.Module):
    """Predicts action given current and next state features."""
    
    def __init__(self, feature_size=256, action_dim=51):
        super(InverseDynamicsModel, self).__init__()
        
        # Inverse dynamics model
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_size * 2, 256),
            nn.LeakyReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, features, next_features):
        # Concatenate current and next state features
        combined = torch.cat([features, next_features], dim=1)
        
        # Predict action logits
        action_logits = self.inverse_model(combined)
        return action_logits

class IntrinsicCuriosityModule(nn.Module):
    """Complete ICM combining feature encoding, forward and inverse models."""
    
    def __init__(self, config, action_dim=51, eta=0.2, beta=0.8, feature_scale=1.0):
        """Initialize the ICM.
        
        Args:
            config: Hardware and training configuration
            action_dim: Dimensionality of the action space
            eta: Scaling factor for intrinsic reward
            beta: Weight for inverse model loss vs forward model loss
            feature_scale: Scaling factor for feature prediction error
        """
        super(IntrinsicCuriosityModule, self).__init__()
        
        self.config = config
        self.device = config.get_device()
        self.action_dim = action_dim
        self.eta = eta
        self.beta = beta
        self.feature_scale = feature_scale
        
        # Feature encoder
        self.feature_encoder = FeatureEncoder(config)
        
        # Forward model
        self.forward_model = ForwardModel(
            feature_size=256,
            action_dim=action_dim
        )
        
        # Inverse model
        self.inverse_model = InverseDynamicsModel(
            feature_size=256,
            action_dim=action_dim
        )
        
        # Moving average for reward normalization
        self.reward_stats = {
            'mean': 0.0,
            'std': 1.0,
            'count': 0
        }
        
        # Memory of recent prediction errors for adaptive scaling
        self.recent_errors = []
        
        # Move to device
        self.to(self.device)
        
    def forward(self, state, next_state, action):
        """Forward pass through ICM.
        
        Args:
            state: Current observation
            next_state: Next observation
            action: Action taken (index)
            
        Returns:
            tuple: (intrinsic_reward, loss)
        """
        # Convert action to one-hot
        action_one_hot = torch.zeros(state.size(0), self.action_dim, device=self.device)
        action_one_hot.scatter_(1, action.unsqueeze(1), 1)
        
        # Encode states into features
        state_features = self.feature_encoder(state)
        next_state_features = self.feature_encoder(next_state)
        
        # Forward model: predict next state features
        predicted_next_features = self.forward_model(state_features, action_one_hot)
        
        # Inverse model: predict action from states
        predicted_action_logits = self.inverse_model(state_features, next_state_features)
        
        # Calculate losses
        forward_loss = F.mse_loss(predicted_next_features, next_state_features, reduction='none')
        forward_loss = forward_loss.mean(dim=1)
        inverse_loss = F.cross_entropy(predicted_action_logits, action.squeeze(-1), reduction='none')
        
        # Calculate intrinsic reward
        intrinsic_reward = self.eta * forward_loss
        
        # Update reward statistics
        for error in forward_loss.detach().cpu().numpy():
            self.recent_errors.append(error)
            if len(self.recent_errors) > 10000:
                self.recent_errors.pop(0)
        
        # Normalize rewards if enough samples
        if len(self.recent_errors) > 100:
            mean = np.mean(self.recent_errors)
            std = np.std(self.recent_errors) + 1e-8
            intrinsic_reward = self.eta * (forward_loss - mean) / std
            
        # Calculate combined loss for training
        loss = (1 - self.beta) * forward_loss.mean() + self.beta * inverse_loss.mean()
        
        return intrinsic_reward, loss
    
    def compute_curiosity_reward(self, state, next_state, action):
        """Compute just the intrinsic reward without training.
        
        Args:
            state: Current observation
            next_state: Next observation
            action: Action taken (index)
            
        Returns:
            torch.Tensor: Intrinsic reward
        """
        with torch.no_grad():
            # Convert action to one-hot
            if isinstance(action, int):
                action = torch.tensor([action], device=self.device)
            action_one_hot = torch.zeros(1, self.action_dim, device=self.device)
            action_one_hot.scatter_(1, action.unsqueeze(1), 1)
            
            # Encode states into features
            state_features = self.feature_encoder(state)
            next_state_features = self.feature_encoder(next_state)
            
            # Forward model: predict next state features
            predicted_next_features = self.forward_model(state_features, action_one_hot)
            
            # Calculate prediction error
            forward_error = F.mse_loss(predicted_next_features, next_state_features, reduction='none')
            forward_error = forward_error.mean(dim=1)
            
            # Scale the reward
            intrinsic_reward = self.eta * forward_error
            
            # Normalize if we have enough samples
            if len(self.recent_errors) > 100:
                mean = np.mean(self.recent_errors)
                std = np.std(self.recent_errors) + 1e-8
                intrinsic_reward = self.eta * (forward_error - mean) / std
            
            return intrinsic_reward 