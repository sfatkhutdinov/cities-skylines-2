"""
World Model for Cities: Skylines 2 agent.
Predicts future states based on current state and actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

class WorldModel(nn.Module):
    """World Model for predicting future states and transitions."""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 latent_dim: int = 256,
                 hidden_dim: int = 512,
                 prediction_horizon: int = 5,
                 device=None):
        """Initialize the World Model.
        
        Args:
            state_dim: Dimension of the state representation
            action_dim: Dimension of the action space
            latent_dim: Dimension of the latent space
            hidden_dim: Dimension of hidden layers
            prediction_horizon: Number of timesteps to predict into the future
            device: Computation device
        """
        super(WorldModel, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        
        # State encoder (from high-dim state to latent representation)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        ).to(self.device)
        
        # State decoder (from latent representation back to state)
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        ).to(self.device)
        
        # Transition model (predicts next latent state given current latent state and action)
        self.transition_model = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        ).to(self.device)
        
        # LSTM for temporal dynamics modeling
        self.dynamics_lstm = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        ).to(self.device)
        
        # Output head for next state prediction
        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        ).to(self.device)
        
        # Optional: Uncertainty estimation (predict variance for each dimension)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Softplus()  # Ensures positive variance
        ).to(self.device)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.critical(f"Initialized World Model on device {self.device} with "
                        f"state_dim={state_dim}, action_dim={action_dim}, latent_dim={latent_dim}")
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def encode_state(self, state):
        """Encode a state into the latent space.
        
        Args:
            state: Input state tensor
            
        Returns:
            Latent representation of the state
        """
        # Ensure state is on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        
        # Handle batch dimension
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        # Encode state
        latent = self.state_encoder(state)
        return latent
    
    def decode_state(self, latent):
        """Decode a latent representation back to state space.
        
        Args:
            latent: Latent state representation
            
        Returns:
            Reconstructed state
        """
        # Ensure latent is on the correct device
        latent = latent.to(self.device)
        
        # Decode latent
        state = self.state_decoder(latent)
        return state
    
    def predict_next_latent(self, latent, action):
        """Predict the next latent state given current latent and action.
        
        Args:
            latent: Current latent state
            action: Action to take
            
        Returns:
            Predicted next latent state
        """
        # Ensure inputs are on the correct device
        latent = latent.to(self.device)
        
        # Convert action to one-hot if it's an index
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            
        if action.dim() == 1 and action.dtype == torch.long:
            action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()
        else:
            action_one_hot = action
            
        # Ensure action has batch dimension if latent does
        if latent.dim() > 1 and action_one_hot.dim() == 1:
            action_one_hot = action_one_hot.unsqueeze(0)
            
        # Concatenate latent state and action
        latent_action = torch.cat([latent, action_one_hot], dim=-1)
        
        # Predict next latent state
        next_latent = self.transition_model(latent_action)
        return next_latent
    
    def predict_trajectory(self, initial_state, actions_sequence):
        """Predict a trajectory of states given initial state and sequence of actions.
        
        Args:
            initial_state: Initial state
            actions_sequence: Sequence of actions to take
            
        Returns:
            Tuple of (predicted_states, uncertainty)
        """
        # Encode initial state to latent representation
        latent = self.encode_state(initial_state)
        
        # Initialize outputs
        batch_size = latent.shape[0]
        seq_len = len(actions_sequence)
        predicted_latents = torch.zeros(batch_size, seq_len, self.latent_dim).to(self.device)
        uncertainties = torch.zeros(batch_size, seq_len, self.latent_dim).to(self.device)
        
        # Initial hidden state for LSTM
        h0 = torch.zeros(2, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim).to(self.device)
        hidden = (h0, c0)
        
        # Process actions sequence
        action_tensors = []
        for action in actions_sequence:
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.long).to(self.device)
                
            if action.dim() == 0:
                action = action.unsqueeze(0)
                
            if action.dtype == torch.long:
                action_tensor = F.one_hot(action, num_classes=self.action_dim).float()
            else:
                action_tensor = action
                
            action_tensors.append(action_tensor)
        
        # Stack actions into sequence
        actions_tensor = torch.stack(action_tensors, dim=0).to(self.device)
        if actions_tensor.dim() == 2:
            # Add batch dimension if missing
            actions_tensor = actions_tensor.unsqueeze(1)
            
        # Transpose to (batch_size, seq_len, action_dim) if needed
        if actions_tensor.shape[0] == seq_len:
            actions_tensor = actions_tensor.transpose(0, 1)
        
        # Prepare input sequence for LSTM
        lstm_inputs = []
        current_latent = latent
        
        for t in range(seq_len):
            action = actions_tensor[:, t, :]
            # Concatenate latent state and action
            lstm_input = torch.cat([current_latent, action], dim=-1)
            lstm_inputs.append(lstm_input)
            
            # Use transition model for rolling prediction
            current_latent = self.predict_next_latent(current_latent, action)
            
        # Stack inputs into sequence
        lstm_inputs = torch.stack(lstm_inputs, dim=1)
        
        # Process through LSTM
        lstm_out, _ = self.dynamics_lstm(lstm_inputs, hidden)
        
        # Predict latent states and uncertainties
        for t in range(seq_len):
            predicted_latents[:, t, :] = self.next_state_head(lstm_out[:, t, :])
            uncertainties[:, t, :] = self.uncertainty_head(lstm_out[:, t, :])
        
        # Decode latent states back to original state space
        predicted_states = torch.zeros(batch_size, seq_len, self.state_dim).to(self.device)
        for t in range(seq_len):
            predicted_states[:, t, :] = self.decode_state(predicted_latents[:, t, :])
            
        return predicted_states, uncertainties
    
    def forward(self, state, action):
        """Forward pass: predict next state given current state and action.
        
        Args:
            state: Current state
            action: Action to take
            
        Returns:
            Tuple of (predicted_next_state, uncertainty)
        """
        # Encode state to latent
        latent = self.encode_state(state)
        
        # Predict next latent
        next_latent = self.predict_next_latent(latent, action)
        
        # Decode to state space
        next_state = self.decode_state(next_latent)
        
        # Calculate uncertainty (optional)
        # To do this properly, we need the LSTM hidden state
        # For a single step prediction, we'll use a simplified approach
        latent_action = torch.cat([latent, 
                                  F.one_hot(action, num_classes=self.action_dim).float() 
                                  if action.dtype == torch.long else action], dim=-1)
        
        # Create a sequence of length 1 for LSTM
        latent_action = latent_action.unsqueeze(1)
        
        # Initialize hidden state
        batch_size = latent.shape[0]
        h0 = torch.zeros(2, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim).to(self.device)
        hidden = (h0, c0)
        
        # Get LSTM output
        lstm_out, _ = self.dynamics_lstm(latent_action, hidden)
        
        # Calculate uncertainty
        uncertainty = self.uncertainty_head(lstm_out.squeeze(1))
        
        return next_state, uncertainty
    
    def compute_loss(self, states, actions, next_states):
        """Compute loss for training the world model.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            next_states: Batch of next states
            
        Returns:
            Dict of losses (reconstruction, prediction, etc.)
        """
        # Encode states
        latent_states = self.encode_state(states)
        
        # Decode and calculate reconstruction loss
        reconstructed_states = self.decode_state(latent_states)
        reconstruction_loss = F.mse_loss(reconstructed_states, states)
        
        # Predict next latent states
        predicted_latent_next = self.predict_next_latent(latent_states, actions)
        
        # Decode predicted next states
        predicted_next_states = self.decode_state(predicted_latent_next)
        
        # Calculate prediction loss
        prediction_loss = F.mse_loss(predicted_next_states, next_states)
        
        # Calculate latent consistency loss (optional)
        # This ensures that the latent space is consistent over time
        actual_latent_next = self.encode_state(next_states)
        latent_consistency_loss = F.mse_loss(predicted_latent_next, actual_latent_next)
        
        # Full trajectory prediction loss (optional, if sequence data is provided)
        trajectory_loss = torch.tensor(0.0).to(self.device)
        
        # Total loss
        total_loss = reconstruction_loss + prediction_loss + 0.1 * latent_consistency_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'prediction_loss': prediction_loss,
            'latent_consistency_loss': latent_consistency_loss,
            'trajectory_loss': trajectory_loss
        } 