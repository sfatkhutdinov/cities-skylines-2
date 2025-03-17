import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class CNNEncoder(nn.Module):
    def __init__(self, input_channels: int = 3):
        """CNN encoder for processing raw pixel inputs.
        
        Args:
            input_channels (int): Number of input channels (3 for RGB)
        """
        super(CNNEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Initialize weights using orthogonal initialization
        nn.init.orthogonal_(self.conv1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.conv3.weight, gain=nn.init.calculate_gain('relu'))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(x.size(0), -1)  # Flatten

class ActorCritic(nn.Module):
    def __init__(self, num_actions: int, hidden_size: int = 512):
        """Actor-Critic network for PPO algorithm.
        
        Args:
            num_actions (int): Number of possible actions
            hidden_size (int): Size of hidden layer
        """
        super(ActorCritic, self).__init__()
        
        self.encoder = CNNEncoder()
        
        # Calculate encoder output size (will be done dynamically in forward pass)
        self.encoder_output_size = None
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize the weights
        self.apply(self._init_weights)
        
    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action_logits, value)
        """
        # Encode the input
        encoded = self.encoder(x)
        
        if self.encoder_output_size is None:
            self.encoder_output_size = encoded.shape[1]
            # Add projection layer if needed
            self.projection = nn.Linear(self.encoder_output_size, 512)
        
        # Project to hidden size if needed
        encoded = self.projection(encoded)
        
        # Get action logits and value
        action_logits = self.actor(encoded)
        value = self.critic(encoded)
        
        return action_logits, value

class CuriosityModule(nn.Module):
    def __init__(self, hidden_size: int = 512, action_dim: int = 64):
        """Intrinsic curiosity module for exploration.
        
        Args:
            hidden_size (int): Size of hidden layers
            action_dim (int): Dimension of action space
        """
        super(CuriosityModule, self).__init__()
        
        self.encoder = CNNEncoder()
        
        # Forward dynamics model
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_size + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Inverse dynamics model
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        
    def forward(self, state: torch.Tensor, next_state: torch.Tensor, 
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through curiosity module.
        
        Args:
            state (torch.Tensor): Current state
            next_state (torch.Tensor): Next state
            action (torch.Tensor): Action taken
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                (predicted_next_features, predicted_action, intrinsic_reward)
        """
        # Encode states
        state_features = self.encoder(state)
        next_state_features = self.encoder(next_state)
        
        # Predict next state features
        combined = torch.cat([state_features, action], dim=1)
        predicted_next_features = self.forward_model(combined)
        
        # Predict action from state transitions
        states_combined = torch.cat([state_features, next_state_features], dim=1)
        predicted_action = self.inverse_model(states_combined)
        
        # Compute intrinsic reward as prediction error
        intrinsic_reward = F.mse_loss(predicted_next_features, next_state_features, reduction='none').mean(dim=1)
        
        return predicted_next_features, predicted_action, intrinsic_reward 