"""
Optimized neural network for Cities: Skylines 2 agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import logging
import random
import math
import time

logger = logging.getLogger(__name__)

# Gradient clipping threshold for numerical stability
GRADIENT_CLIP_THRESHOLD = 10.0

# NaN/Inf detection and tracking dictionary
nan_inf_stats = {
    'action_logits_occurrences': 0,
    'value_occurrences': 0,
    'softmax_occurrences': 0
}

class ConvBlock(nn.Module):
    """Optimized convolutional block with batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()  # SiLU (Swish) activation for better gradient flow
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))

class SelfAttention(nn.Module):
    """Self-attention mechanism for temporal and spatial attention."""
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_linear = nn.Linear(embed_dim, embed_dim)
        self.output_dropout = nn.Dropout(dropout)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """Forward pass for self-attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Apply layer normalization
        residual = x
        x = self.layer_norm(x)
        
        # Project to queries, keys, and values
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        # Final projection and residual connection
        attn_output = self.output_linear(attn_output)
        attn_output = self.output_dropout(attn_output)
        output = residual + attn_output
        
        return output

class OptimizedNetwork(nn.Module):
    """Optimized neural network for the PPO agent."""
    
    def __init__(self, input_channels=3, num_actions=137, conv_channels=(32, 64, 64),
                 hidden_size=256, feature_size=512, frame_size=(84, 84),
                 frames_to_stack=1, use_lstm=True, lstm_hidden_size=256, num_layers=1,
                 device='cuda', use_attention=False, is_visual_input=True):
        """Initialize the optimized network.
        
        Args:
            input_channels: Number of input channels (in case of images)
            num_actions: Number of actions in the action space
            conv_channels: Tuple of convolutional channels
            hidden_size: Size of hidden layers
            feature_size: Size of feature representation
            frame_size: Size of input frames (height, width)
            frames_to_stack: Number of frames to stack
            use_lstm: Whether to use LSTM for temporal dependencies
            lstm_hidden_size: Size of LSTM hidden state
            num_layers: Number of LSTM layers
            device: Computation device
            use_attention: Whether to use attention mechanism
            is_visual_input: Whether input is visual (images) or vector
        """
        super(OptimizedNetwork, self).__init__()
        
        self.device = device
        self.use_lstm = use_lstm
        self.num_actions = num_actions
        self.is_visual_input = is_visual_input
        self.frames_to_stack = frames_to_stack
        
        # NaN/Inf detection and tracking dictionary
        self.nan_inf_stats = {
            'nan_occurrences': 0,
            'inf_occurrences': 0,
            'total_forwards': 0,
            'last_healthy_weights': None,
            'weight_reset_count': 0,
            'consecutive_nan_count': 0
        }
        
        # Set max gradient norm for clipping
        self.max_grad_norm = 1.0
        
        # Weight decay for regularization
        self.weight_decay = 1e-5
        
        # Add BatchNorm for stabilization
        self.use_batch_norm = True
        
        # Store frame size for calculating conv output
        self.frame_height, self.frame_width = frame_size
        
        if is_visual_input:
            # Build convolutional layers for processing visual input
            self.conv_layers = self._build_conv_layers(conv_channels)
            
            # Calculate the output size of the convolutional layers
            conv_output_size = self._calculate_conv_output_size(
                self.frame_height, self.frame_width, input_channels * frames_to_stack, conv_channels
            )
            
            # Input size for shared layers
            shared_input_size = conv_output_size
        else:
            # For vector input, input size is just the input dimension
            shared_input_size = input_channels * frames_to_stack
            self.conv_layers = None
            
        # Shared layers
        self.shared_layers = self._build_shared_layers(shared_input_size, feature_size)
        
        # LSTM for temporal dependencies
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=feature_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            self.lstm_hidden_size = lstm_hidden_size
            feature_output_size = lstm_hidden_size
        else:
            self.lstm = None
            feature_output_size = feature_size
        
        # Attention mechanism
        if use_attention:
            self.attention = SelfAttention(embed_dim=feature_output_size)
        else:
            self.attention = None
            
        # Policy head (actor)
        if self.use_batch_norm:
            self.policy_head = nn.Sequential(
                nn.Linear(feature_output_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_actions)
            )
            
            # Value head (critic)
            self.value_head = nn.Sequential(
                nn.Linear(feature_output_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        else:
            self.policy_head = nn.Sequential(
                nn.Linear(feature_output_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_actions)
            )
            
            # Value head (critic)
            self.value_head = nn.Sequential(
                nn.Linear(feature_output_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        
        # Move model to device
        self.to(device)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Save initial weights as fallback
        self.save_healthy_weights()
        
        logger.critical(f"Initialized OptimizedNetwork on device {device}")

    def _build_conv_layers(self, conv_channels):
        """
        Build the convolutional layers of the network.
        """
        layers = []
        
        # First conv layer - use effective_input_channels to account for stacked frames
        layers.extend([
            nn.Conv2d(self.effective_input_channels, conv_channels[0], kernel_size=8, stride=4),
            nn.BatchNorm2d(conv_channels[0], track_running_stats=False),
            nn.ReLU()
        ])
        
        # Second conv layer
        layers.extend([
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=4, stride=2),
            nn.BatchNorm2d(conv_channels[1], track_running_stats=False),
            nn.ReLU()
        ])
        
        # Third conv layer
        layers.extend([
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1),
            nn.BatchNorm2d(conv_channels[2], track_running_stats=False),
            nn.ReLU()
        ])
        
        return nn.Sequential(*layers)

    def _build_shared_layers(self, input_size, output_size):
        """
        Build the shared layers of the network.
        """
        logger.critical(f"Building shared layers with input_size={input_size}, output_size={output_size}")
        return nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size, track_running_stats=False),  # Add batch norm with track_running_stats=False
            nn.ReLU(),
            nn.Linear(self.hidden_size, output_size),
            nn.BatchNorm1d(output_size, track_running_stats=False),  # Add batch norm with track_running_stats=False
            nn.ReLU()
        )
    
    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _calculate_conv_output_size(self, height, width, input_channels, conv_channels):
        """Calculate the output size of the convolutional layers.
        
        Args:
            height: Height of input
            width: Width of input
            input_channels: Number of input channels
            conv_channels: Tuple of convolutional channel sizes
            
        Returns:
            int: Size of flattened output
        """
        # Simple estimation based on common architecture
        # Adjust if your conv architecture is different
        h, w = height, width
        
        # First conv layer: 8x8 kernel, stride 4
        h = (h - 8) // 4 + 1
        w = (w - 8) // 4 + 1
        
        # Second conv layer: 4x4 kernel, stride 2
        h = (h - 4) // 2 + 1
        w = (w - 4) // 2 + 1
        
        # Third conv layer: 3x3 kernel, stride 1
        h = (h - 3) // 1 + 1
        w = (w - 3) // 1 + 1
        
        # Size of flattened output
        return conv_channels[-1] * h * w

    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            hidden_state (tuple, optional): Previous hidden state for LSTM
            
        Returns:
            Tuple: (action_probs, value, next_hidden_state)
        """
        try:
            # Track forward passes
            self.nan_inf_stats['total_forwards'] += 1
            
            # Ensure input is on the correct device
            x = x.to(self.device)
            
            # Store batch size
            batch_size = x.size(0) if len(x.shape) > 1 else 1
            
            # Check for single sample (which may cause issues with BatchNorm)
            single_sample = batch_size == 1
            
            # If single sample and network is in training mode, temporarily switch to eval
            training_mode = self.training
            if single_sample and training_mode:
                self.eval()
            
            logger.debug(f"Network forward pass: input shape={x.shape}")
            
            # Process based on input type
            if self.is_visual_input:
                # Check if we need to add batch dimension
                if len(x.shape) == 3:  # [C, H, W]
                    x = x.unsqueeze(0)  # Add batch dimension
                
                # Process through convolutional layers
                features = self.conv_layers(x)
                features = features.reshape(x.size(0), -1)
                logger.critical(f"Flatten features shape: {features.shape}")
            else:
                # For vector inputs
                if len(x.shape) == 1:  # [D]
                    x = x.unsqueeze(0)  # Add batch dimension
                features = x
                
            # Process through shared layers
            features = self.shared_layers(features)
            
            # Process through LSTM if present
            next_hidden = None
            if self.use_lstm:
                logger.critical(f"Processing through LSTM")
                # If we're given a hidden state, use it
                if hidden_state is not None:
                    h, c = hidden_state
                    # Make sure hidden state is on same device
                    h, c = h.to(self.device), c.to(self.device)
                else:
                    # Initialize hidden state
                    h = torch.zeros(self.lstm.num_layers, features.size(0), self.lstm_hidden_size, device=self.device)
                    c = torch.zeros(self.lstm.num_layers, features.size(0), self.lstm_hidden_size, device=self.device)
                
                # Reshape for LSTM
                features = features.unsqueeze(1)  # Add sequence dimension
                
                # Process through LSTM
                features, (h, c) = self.lstm(features, (h, c))
                
                # Remove sequence dimension
                features = features.squeeze(1)
                
                # Save hidden state for next time
                next_hidden = (h, c)
            
            # Apply attention if present
            if self.attention is not None:
                logger.critical(f"Processing through attention mechanism")
                attention_input = features
                features = self.attention(attention_input)
                logger.critical(f"Attention transformation: {attention_input.shape} -> {features.shape}")
            
            logger.critical(f"LSTM/Attention output shape: {features.shape}")
            
            # Policy head (actor)
            logger.critical(f"Computing action logits")
            if single_sample and self.use_batch_norm:
                # If single sample and using batch norm, we need to manually apply layers
                action_logits = features
                for i, layer in enumerate(self.policy_head):
                    if isinstance(layer, nn.BatchNorm1d):
                        # Skip batch norm for single sample
                        continue
                    action_logits = layer(action_logits)
                    # Apply ReLU after linear layers except the last one
                    if isinstance(layer, nn.Linear) and i < len(self.policy_head) - 1:
                        action_logits = F.relu(action_logits)
            else:
                # Normal path with batch norm for batch size > 1
                logger.critical(f"Applying policy_head to features of shape: {features.shape}")
                action_logits = self.policy_head(features)
                
            logger.critical(f"Action logits shape: {action_logits.shape}")
            
            # Check for NaN/Inf values in action_logits
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                nan_count = torch.isnan(action_logits).sum().item()
                inf_count = torch.isinf(action_logits).sum().item()
                total_elements = action_logits.numel()
                nan_percent = 100.0 * nan_count / total_elements
                inf_percent = 100.0 * inf_count / total_elements
                
                # Update NaN statistics
                self.nan_inf_stats['nan_occurrences'] += 1
                self.nan_inf_stats['consecutive_nan_count'] += 1
                
                logger.critical(f"WARNING: NaN or Inf values detected in action_logits, applying fix")
                logger.critical(f"NaN percentage: {nan_percent:.2f}%, Inf percentage: {inf_percent:.2f}%")
                
                # If we've seen too many consecutive NaN values, try restoring healthy weights
                if self.nan_inf_stats['consecutive_nan_count'] >= 5:
                    if self.restore_healthy_weights():
                        # Re-run forward pass with healthy weights
                        return self.forward(x, hidden_state)
                    
                # Replace NaN/Inf with small random values
                mask = torch.isnan(action_logits) | torch.isinf(action_logits)
                random_values = torch.rand_like(action_logits) * 0.01 - 0.005  # Small values between -0.005 and 0.005
                action_logits = torch.where(mask, random_values, action_logits)
            else:
                # Reset consecutive NaN counter if we see healthy values
                self.nan_inf_stats['consecutive_nan_count'] = 0
                
                # If we have a reasonable number of forward passes and haven't seen NaNs,
                # consider saving the current weights as healthy
                if (self.nan_inf_stats['total_forwards'] % 1000 == 0 and 
                    self.nan_inf_stats['nan_occurrences'] < self.nan_inf_stats['total_forwards'] * 0.01):
                    self.save_healthy_weights()
            
            # Value head (critic)
            logger.critical(f"Computing value")
            if single_sample and self.use_batch_norm:
                # Apply value head manually without batch norm
                value = features
                for i, layer in enumerate(self.value_head):
                    if isinstance(layer, nn.BatchNorm1d):
                        continue  # Skip batch norm for single sample
                    value = layer(value)
                    # Apply ReLU after linear layers except the last one
                    if isinstance(layer, nn.Linear) and i < len(self.value_head) - 1:
                        value = F.relu(value)
            else:
                # Normal path with batch norm for batch size > 1
                logger.critical(f"Applying value_head to features of shape: {features.shape}")
                value = self.value_head(features)
                
            logger.critical(f"Value shape: {value.shape}")
            
            # Check for NaN/Inf values in value
            if torch.isnan(value).any() or torch.isinf(value).any():
                nan_count = torch.isnan(value).sum().item()
                inf_count = torch.isinf(value).sum().item()
                total_elements = value.numel()
                nan_percent = 100.0 * nan_count / total_elements
                inf_percent = 100.0 * inf_count / total_elements
                
                # Update NaN statistics
                self.nan_inf_stats['nan_occurrences'] += 1
                
                logger.critical(f"WARNING: NaN or Inf values detected in value, applying fix")
                logger.critical(f"NaN percentage: {nan_percent:.2f}%, Inf percentage: {inf_percent:.2f}%")
                
                # Replace NaN/Inf with small random values
                mask = torch.isnan(value) | torch.isinf(value)
                random_values = torch.rand_like(value) * 0.1 - 0.05  # Small values between -0.05 and 0.05
                value = torch.where(mask, random_values, value)
            
            # Apply softmax to get action probabilities
            logger.critical(f"Applying softmax to get action probabilities")
            # Use temperature scaling for numerical stability
            temperature = 1.0
            action_probs = F.softmax(action_logits / temperature, dim=-1)
            logger.critical(f"Action probabilities shape: {action_probs.shape}")
            
            # Check for NaN values after softmax
            if torch.isnan(action_probs).any():
                # If softmax produced NaNs, use uniform distribution as fallback
                action_probs = torch.ones_like(action_logits) / action_logits.size(-1)
            
            logger.critical(f"Forward pass completed successfully")
            
            # Restore original training mode if we changed it
            if single_sample and training_mode:
                self.train()
                
            return action_probs, value.view(-1, 1), next_hidden
            
        except Exception as e:
            logger.critical(f"ERROR in forward: {str(e)}")
            import traceback
            logger.critical(f"Traceback: {traceback.format_exc()}")
            
            # Try restoring healthy weights and doing a second attempt
            if self.restore_healthy_weights():
                try:
                    # Second attempt with restored weights
                    return self.forward(x, hidden_state)
                except Exception as e2:
                    logger.critical(f"Second attempt also failed: {str(e2)}")
                    pass
            
            # If restoration failed or second attempt failed, use fallback
            # Fallback to uniform distribution and zero value
            action_probs = torch.ones((batch_size, self.num_actions), device=self.device) / self.num_actions
            value = torch.zeros((batch_size, 1), device=self.device)
            
            # Ensure requires_grad for training
            action_probs = action_probs.detach().requires_grad_(True)
            value = value.detach().requires_grad_(True)
            
            # Restore original training mode if we changed it
            if single_sample and training_mode:
                self.train()
                
            return action_probs, value, None

    def get_action_probs(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get action probabilities for a state.
        
        Args:
            x (torch.Tensor): Input state tensor
            hidden_state: Optional hidden state for LSTM
            
        Returns:
            Tuple: (action_probs, next_hidden_state)
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Process based on input type
        if self.is_visual_input:
            # Check if we need to add batch dimension
            if len(x.shape) == 3:  # [C, H, W]
                x = x.unsqueeze(0)  # Add batch dimension
                
            # Process through convolutional layers
            features = self.conv_layers(x)
            features = features.reshape(x.size(0), -1)
        else:
            # For vector inputs
            if len(x.shape) == 1:  # [D]
                x = x.unsqueeze(0)  # Add batch dimension
            features = x
            
        # Process through shared layers
        features = self.shared_layers(features)
        
        # Process through LSTM if enabled
        next_hidden = None
        if self.use_lstm:
            if hidden_state is None:
                # Initialize hidden state if not provided
                h0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                c0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                hidden_state = (h0, c0)
            
            # Add time dimension for LSTM
            if len(features.shape) == 2:  # [B, F]
                features = features.unsqueeze(1)  # [B, 1, F]
            
            features, next_hidden = self.lstm(features, hidden_state)
            
            # Process through attention if enabled
            if self.use_attention:
                features = self.attention(features)
                # If using attention, take the last time step
                features = features.squeeze(1) if features.size(1) == 1 else features[:, -1]
        
        # Process through policy head
        logits = self.policy_head(features)
        
        # Apply softmax to get probabilities
        return torch.softmax(logits, dim=1), next_hidden
    
    def get_value(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get value for a state.
        
        Args:
            x (torch.Tensor): Input state tensor
            hidden_state: Optional hidden state for LSTM
            
        Returns:
            Tuple: (value, next_hidden_state)
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Process based on input type
        if self.is_visual_input:
            # Check if we need to add batch dimension
            if len(x.shape) == 3:  # [C, H, W]
                x = x.unsqueeze(0)  # Add batch dimension
                
            # Process through convolutional layers
            features = self.conv_layers(x)
            features = features.reshape(x.size(0), -1)
        else:
            # For vector inputs
            if len(x.shape) == 1:  # [D]
                x = x.unsqueeze(0)  # Add batch dimension
            features = x
            
        # Process through shared layers
        features = self.shared_layers(features)
        
        # Process through LSTM if enabled
        next_hidden = None
        if self.use_lstm:
            if hidden_state is None:
                # Initialize hidden state if not provided
                h0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                c0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                hidden_state = (h0, c0)
            
            # Add time dimension for LSTM
            if len(features.shape) == 2:  # [B, F]
                features = features.unsqueeze(1)  # [B, 1, F]
            
            features, next_hidden = self.lstm(features, hidden_state)
            
            # Process through attention if enabled
            if self.use_attention:
                features = self.attention(features)
                # If using attention, take the last time step
                features = features.squeeze(1) if features.size(1) == 1 else features[:, -1]
        
        # Process through value head
        value = self.value_head(features)
        
        return value.squeeze(-1), next_hidden

    def save_healthy_weights(self):
        """Save current weights as healthy weights that can be restored if NaNs occur."""
        self.nan_inf_stats['last_healthy_weights'] = {
            name: param.data.clone().detach() 
            for name, param in self.named_parameters()
        }
        logger.critical("Saved healthy network weights")
        
    def restore_healthy_weights(self):
        """Restore last known healthy weights if available."""
        if self.nan_inf_stats['last_healthy_weights'] is not None:
            for name, param in self.named_parameters():
                if name in self.nan_inf_stats['last_healthy_weights']:
                    param.data.copy_(self.nan_inf_stats['last_healthy_weights'][name])
            
            self.nan_inf_stats['weight_reset_count'] += 1
            logger.critical(f"Restored healthy network weights (reset count: {self.nan_inf_stats['weight_reset_count']})")
            return True
        else:
            logger.critical("No healthy weights available to restore")
            return False 