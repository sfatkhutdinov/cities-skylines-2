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
        """
        Initialize the network.
        """
        super(OptimizedNetwork, self).__init__()
        
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.frame_size = frame_size
        self.frames_to_stack = frames_to_stack
        self.device = device
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.is_visual_input = is_visual_input
        
        # Flag to control batch norm behavior for single samples
        self.eval_bn_in_single_sample = True
        
        # Calculate effective input channels (for stacked frames)
        self.effective_input_channels = input_channels * frames_to_stack if is_visual_input else input_channels
        logger.critical(f"Initializing network with effective input channels: {self.effective_input_channels} (input_channels={input_channels}, frames_to_stack={frames_to_stack})")

        # Build the visual encoder (convolutional layers)
        if is_visual_input:
            self.conv_layers = self._build_conv_layers(conv_channels)
        
        # Calculate the size of the features after convolution
        if is_visual_input:
            # Use effective input channels that includes frame stacking
            dummy_input = torch.zeros(1, self.effective_input_channels, frame_size[0], frame_size[1])
            with torch.no_grad():
                x = self.conv_layers(dummy_input)
                conv_output_size = x.view(1, -1).size(1)
            logger.critical(f"Calculated conv output size: {conv_output_size} from dummy input shape {dummy_input.shape}")
        else:
            # For non-visual input, use input size directly
            conv_output_size = input_channels
        
        # Build shared fully connected layers
        self.shared_layers = self._build_shared_layers(conv_output_size, feature_size)
        
        # Build LSTM if requested
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=feature_size,
                hidden_size=lstm_hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            
            # Add attention mechanism if requested
            if use_attention:
                # Modified attention mechanism that preserves the feature dimensions
                self.attention = nn.Sequential(
                    nn.Linear(lstm_hidden_size, lstm_hidden_size),
                    nn.Tanh(),
                    nn.Linear(lstm_hidden_size, lstm_hidden_size)
                )
            
            self.output_size = lstm_hidden_size
        else:
            self.lstm = None
            self.output_size = feature_size
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.BatchNorm1d(256, track_running_stats=False),  # Add batch norm with track_running_stats=False
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.BatchNorm1d(256, track_running_stats=False),  # Add batch norm with track_running_stats=False
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # LSTM state
        self.hidden = None
        
        # NaN detection and handling
        self.nan_occurrences = {
            'action_logits': 0,
            'value': 0,
            'after_softmax': 0
        }
        
        # Move to the specified device
        self.to(device)

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

    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            hidden_state: Optional hidden state for LSTM (h, c)
            
        Returns:
            Tuple: (action_probs, value, next_hidden_state)
        """
        logger.debug(f"Network forward pass: input shape={x.shape}")
        
        # Detect batch size for batch norm handling
        batch_size = x.size(0)
        single_sample = batch_size == 1
        
        # Temporarily set to eval mode for batch norm if single sample
        training_mode = self.training
        if single_sample and training_mode:
            self.eval()
            
        try:
            # Check if the input is already a feature vector
            if len(x.shape) == 2:
                logger.critical(f"Network forward called with input shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
                logger.critical(f"Input appears to be a feature vector with shape {x.shape}, skipping conv layers")
                
                # Input is already features from encoder, bypass convolution
                features = x
                # Check if feature dimensions match the expected shared layer output
                if features.shape[1] == self.feature_size:
                    logger.critical(f"Feature vector already matches feature_size ({self.feature_size}), skipping shared layers")
                    logger.critical(f"Skipping shared layers, features already match expected size: {features.shape}")
                    encoder_features = features
                else:
                    # Process through shared layers
                    if single_sample:
                        # Apply shared layers manually without batch norm for single sample
                        for i, layer in enumerate(self.shared_layers):
                            if isinstance(layer, nn.BatchNorm1d):
                                continue  # Skip batch norm for single sample
                            features = layer(features)
                            # Apply ReLU after linear layers
                            if isinstance(layer, nn.Linear) and i < len(self.shared_layers) - 1:
                                features = F.relu(features)
                        encoder_features = features
                    else:
                        # Normal path with batch norm for batch size > 1
                        encoder_features = self.shared_layers(features)
            else:
                # Full forward pass with convolutional layers
                if single_sample:
                    # For single sample, apply conv layers manually without batch norm
                    features = x
                    
                    # Extract conv layers from self.conv_layers
                    for i, layer in enumerate(self.conv_layers):
                        if isinstance(layer, nn.BatchNorm2d):
                            continue  # Skip batch norm for single sample
                        features = layer(features)
                        # Apply ReLU after conv layers
                        if isinstance(layer, nn.Conv2d):
                            features = F.relu(features)
                    
                    # Flatten
                    features = features.view(features.size(0), -1)
                    
                    # Apply shared layers manually without batch norm
                    for i, layer in enumerate(self.shared_layers):
                        if isinstance(layer, nn.BatchNorm1d):
                            continue  # Skip batch norm for single sample
                        features = layer(features)
                        # Apply ReLU after linear layers
                        if isinstance(layer, nn.Linear) and i < len(self.shared_layers) - 1:
                            features = F.relu(features)
                    
                    encoder_features = features
                else:
                    # Normal path with batch norm for batch size > 1
                    # Apply convolutional layers
                    features = self.conv_layers(x)
                    
                    # Flatten features for the linear layers
                    features = features.view(batch_size, -1)
                    
                    # Check if shape matches expected input for shared layers
                    logger.critical(f"Flatten features shape: {features.shape}")
                    
                    # Apply shared layers
                    encoder_features = self.shared_layers(features)
            
            # Process through LSTM if present
            next_hidden = None
            if self.use_lstm:
                logger.critical(f"Processing through LSTM")
                # Add time dimension for LSTM
                lstm_in = encoder_features.unsqueeze(1)
                # LSTM output
                if hidden_state is None:
                    # Initialize hidden state
                    batch_size = lstm_in.size(0)
                    h0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=self.device)
                    c0 = torch.zeros(1, batch_size, self.lstm_hidden_size, device=self.device)
                    hidden_state = (h0, c0)
                else:
                    # Reuse existing hidden state
                    h, c = hidden_state
                    if h.device != self.device:
                        h = h.to(self.device)
                    if c.device != self.device:
                        c = c.to(self.device)
                    hidden_state = (h, c)
                
                lstm_out, next_hidden = self.lstm(lstm_in, hidden_state)
                # Reshape LSTM output
                features = lstm_out.squeeze(1)
                
                # Process through attention if enabled
                if self.use_attention:
                    logger.critical(f"Processing through attention mechanism")
                    # Store original feature shape for logging
                    original_shape = features.shape
                    features = self.attention(features)
                    logger.critical(f"Attention transformation: {original_shape} -> {features.shape}")
                
                logger.critical(f"LSTM/Attention output shape: {features.shape}")
            
            # Policy head (actor)
            logger.critical(f"Computing action logits")
            if single_sample:
                # Apply policy head manually without batch norm
                action_logits = features
                for i, layer in enumerate(self.policy_head):
                    if isinstance(layer, nn.BatchNorm1d):
                        continue  # Skip batch norm for single sample
                    try:
                        action_logits = layer(action_logits)
                        logger.critical(f"After layer {i}, action_logits shape: {action_logits.shape}")
                    except Exception as e:
                        logger.critical(f"Error in policy head layer {i}: {str(e)} - input shape: {action_logits.shape}, layer: {layer}")
                        raise
                    # Apply ReLU after linear layers except the last one
                    if isinstance(layer, nn.Linear) and i < len(self.policy_head) - 1:
                        action_logits = F.relu(action_logits)
            else:
                # Normal path with batch norm for batch size > 1
                logger.critical(f"Applying policy_head to features of shape: {features.shape}")
                try:
                    action_logits = self.policy_head(features)
                except Exception as e:
                    logger.critical(f"Error in policy_head: {str(e)} - input shape: {features.shape}")
                    raise
                
            logger.critical(f"Action logits shape: {action_logits.shape}")
            
            # Check for NaN/Inf values in action_logits
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                nan_count = torch.isnan(action_logits).sum().item()
                inf_count = torch.isinf(action_logits).sum().item()
                total_elements = action_logits.numel()
                nan_percent = 100.0 * nan_count / total_elements
                inf_percent = 100.0 * inf_count / total_elements
                
                logger.critical(f"WARNING: NaN or Inf values detected in action_logits, applying fix")
                logger.critical(f"NaN percentage: {nan_percent:.2f}%, Inf percentage: {inf_percent:.2f}%")
                
                # Replace NaN/Inf with small random values instead of zeros
                mask = torch.isnan(action_logits) | torch.isinf(action_logits)
                random_values = torch.rand_like(action_logits) * 0.01 - 0.005  # Small values between -0.005 and 0.005
                action_logits = torch.where(mask, random_values, action_logits)
            
            # Value head (critic)
            logger.critical(f"Computing value")
            if single_sample:
                # Apply value head manually without batch norm
                value = features
                for i, layer in enumerate(self.value_head):
                    if isinstance(layer, nn.BatchNorm1d):
                        continue  # Skip batch norm for single sample
                    try:
                        value = layer(value)
                        logger.critical(f"After value layer {i}, value shape: {value.shape}")
                    except Exception as e:
                        logger.critical(f"Error in value head layer {i}: {str(e)} - input shape: {value.shape}, layer: {layer}")
                        raise
                    # Apply ReLU after linear layers except the last one
                    if isinstance(layer, nn.Linear) and i < len(self.value_head) - 1:
                        value = F.relu(value)
            else:
                # Normal path with batch norm for batch size > 1
                logger.critical(f"Applying value_head to features of shape: {features.shape}")
                try:
                    value = self.value_head(features)
                except Exception as e:
                    logger.critical(f"Error in value_head: {str(e)} - input shape: {features.shape}")
                    raise
                
            logger.critical(f"Value shape: {value.shape}")
            
            # Check for NaN/Inf values in value
            if torch.isnan(value).any() or torch.isinf(value).any():
                nan_count = torch.isnan(value).sum().item()
                inf_count = torch.isinf(value).sum().item()
                total_elements = value.numel()
                nan_percent = 100.0 * nan_count / total_elements
                inf_percent = 100.0 * inf_count / total_elements
                
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
            
            # Fallback to uniform distribution and zero value
            action_probs = torch.ones((batch_size, self.policy_head[-1].out_features), device=x.device) / self.policy_head[-1].out_features
            value = torch.zeros((batch_size, 1), device=x.device, requires_grad=True)
            
            logger.critical(f"Returning fallback values with shapes: probs={action_probs.shape}, value={value.shape}")
            logger.critical(f"Fallback values require_grad: probs={action_probs.requires_grad}, value={value.requires_grad}")
            
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