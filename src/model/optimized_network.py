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
GRADIENT_CLIP_THRESHOLD = 1.0

# NaN/Inf detection and tracking dictionary
nan_inf_stats = {
    'action_logits_occurrences': 0,
    'value_occurrences': 0,
    'softmax_occurrences': 0,
    'last_reset': time.time()
}

def reset_nan_counter():
    """Reset NaN counter periodically for tracking"""
    current_time = time.time()
    if current_time - nan_inf_stats['last_reset'] > 3600:  # Reset every hour
        for key in nan_inf_stats:
            if key != 'last_reset':
                nan_inf_stats[key] = 0
        nan_inf_stats['last_reset'] = current_time

def fix_nan_tensor(tensor: torch.Tensor, small_random=False, default_value=0.0, name="tensor"):
    """Fix NaN and Inf values in a tensor.
    
    Args:
        tensor: The tensor to fix
        small_random: Whether to use small random values
        default_value: Default value to use if not using random
        name: Name for logging
        
    Returns:
        Fixed tensor
    """
    if not torch.is_tensor(tensor):
        return tensor
        
    # Check if tensor requires gradients - save this information
    requires_grad = tensor.requires_grad
    
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total_elements = tensor.numel()
        nan_percent = 100.0 * nan_count / total_elements
        inf_percent = 100.0 * inf_count / total_elements
        
        logger.critical(f"WARNING: NaN or Inf values detected in {name}, applying fix")
        logger.critical(f"NaN percentage: {nan_percent:.2f}%, Inf percentage: {inf_percent:.2f}%")
        
        # Replace NaN/Inf with small random values or defaults
        mask = torch.isnan(tensor) | torch.isinf(tensor)
        
        # Create a detached copy for the replacement operation
        tensor_detached = tensor.detach().clone()
        
        if small_random:
            # Small values centered around default_value with small variance
            random_values = torch.rand_like(tensor_detached) * 0.01 + (default_value - 0.005)
            fixed_tensor = torch.where(mask, random_values, tensor_detached)
        else:
            fixed_tensor = torch.where(mask, torch.full_like(tensor_detached, default_value), tensor_detached)
        
        # Ensure the fixed tensor requires gradients if the original did
        if requires_grad:
            fixed_tensor = fixed_tensor.requires_grad_(True)
            
        return fixed_tensor
    
    return tensor

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
        
        # Flag to control batch norm behavior for single samples - set to False to prevent errors
        self.eval_bn_in_single_sample = True
        
        # Calculate effective input channels (for stacked frames)
        # Always initialize this attribute to avoid AttributeError
        self.effective_input_channels = input_channels * frames_to_stack if is_visual_input else input_channels
        logger.critical(f"Initializing network with effective input channels: {self.effective_input_channels} (input_channels={input_channels}, frames_to_stack={frames_to_stack})")

        # Build the visual encoder (convolutional layers)
        if is_visual_input:
            self.conv_layers = self._build_conv_layers(conv_channels)
        else:
            # Create a dummy identity layer for non-visual inputs
            self.conv_layers = nn.Identity()
        
        # Calculate the size of the features after convolution
        if is_visual_input:
            try:
                # Use effective input channels that includes frame stacking
                dummy_input = torch.zeros(1, self.effective_input_channels, frame_size[0], frame_size[1])
                with torch.no_grad():
                    x = self.conv_layers(dummy_input)
                    conv_output_size = x.view(1, -1).size(1)
                logger.critical(f"Calculated conv output size: {conv_output_size} from dummy input shape {dummy_input.shape}")
            except Exception as e:
                logger.error(f"Error calculating conv output size: {e}")
                # Fallback to a reasonable default size
                conv_output_size = 1024
                logger.critical(f"Using fallback conv output size: {conv_output_size}")
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
        
        # Policy head (actor) - Modified to use ReLU instead of BatchNorm to avoid batch size=1 errors
        self.policy_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.ReLU(),  # Replace BatchNorm with ReLU to avoid issues with batch size=1
            nn.Linear(256, num_actions)
        )
        
        # Value head (critic) - Modified to use ReLU instead of BatchNorm to avoid batch size=1 errors
        self.value_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.ReLU(),  # Replace BatchNorm with ReLU to avoid issues with batch size=1
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
            'after_softmax': 0,
            'input': 0,
            'features': 0,
            'lstm': 0
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
            nn.ReLU()  # Replace BatchNorm2d with just ReLU to avoid batch size issues
        ])
        
        # Second conv layer
        layers.extend([
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=4, stride=2),
            nn.ReLU()  # Replace BatchNorm2d with just ReLU
        ])
        
        # Third conv layer
        layers.extend([
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=3, stride=1),
            nn.ReLU()  # Replace BatchNorm2d with just ReLU
        ])
        
        return nn.Sequential(*layers)

    def _build_shared_layers(self, input_size, output_size):
        """
        Build shared fully connected layers.
        """
        # Check if dimensions mismatched with what we expect
        if input_size < 64:
            logger.critical(f"Input size for shared layers ({input_size}) is unexpectedly small")
            input_size = 1024  # Use a reasonable fallback size
        
        # Define the shared layers with simpler structure to avoid gradient issues
        layers = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()  # Use ReLU instead of LeakyReLU to avoid potential gradient issues
        )
        
        return layers
    
    def _init_weights(self, module):
        """Initialize weights using orthogonal initialization for better training stability.
        
        Args:
            module: Module to initialize
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
                    # Set forget gate bias to 1.0 for better memory retention
                    param_shape = param.shape
                    if len(param_shape) > 0:  # Skip if empty
                        # For LSTM, the forget gate bias is in the second quarter of the bias
                        forget_gate_size = param_shape[0] // 4
                        param.data[forget_gate_size:2*forget_gate_size].fill_(1.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor
            hidden_state: LSTM hidden state
            
        Returns:
            action_probs: Action probabilities
            value: Value estimate
            next_hidden: Next LSTM hidden state
        """
        # Reset NaN counter periodically
        reset_nan_counter()
        
        # Initial input validation
        if x is None:
            logger.error("Input tensor is None!")
            # Return a safe fallback
            action_logits = torch.zeros(1, self.num_actions, device=self.device, requires_grad=True)
            value = torch.zeros(1, 1, device=self.device, requires_grad=True)
            return F.softmax(action_logits, dim=-1), value, hidden_state
            
        # Check for NaN in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            self.nan_occurrences['input'] += 1
            logger.critical(f"NaN or Inf detected in input tensor! Occurrence #{self.nan_occurrences['input']}")
            # Fix NaN values in input
            x = fix_nan_tensor(x, small_random=True, default_value=0.0, name="input")
            
        # Ensure input requires gradients for backpropagation
        if not x.requires_grad:
            x = x.detach().clone().requires_grad_(True)
        
        # Get batch size and maintain input tensor shape
        batch_size = x.size(0)
        has_time_dim = len(x.shape) > 4
        orig_shape = x.shape
        
        # Batch size check - exit early with dummy output for empty batches
        if batch_size == 0:
            logger.critical("Empty batch detected! Returning dummy outputs.")
            action_logits = torch.zeros(1, self.num_actions, device=self.device, requires_grad=True)
            value = torch.zeros(1, 1, device=self.device, requires_grad=True)
            return F.softmax(action_logits, dim=-1), value, hidden_state
        
        try:
            # Reshape if there's a time dimension
            if has_time_dim:
                # Reshape to [batch_size * time_steps, channels, height, width]
                x = x.view(-1, *x.shape[2:])
                
            # Handle edge case of missing input channels
            if self.is_visual_input and x.size(1) != self.effective_input_channels:
                logger.critical(f"Input channel mismatch: expected {self.effective_input_channels}, got {x.size(1)} - adapting input")
                
                # Try to adapt the input to match expected channels through various means
                if x.size(1) < self.effective_input_channels:
                    # Add missing channels by duplication or zero padding
                    missing_channels = self.effective_input_channels - x.size(1)
                    if x.size(1) > 0:
                        # Duplicate existing channels
                        repeat_times = math.ceil(self.effective_input_channels / x.size(1))
                        repeated = x.repeat(1, repeat_times, 1, 1)
                        x = repeated[:, :self.effective_input_channels]
                    else:
                        # Create new input with zeros
                        x = torch.zeros(x.size(0), self.effective_input_channels, *x.shape[2:], device=x.device)
                elif x.size(1) > self.effective_input_channels:
                    # Take only the needed channels
                    x = x[:, :self.effective_input_channels]
                
            # Pass through convolutional layers if using visual input
            if self.is_visual_input:
                # Ensure tensor has correct dimensions for conv2d (batch, channels, height, width)
                if len(x.shape) == 2:  # [batch, channels] shape detected
                    logger.critical(f"Reshaping input from {x.shape} to proper 4D shape for Conv2D")
                    # If this is a [batch, features] tensor, we need to reshape it to [batch, channels, height, width]
                    batch_size = x.size(0)
                    
                    # Create a proper sized tensor with minimum dimensions that the conv layers can handle
                    # Ensure height and width are at least as large as the first kernel size (typically 8x8)
                    min_spatial_dim = 8
                    
                    # Create a tensor with correct shape (batch, channels, height, width)
                    # Default to 8x8 spatial dimensions which should be large enough for first kernel
                    x_reshaped = torch.zeros(
                        batch_size, 
                        self.effective_input_channels, 
                        min_spatial_dim, 
                        min_spatial_dim, 
                        device=x.device
                    )
                    
                    # Copy data where possible
                    feature_count = min(x.size(1), self.effective_input_channels)
                    
                    # Fill the first channel(s) with the original data, spreading features across channels
                    for i in range(min(feature_count, self.effective_input_channels)):
                        if i < x.size(1):
                            # Spread the feature data across the spatial dimensions
                            # Simple approach: repeat the same value across the spatial dimensions
                            x_reshaped[:, i, :, :] = x[:, i].unsqueeze(-1).unsqueeze(-1).expand(-1, min_spatial_dim, min_spatial_dim)
                    
                    # Replace x with properly reshaped tensor
                    x = x_reshaped
                    logger.critical(f"Reshaped input to {x.shape} for convolutional processing")
                
                # Final check to ensure we have 4D tensor with sufficient spatial dimensions
                if len(x.shape) != 4 or x.shape[2] < 8 or x.shape[3] < 8:
                    logger.critical(f"Input shape {x.shape} is not suitable for convolution, creating proper tensor")
                    
                    # Get current batch size or default to 1
                    batch_size = x.size(0) if len(x.shape) > 0 else 1
                    
                    # Create a tensor with correct shape (batch, channels, height, width)
                    x_proper = torch.zeros(
                        batch_size, 
                        self.effective_input_channels, 
                        8, 8, 
                        device=x.device
                    )
                    
                    # Copy values if possible
                    if len(x.shape) == 3:  # [batch, height, width]
                        for c in range(min(self.effective_input_channels, 3)):
                            # Broadcast to channels
                            x_proper[:, c] = F.interpolate(
                                x.unsqueeze(1), 
                                size=(8, 8), 
                                mode='nearest'
                            ).squeeze(1)
                    
                    # Replace with properly dimensioned tensor
                    x = x_proper

                try:
                    features = self.conv_layers(x)
                    features = features.view(features.size(0), -1)  # Flatten
                except Exception as e:
                    logger.critical(f"Forward pass failed with exception: {str(e)}")
                    logger.critical(f"Input shape was {x.shape}")
                    # Create fallback features with the expected output size of the conv layers
                    # Calculate the expected output size based on our network structure
                    expected_flatten_size = self._calculate_conv_output_size()
                    features = torch.zeros(x.size(0), expected_flatten_size, device=x.device, requires_grad=True)
                    logger.critical(f"Using fallback features with shape {features.shape}")
            else:
                features = x
                
            # Check for NaN in features
            if torch.isnan(features).any() or torch.isinf(features).any():
                self.nan_occurrences['features'] += 1
                logger.critical(f"NaN or Inf detected in features! Occurrence #{self.nan_occurrences['features']}")
                # Fix NaN values
                features = fix_nan_tensor(features, small_random=True, default_value=0.1, name="features")
                
            # Pass through shared layers
            try:
                # Ensure features match the expected input size for shared_layers
                conv_output_size = self._calculate_conv_output_size()
                
                # If dimensions don't match, reshape features to expected size
                if features.shape[-1] != conv_output_size:
                    logger.critical(f"Features shape {features.shape} doesn't match expected conv output size {conv_output_size}")
                    if features.dim() == 2:  # [batch, features]
                        # Resize features to match expected conv output size
                        resized_features = torch.zeros(features.size(0), conv_output_size, device=features.device, requires_grad=True)
                        # Copy data if possible
                        min_size = min(features.size(1), conv_output_size)
                        resized_features[:, :min_size] = features[:, :min_size]
                        features = resized_features
                        logger.critical(f"Resized features to shape {features.shape}")
                
                features = self.shared_layers(features)
            except Exception as e:
                logger.critical(f"Error in shared_layers: {str(e)}")
                # Fallback to create output with the right dimensions
                features = torch.zeros(features.size(0), self.feature_size, device=features.device, requires_grad=True)
                logger.critical(f"Using fallback features for shared layers with shape {features.shape}")
            
            # Reshape features back to include time dimension if needed
            if has_time_dim:
                features = features.view(batch_size, -1, features.size(-1))  # [batch_size, time_steps, features]
            
            # LSTM processing if enabled
            lstm_output = features
            next_hidden = None
            
            if self.use_lstm:
                if features.dim() == 2:
                    # Add time dimension for single step input
                    lstm_input = features.unsqueeze(1)
                else:
                    lstm_input = features
                    
                try:
                    lstm_output, next_hidden = self.lstm(lstm_input, hidden_state)
                    
                    # Check for NaN in LSTM output
                    if torch.isnan(lstm_output).any() or torch.isinf(lstm_output).any() or \
                       (next_hidden is not None and (torch.isnan(next_hidden[0]).any() or torch.isinf(next_hidden[0]).any())):
                        self.nan_occurrences['lstm'] += 1
                        logger.critical(f"NaN or Inf detected in LSTM output or hidden state! Occurrence #{self.nan_occurrences['lstm']}")
                        
                        # Fix LSTM output
                        lstm_output = fix_nan_tensor(lstm_output, small_random=True, default_value=0.1, name="lstm_output")
                        
                        # Fix hidden state
                        if next_hidden is not None:
                            h, c = next_hidden
                            h = fix_nan_tensor(h, small_random=True, default_value=0.0, name="lstm_h")
                            c = fix_nan_tensor(c, small_random=True, default_value=0.0, name="lstm_c")
                            next_hidden = (h, c)
                            
                except Exception as e:
                    logger.error(f"LSTM forward pass failed: {e}")
                    # Use features directly as fallback with gradients
                    lstm_output = features.detach().clone().requires_grad_(True)
                    next_hidden = hidden_state
                
                # Apply attention if requested
                if self.use_attention:
                    try:
                        lstm_output = self.attention(lstm_output)
                    except Exception as e:
                        logger.error(f"Attention mechanism failed: {e}")
                
                # Extract last timestep for single step prediction
                if has_time_dim:
                    lstm_output = lstm_output[:, -1]
                else:
                    lstm_output = lstm_output.squeeze(1)
            
            # Get action logits and value estimate
            try:
                # No special handling for batch norm with single samples - always use the same forward pass
                action_logits = self.policy_head(lstm_output)
                value = self.value_head(lstm_output)
            except Exception as e:
                logger.error(f"Head forward pass failed: {e}")
                # Fallback values with gradients enabled
                action_logits = torch.zeros(lstm_output.size(0), self.num_actions, device=self.device, requires_grad=True)
                value = torch.zeros(lstm_output.size(0), 1, device=self.device, requires_grad=True)
            
            # Check for NaN values in action logits
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                self.nan_occurrences['action_logits'] += 1
                logger.critical(f"NaN or Inf detected in action logits! Occurrence #{self.nan_occurrences['action_logits']}")
                
                # Generate fallback action logits
                action_logits = fix_nan_tensor(action_logits, small_random=True, default_value=0.0, name="action_logits")
                
            # Check for NaN values in value estimates
            if torch.isnan(value).any() or torch.isinf(value).any():
                self.nan_occurrences['value'] += 1
                logger.critical(f"NaN or Inf detected in value estimate! Occurrence #{self.nan_occurrences['value']}")
                
                # Generate fallback value
                value = fix_nan_tensor(value, small_random=False, default_value=0.0, name="value")
            
            # Apply softmax to get action probabilities
            try:
                action_probs = F.softmax(action_logits, dim=-1)
                
                # Check for NaN values after softmax (can happen with extreme values)
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    self.nan_occurrences['after_softmax'] += 1
                    logger.critical(f"NaN or Inf detected after softmax! Occurrence #{self.nan_occurrences['after_softmax']}")
                    
                    # Fall back to uniform distribution with gradient
                    action_probs = torch.ones_like(action_logits, requires_grad=True) / self.num_actions
            except Exception as e:
                logger.error(f"Softmax failed: {e}")
                # Fall back to uniform distribution with gradient
                action_probs = torch.ones_like(action_logits, requires_grad=True) / self.num_actions
                
            return action_probs, value, next_hidden
            
        except Exception as e:
            import traceback
            logger.critical(f"Forward pass failed with exception: {e}")
            logger.critical(traceback.format_exc())
            
            # Return safe fallback values
            action_logits = torch.ones(batch_size, self.num_actions, device=self.device, requires_grad=True) / self.num_actions
            value = torch.zeros(batch_size, 1, device=self.device, requires_grad=True)
            
            return action_logits, value, hidden_state
        
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

    def _calculate_conv_output_size(self):
        """Calculate the expected flattened output size of the convolutional layers."""
        try:
            # Use the conv_channels from instance
            channels = getattr(self, 'conv_channels', (32, 64, 64))
            
            # Start with frame_size from initialization
            h, w = self.frame_size
            
            # Apply convolution formulas with actual parameters from our architecture
            # 1st layer: kernel=8, stride=4, padding=0
            h = (h - 8) // 4 + 1
            w = (w - 8) // 4 + 1
            
            # 2nd layer: kernel=4, stride=2, padding=0
            h = (h - 4) // 2 + 1
            w = (w - 4) // 2 + 1
            
            # 3rd layer: kernel=3, stride=1, padding=0
            h = (h - 3) // 1 + 1
            w = (w - 3) // 1 + 1
            
            # Calculate final size
            output_size = channels[-1] * h * w
            
            logger.debug(f"Calculated conv output size: {output_size} (h={h}, w={w}, c={channels[-1]})")
            
            return output_size
            
        except Exception as e:
            logger.error(f"Error calculating conv output size: {e}")
            # Return a fallback value that matches our network architecture for 84x84 input
            # For 84x84 input with our architecture, should be close to 3136 (= 64 * 7 * 7)
            fallback_size = 3136
            logger.warning(f"Using fallback conv output size: {fallback_size}")
            return fallback_size 