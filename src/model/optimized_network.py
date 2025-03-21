"""
Optimized neural network for Cities: Skylines 2 agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

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
    
    def __init__(self, input_shape, num_actions, device=None, use_lstm=True, lstm_hidden_size=256, 
                 use_attention=False, attention_heads=4):
        """Initialize the network.
        
        Args:
            input_shape: Shape of the input state (channels, height, width) or flattened size
            num_actions: Number of actions in the action space
            device: Computation device
            use_lstm: Whether to use LSTM for temporal reasoning
            lstm_hidden_size: Size of LSTM hidden state
            use_attention: Whether to use self-attention mechanism
            attention_heads: Number of attention heads if attention is used
        """
        super(OptimizedNetwork, self).__init__()
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.critical(f"Initializing network with input_shape={input_shape}, num_actions={num_actions}, device={self.device}")
        
        # LSTM configuration
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        logger.critical(f"LSTM enabled: {self.use_lstm}, hidden size: {self.lstm_hidden_size}")
        
        # Attention configuration
        self.use_attention = use_attention
        self.attention_heads = attention_heads
        logger.critical(f"Attention enabled: {self.use_attention}, heads: {self.attention_heads}")
        
        # Handle different types of input shapes
        if isinstance(input_shape, tuple) and len(input_shape) >= 3:
            # It's an image with shape (channels, height, width)
            in_channels, height, width = input_shape
            self.is_visual_input = True
            logger.critical(f"Visual input detected: channels={in_channels}, height={height}, width={width}")
        elif isinstance(input_shape, tuple) and len(input_shape) == 1:
            # It's a flattened vector with shape (n,)
            in_channels, height, width = 1, 1, input_shape[0]
            self.is_visual_input = False
            logger.critical(f"Vector input detected (tuple): size={input_shape[0]}")
        elif isinstance(input_shape, int):
            # It's a flattened vector with size n
            in_channels, height, width = 1, 1, input_shape
            self.is_visual_input = False
            logger.critical(f"Vector input detected (int): size={input_shape}")
        else:
            error_msg = f"Unsupported input shape: {input_shape}, type: {type(input_shape)}"
            logger.critical(error_msg)
            raise ValueError(error_msg)
            
        # Define network architecture based on input type
        if self.is_visual_input:
            # Convolutional network for visual inputs
            logger.critical("Creating convolutional network for visual input")
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            
            # Create a dummy input tensor to calculate output size
            logger.critical("Calculating convolutional output size...")
            try:
                with torch.no_grad():
                    dummy_input = torch.zeros(1, in_channels, height, width)
                    dummy_output = self.conv_layers(dummy_input)
                    conv_output_size = int(np.prod(dummy_output.shape))
                    logger.critical(f"Conv dummy input shape: {dummy_input.shape}")
                    logger.critical(f"Conv dummy output shape: {dummy_output.shape}")
                    logger.critical(f"Conv output size: {conv_output_size}")
            except Exception as e:
                logger.critical(f"Error calculating conv output size: {e}")
                import traceback
                logger.critical(f"Traceback: {traceback.format_exc()}")
                # Use a conservative estimate
                conv_output_size = 1024
                logger.critical(f"Using fallback conv output size: {conv_output_size}")
                
            # Shared feature layers
            logger.critical(f"Creating shared layers with input size {conv_output_size}")
            self.shared_layers = nn.Sequential(
                nn.Linear(conv_output_size, 512),
                nn.ReLU()
            )
            
            # Track the feature size for LSTM
            self.feature_size = 512
        else:
            # For vector inputs, we won't use conv layers
            logger.critical("Creating linear network for vector input")
            self.conv_layers = nn.Identity()
            
            # Fully connected network for vector inputs
            logger.critical(f"Creating shared layers with input size {width}")
            self.shared_layers = nn.Sequential(
                nn.Linear(width, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU()
            )
            
            # Track the feature size for LSTM
            self.feature_size = 512
            
        # LSTM layer for temporal reasoning if enabled
        if self.use_lstm:
            logger.critical(f"Creating LSTM layer with hidden size {self.lstm_hidden_size}")
            self.lstm = nn.LSTM(
                input_size=self.feature_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=1,
                batch_first=True
            )
            
            # Update feature size for heads
            self.output_size = self.lstm_hidden_size
        else:
            # No LSTM
            self.lstm = None
            self.output_size = self.feature_size
        
        # Attention layer after LSTM if enabled
        if self.use_attention:
            logger.critical(f"Creating attention layer with {self.attention_heads} heads")
            self.attention = SelfAttention(
                embed_dim=self.output_size,
                num_heads=self.attention_heads,
                dropout=0.1
            )
        else:
            self.attention = None
            
        # Policy head
        logger.critical(f"Creating policy head with input size {self.output_size}")
        self.policy_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
        # Value head
        logger.critical(f"Creating value head with input size {self.output_size}")
        self.value_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights before moving to device
        logger.critical("Initializing network weights")
        self.apply(self._init_weights)
        
        # Move all components to the specified device after creating all layers
        try:
            logger.critical(f"Moving network to device: {self.device}")
            self.to(self.device)
            logger.critical(f"Network successfully moved to device: {self.device}")
        except Exception as e:
            logger.critical(f"Error moving network to device: {e}")
            import traceback
            logger.critical(f"Traceback: {traceback.format_exc()}")
        
        # Verify that all components are on the correct device
        device_issues = False
        for name, module in self.named_children():
            for param_name, param in module.named_parameters():
                if param.device != self.device:
                    logger.critical(f"Parameter {name}.{param_name} is on {param.device}, not {self.device}")
                    device_issues = True
        if not device_issues:
            logger.critical("All network parameters confirmed on correct device")
        
        # Log model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.critical(f"Network initialized with {total_params} total parameters ({trainable_params} trainable)")
        logger.critical("Network initialization complete")

    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            hidden_state: Optional hidden state for LSTM (h, c)
            
        Returns:
            Tuple: (action_probs, value, next_hidden_state)
        """
        try:
            logger.critical(f"Network forward called with input shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
            if torch.isnan(x).any():
                logger.critical("WARNING: Input contains NaN values!")
            if torch.isinf(x).any():
                logger.critical("WARNING: Input contains Inf values!")
                
            # Ensure input is on the correct device 
            if x.device != self.device:
                logger.critical(f"Moving input from {x.device} to {self.device}")
                x = x.to(self.device)
                
            # Handle batch dimension for visual inputs
            if self.is_visual_input:
                # Check if the input is already processed and is a feature vector (e.g., [1, 512])
                # This means we should skip the convolutional processing
                if len(x.shape) == 2:
                    logger.critical(f"Input appears to be a feature vector with shape {x.shape}, skipping conv layers")
                    features = x  # Use directly as features
                    
                    # Check if the feature vector already matches the output dimension of shared layers
                    if features.shape[1] == self.feature_size:
                        logger.critical(f"Feature vector already matches feature_size ({self.feature_size}), skipping shared layers")
                        # Skip shared layers since dimensions already match the expected output
                        # Continue with the rest of the network (LSTM, policy head, etc.)
                        pass
                    else:
                        # Pass through shared layers if dimensions don't match
                        logger.critical("Passing through shared layers")
                        features = self.shared_layers(features)
                        logger.critical(f"Shared features shape: {features.shape}")
                else:
                    # Check if batch dimension is missing
                    if len(x.shape) == 3:  # [C, H, W]
                        logger.critical("Adding batch dimension to input")
                        x = x.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
                    
                    # Ensure input has 4D shape for convolutional layers
                    if len(x.shape) == 4:
                        # Check if the input channels don't match what the first conv layer expects
                        expected_channels = self.conv_layers[0].in_channels
                        actual_channels = x.shape[1]
                        logger.critical(f"Passing through conv layers, input shape: {x.shape}, " +
                                       f"expected channels: {expected_channels}, actual channels: {actual_channels}")
                        
                        if actual_channels != expected_channels:
                            # This is likely a frame stack (e.g., 4 frames stacked with 3 channels each)
                            # We need to reshape or adapt the input
                            if actual_channels % expected_channels == 0:
                                # Case where we have a clean multiple (e.g., 12 = 4 frames * 3 channels)
                                frame_stack = actual_channels // expected_channels
                                logger.critical(f"Handling frame stack of {frame_stack} frames")
                                
                                batch_size = x.shape[0]
                                height = x.shape[2]
                                width = x.shape[3]
                                
                                # Option 1: Process each frame separately and average features
                                # Reshape to [batch_size * frame_stack, expected_channels, height, width]
                                reshaped_x = x.view(batch_size * frame_stack, expected_channels, height, width)
                                
                                # Process through conv layers
                                logger.critical(f"Processing reshaped input with shape: {reshaped_x.shape}")
                                reshaped_features = self.conv_layers(reshaped_x)
                                
                                # Reshape back and average across frames
                                reshaped_features = reshaped_features.view(batch_size, frame_stack, -1)
                                x = torch.mean(reshaped_features, dim=1)  # Average across frames
                                logger.critical(f"Averaged features shape: {x.shape}")
                                features = x
                            else:
                                # Fallback: take only the first 'expected_channels' channels
                                logger.critical(f"Channel mismatch that isn't a clean multiple. Using first {expected_channels} channels")
                                x = x[:, :expected_channels, :, :]
                                x = self.conv_layers(x)
                                x = x.reshape(x.size(0), -1)
                                features = x
                        else:
                            # Normal case - channels match
                            x = self.conv_layers(x)
                            logger.critical(f"Conv output shape: {x.shape}")
                            
                            # Flatten the features but keep batch dimension
                            x = x.reshape(x.size(0), -1)
                            logger.critical(f"Flattened shape: {x.shape}")
                            features = x
                    else:
                        logger.critical(f"Unexpected input shape {x.shape} for convolutional layers, using fallback")
                        # Create a fallback feature tensor of appropriate size
                        features = x  # Use as is if already a feature vector
            else:
                # For vector inputs
                if len(x.shape) == 1:  # [D]
                    logger.critical("Adding batch dimension to vector input")
                    x = x.unsqueeze(0)  # Add batch dimension -> [1, D]
                features = x
                    
            # Pass through shared layers
            if not (self.is_visual_input and len(x.shape) == 2 and features.shape[1] == self.feature_size):
                logger.critical("Passing through shared layers")
                features = self.shared_layers(features)
                logger.critical(f"Shared features shape: {features.shape}")
            else:
                logger.critical(f"Skipping shared layers, features already match expected size: {features.shape}")
            
            # Process through LSTM if enabled
            next_hidden = None
            if self.use_lstm:
                if hidden_state is None:
                    # Initialize hidden state if not provided
                    h0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                    c0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                    hidden_state = (h0, c0)
                else:
                    # Ensure hidden state is on the correct device
                    h, c = hidden_state
                    if h.device != self.device:
                        h = h.to(self.device)
                    if c.device != self.device:
                        c = c.to(self.device)
                    hidden_state = (h, c)
                
                logger.critical("Processing through LSTM")
                # Add time dimension for LSTM if needed
                if len(features.shape) == 2:  # [B, F]
                    features = features.unsqueeze(1)  # [B, 1, F]
                
                features, next_hidden = self.lstm(features, hidden_state)
                
                # Process through attention if enabled
                # Note: We keep the time dimension for attention
                if self.use_attention:
                    logger.critical("Processing through attention mechanism")
                    features = self.attention(features)
                
                # Remove time dimension if added and not using attention
                if features.size(1) == 1 and not self.use_attention:
                    features = features.squeeze(1)
                elif self.use_attention:
                    # If using attention, ensure we're using the right dimension
                    features = features.squeeze(1) if features.size(1) == 1 else features[:, -1]
                
                logger.critical(f"LSTM/Attention output shape: {features.shape}")
            
            # Get policy (action probabilities) and value
            logger.critical("Computing action logits")
            action_logits = self.policy_head(features)
            logger.critical(f"Action logits shape: {action_logits.shape}")
            
            # Check and fix NaN/Inf values in logits
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                logger.critical("WARNING: NaN or Inf values detected in action_logits, applying fix")
                # Replace NaN/Inf with zeros
                action_logits = torch.where(torch.isnan(action_logits) | torch.isinf(action_logits), 
                                           torch.zeros_like(action_logits), 
                                           action_logits)
            
            # Clip logits for numerical stability
            action_logits = torch.clamp(action_logits, min=-20.0, max=20.0)
            
            logger.critical("Computing value")
            value = self.value_head(features)
            logger.critical(f"Value shape: {value.shape}")
            
            # Check and fix NaN/Inf values in value
            if torch.isnan(value).any() or torch.isinf(value).any():
                logger.critical("WARNING: NaN or Inf values detected in value, applying fix")
                # Replace NaN/Inf with zeros
                value = torch.where(torch.isnan(value) | torch.isinf(value), 
                                   torch.zeros_like(value), 
                                   value)
            
            # Clip value for numerical stability
            value = torch.clamp(value, min=-100.0, max=100.0)
            
            # Apply softmax to get probabilities with additional numerical stability
            logger.critical("Applying softmax to get action probabilities")
            action_probs = F.softmax(action_logits, dim=-1)
            
            # If there are still NaN values after softmax, use uniform distribution
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                logger.critical("WARNING: Still detected NaN/Inf values after softmax, using uniform distribution")
                action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
                
            logger.critical(f"Action probabilities shape: {action_probs.shape}")
            
            # Check for NaN/Inf in outputs
            if torch.isnan(action_probs).any():
                logger.critical("WARNING: action_probs contains NaN values!")
            if torch.isinf(action_probs).any():
                logger.critical("WARNING: action_probs contains Inf values!")
            if torch.isnan(value).any():
                logger.critical("WARNING: value contains NaN values!")
            if torch.isinf(value).any():
                logger.critical("WARNING: value contains Inf values!")
            
            # Squeeze value if needed
            value = value.squeeze(-1)
            
            logger.critical("Forward pass completed successfully")
            return action_probs, value, next_hidden
            
        except Exception as e:
            import traceback
            logger.critical(f"ERROR in network forward pass: {e}")
            logger.critical(f"Traceback: {traceback.format_exc()}")
            
            # Try to return usable values with gradient support
            batch_size = x.size(0) if len(x.shape) > 1 else 1
            
            # Create parameters that require gradients
            if not hasattr(self, 'fallback_policy_param'):
                self.fallback_policy_param = nn.Parameter(
                    torch.zeros(self.policy_head[-1].out_features, device=self.device),
                    requires_grad=True
                )
                self.fallback_value_param = nn.Parameter(
                    torch.zeros(1, device=self.device),
                    requires_grad=True
                )
            
            # Create tensors that share storage with parameters (and thus have gradients)
            try:
                # Generate uniform probabilities that require gradients
                uniform_probs = F.softmax(
                    self.fallback_policy_param.expand(batch_size, -1),
                    dim=1
                )
                
                # Generate zero values that require gradients
                zero_value = self.fallback_value_param.expand(batch_size)
                
                logger.critical(f"Returning fallback values with shapes: probs={uniform_probs.shape}, value={zero_value.shape}")
                logger.critical(f"Fallback values require_grad: probs={uniform_probs.requires_grad}, value={zero_value.requires_grad}")
                
                return uniform_probs, zero_value, None
            except Exception as fallback_error:
                # Last resort fallback if even our fallback fails
                logger.critical(f"Error in fallback mechanism: {fallback_error}")
                # Create simple tensors with requires_grad=True
                uniform_probs = torch.ones(batch_size, self.policy_head[-1].out_features, device=self.device, requires_grad=True) / self.policy_head[-1].out_features
                zero_value = torch.zeros(batch_size, device=self.device, requires_grad=True)
                logger.critical("Using simple tensor fallback with requires_grad=True")
                return uniform_probs, zero_value, None
        
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
    
    def _init_weights(self, module):
        """Initialize network weights.
        
        Args:
            module: Module to initialize weights for
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0) 