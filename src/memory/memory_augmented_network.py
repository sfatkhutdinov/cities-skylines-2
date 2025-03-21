"""
Memory-Augmented Neural Network (MANN) for Cities: Skylines 2 agent.
Integrates episodic memory with existing neural network architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

from src.model.optimized_network import OptimizedNetwork
from src.memory.episodic_memory import MANNController

logger = logging.getLogger(__name__)

class MemoryAugmentedNetwork(nn.Module):
    """Memory-Augmented Neural Network combining episodic memory with policy network."""
    
    def __init__(self,
                 input_shape,
                 num_actions,
                 memory_size: int = 1000,
                 device=None,
                 use_lstm=True,
                 lstm_hidden_size=256,
                 use_attention=True,
                 attention_heads=4):
        """Initialize the MANN.
        
        Args:
            input_shape: Shape of the input state
            num_actions: Number of actions in the action space
            memory_size: Maximum number of memories to store
            device: Computation device
            use_lstm: Whether to use LSTM
            lstm_hidden_size: Size of LSTM hidden state
            use_attention: Whether to use attention
            attention_heads: Number of attention heads
        """
        super(MemoryAugmentedNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.critical(f"Initializing Memory-Augmented Network on device {self.device}")
        
        # Create base network
        self.base_network = OptimizedNetwork(
            input_shape=input_shape,
            num_actions=num_actions,
            device=device,
            use_lstm=use_lstm,
            lstm_hidden_size=lstm_hidden_size,
            use_attention=use_attention,
            attention_heads=attention_heads
        )
        
        # Determine embedding size for memory (using LSTM hidden size or feature size)
        self.embedding_size = lstm_hidden_size if use_lstm else self.base_network.feature_size
        logger.critical(f"Using embedding size {self.embedding_size} for memory")
        
        # Create MANN controller
        self.memory_controller = MANNController(
            embedding_size=self.embedding_size,
            output_size=self.embedding_size,  # Match size for residual connection
            memory_size=memory_size,
            device=self.device
        )
        
        # Memory integration gate (controls how much memory influences the output)
        self.memory_gate = nn.Sequential(
            nn.Linear(self.embedding_size * 2, 128),
            nn.Sigmoid(),
            nn.Linear(128, 1)
        ).to(self.device)
        
        # Memory importance predictor (for determining which memories to store)
        self.importance_predictor = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Track memory usage
        self.memory_usage_counter = 0
        self.total_steps = 0
        
        # Important experiences to remember (configurations)
        self.important_experience_types = {
            "reward_threshold": 0.5,  # Remember experiences with high rewards
            "novel_state_threshold": 0.7,  # Remember novel states
            "regular_sampling_interval": 100  # Remember states at regular intervals
        }
        
        self.to(self.device)
        logger.critical(f"Memory-Augmented Network initialized with memory size {memory_size}")
        
    def forward(self, x: torch.Tensor, 
               hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
               use_memory: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the network.
        
        Args:
            x: Input state tensor
            hidden_state: Optional hidden state for LSTM
            use_memory: Whether to use episodic memory
            
        Returns:
            Tuple of (action_probs, value, next_hidden_state)
        """
        # Process through base network first
        action_probs, value, next_hidden = self.base_network(x, hidden_state)
        
        # Extract features/embeddings from appropriate layer
        if self.base_network.use_lstm and next_hidden is not None:
            # Use LSTM hidden state as the state embedding
            state_embedding = next_hidden[0].squeeze(0)  # Use h, not c
        else:
            # If not using LSTM, extract features from the shared layers
            with torch.no_grad():
                if self.base_network.is_visual_input:
                    # For visual inputs
                    if len(x.shape) == 3:  # [C, H, W]
                        x_batched = x.unsqueeze(0)
                    else:
                        x_batched = x
                        
                    features = self.base_network.conv_layers(x_batched)
                    features = features.reshape(x_batched.size(0), -1)
                else:
                    # For vector inputs
                    if len(x.shape) == 1:
                        x_batched = x.unsqueeze(0)
                    else:
                        x_batched = x
                    
                features = self.base_network.shared_layers(x_batched)
                state_embedding = features.squeeze(0)  # Remove batch dimension
        
        # Only use memory if explicitly enabled and we have enough steps
        if use_memory and self.total_steps > 100:  # Allow some warm-up time
            try:
                # Get memory-augmented features
                memory_output = self.memory_controller(state_embedding)
                
                # Calculate gate value to determine influence of memory
                # (allows the network to decide how much to rely on memory)
                gate_input = torch.cat([state_embedding, memory_output], dim=0)
                gate_value = torch.sigmoid(self.memory_gate(gate_input))
                
                # Apply gate to mix original features with memory-augmented features
                memory_influence = gate_value * memory_output
                
                # Add memory influence to original embeddings (residual connection)
                augmented_embedding = state_embedding + memory_influence
                
                # Process augmented embedding through the base network's policy and value heads
                # We'll need to handle this separately because we don't want to re-run the entire network
                
                # For now, just use the original outputs
                # In a more advanced implementation, we would modify the action_probs and value based on memory
                
                self.memory_usage_counter += 1
            except Exception as e:
                logger.error(f"Error in memory augmentation: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # In case of error, just use the base network's outputs
        
        self.total_steps += 1
        return action_probs, value, next_hidden
    
    def should_store_memory(self, state_embedding: torch.Tensor, reward: float, done: bool) -> Tuple[bool, float]:
        """Decide whether to store the current state in memory.
        
        Args:
            state_embedding: Embedding of the current state
            reward: Reward received
            done: Whether the episode is done
            
        Returns:
            Tuple of (should_store, importance)
        """
        # Always remember episode end states
        if done:
            return True, 1.0
            
        # Remember states with high rewards
        if reward > self.important_experience_types["reward_threshold"]:
            return True, min(1.0, reward)
            
        # Use predictor to determine state novelty/importance
        with torch.no_grad():
            predicted_importance = self.importance_predictor(state_embedding).item()
            
        # Remember novel states
        if predicted_importance > self.important_experience_types["novel_state_threshold"]:
            return True, predicted_importance
            
        # Regular sampling
        if self.total_steps % self.important_experience_types["regular_sampling_interval"] == 0:
            return True, 0.5
        
        return False, 0.0
    
    def store_memory(self, state_embedding: torch.Tensor, 
                    memory_data: Dict[str, Any], 
                    importance: float = 1.0) -> bool:
        """Store a memory.
        
        Args:
            state_embedding: Embedding of the current state
            memory_data: Additional data to store with the memory
            importance: Importance score for the memory
            
        Returns:
            bool: Success of storage operation
        """
        return self.memory_controller.write_memory(state_embedding, memory_data, importance)
    
    def extract_state_embedding(self, x: torch.Tensor, 
                               hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """Extract state embedding for memory operations.
        
        Args:
            x: Input state tensor
            hidden_state: Optional hidden state for LSTM
            
        Returns:
            torch.Tensor: State embedding
        """
        # Extract features/embeddings from appropriate layer
        with torch.no_grad():
            if self.base_network.use_lstm:
                # Run forward pass to get hidden state
                _, _, next_hidden = self.base_network(x, hidden_state)
                # Use LSTM hidden state as the state embedding
                state_embedding = next_hidden[0].squeeze(0)  # Use h, not c
            else:
                # If not using LSTM, extract features from the shared layers
                if self.base_network.is_visual_input:
                    # For visual inputs
                    if len(x.shape) == 3:  # [C, H, W]
                        x_batched = x.unsqueeze(0)
                    else:
                        x_batched = x
                        
                    features = self.base_network.conv_layers(x_batched)
                    features = features.reshape(x_batched.size(0), -1)
                else:
                    # For vector inputs
                    if len(x.shape) == 1:
                        x_batched = x.unsqueeze(0)
                    else:
                        x_batched = x
                    
                features = self.base_network.shared_layers(x_batched)
                state_embedding = features.squeeze(0)  # Remove batch dimension
                
        return state_embedding
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dict of memory statistics
        """
        controller_stats = self.memory_controller.get_stats()
        
        # Add MANN-specific stats
        mann_stats = {
            "memory_usage_rate": self.memory_usage_counter / max(1, self.total_steps),
            "total_steps": self.total_steps
        }
        
        return {**controller_stats, **mann_stats}
        
    def clear_memory(self) -> None:
        """Clear all memories."""
        self.memory_controller.clear_memory()
        logger.info("Memory cleared") 