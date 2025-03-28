"""
Memory-Augmented Agent for Cities: Skylines 2.
Extends PPO agent with episodic memory capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque

from src.agent.core.ppo_agent import PPOAgent
from src.memory.memory_augmented_network import MemoryAugmentedNetwork
from src.memory.episodic_memory import MANNController

logger = logging.getLogger(__name__)

class MemoryAugmentedAgent(PPOAgent):
    """PPO agent extended with episodic memory capabilities."""
    
    def __init__(self,
                 policy_network,
                 observation_space,
                 action_space,
                 device: torch.device = None,
                 memory_size: int = 1000,
                 memory_use_prob: float = 0.8,  # Probability of using memory
                 **kwargs):
        """Initialize the memory-augmented agent.
        
        Args:
            policy_network: Policy network (MemoryAugmentedNetwork)
            observation_space: Observation space
            action_space: Action space
            device: Computation device
            memory_size: Maximum number of memories to store
            memory_use_prob: Probability of using memory during inference
            **kwargs: Additional arguments for PPOAgent
        """
        if not isinstance(policy_network, MemoryAugmentedNetwork):
            logger.critical("Converting policy network to MemoryAugmentedNetwork")
            # Create memory-augmented network with the provided network as base
            input_shape = observation_space.shape
            num_actions = action_space.n if hasattr(action_space, 'n') else action_space
            
            memory_network = MemoryAugmentedNetwork(
                input_shape=input_shape,
                num_actions=num_actions,
                memory_size=memory_size,
                device=device,
                use_lstm=getattr(policy_network, 'use_lstm', True),
                lstm_hidden_size=getattr(policy_network, 'lstm_hidden_size', 256),
                use_attention=getattr(policy_network, 'use_attention', True),
                attention_heads=getattr(policy_network, 'attention_heads', 4)
            )
            policy_network = memory_network
        
        # Extract state_dim and action_dim for PPOAgent
        if hasattr(observation_space, 'shape'):
            state_dim = observation_space.shape
        else:
            state_dim = observation_space
            
        if hasattr(action_space, 'n'):
            action_dim = action_space.n
        else:
            action_dim = action_space
            
        # Create a hardware config with the device if not already in kwargs
        from src.config.hardware_config import HardwareConfig
        config = kwargs.pop('config', None)
        if config is None:
            config = HardwareConfig()
            if device is not None:
                config.device = str(device).split(':')[0]  # Extract 'cuda' or 'cpu' from the device

        # Initialize base PPO agent with the appropriate parameters
        super().__init__(state_dim, action_dim, config, **kwargs)
        
        # Store the policy network
        self.policy = policy_network
        
        self.memory_use_prob = memory_use_prob
        self.memory_enabled = True
        self.memory_stats = {
            "writes": 0,
            "reads": 0,
            "hits": 0,
            "important_experiences": 0
        }
        
        self.experience_buffer = deque(maxlen=5)  # Short-term buffer for recent experiences
        
        logger.critical(f"Initialized memory-augmented agent with memory size {memory_size}")
    
    def select_action(self, state, deterministic=False, info=None):
        """Select an action based on current state and memory.
        
        Args:
            state: Current state observation
            deterministic: Whether to select the highest probability action
            info: Additional information from environment for action selection
            
        Returns:
            Dictionary containing:
                - action: Selected action tensor
                - log_prob: Log probability of selected action
                - value: Value estimate for current state
                - action_probs: Action probabilities tensor
        """
        try:
            with torch.no_grad():
                # Ensure state is a tensor and on the correct device
                if not isinstance(state, torch.Tensor):
                    state = torch.as_tensor(state, device=self.device).float()
                elif state.device != self.device:
                    state = state.to(self.device)

                # Add batch dim if missing
                if state.dim() == 3:  # [C, H, W]
                    state = state.unsqueeze(0)

                # Get memory context if enabled
                memory_context = None
                if self.memory_enabled and self.policy.memory_network is not None:
                    memory_context = self.policy.memory_network.get_context(state)

                # Forward pass through policy network with memory context
                action_probs, value, next_hidden = self.policy(state, self.hidden_state, memory_context)
                
                # Update hidden state for next time
                self.hidden_state = next_hidden
                
                # Process environment info if provided
                if info is not None:
                    self.process_step_info(info)
                    
                    # Track if we're currently in a menu
                    self.in_menu = info.get('in_menu', False)
                
                # Apply menu action penalties
                action_probs = self.adjust_action_probs(action_probs)
                
                # Apply action smoothing
                action_probs = self._apply_action_smoothing(action_probs)
                
                # Create categorical distribution
                action_distribution = torch.distributions.Categorical(action_probs)
                
                # Choose action
                if deterministic:
                    action = torch.argmax(action_probs, dim=1)
                else:
                    action = action_distribution.sample()
                
                # Store action in history for future smoothing
                action_item = action.cpu().item()
                self._update_action_history(action_item)
                
                # Store current value and log probability for update
                self.last_state = state
                self.last_action_tensor = action  # Store the tensor action for potential updates
                self.last_action = action_item  # Store the int action for history/logging
                self.last_value = value
                self.last_log_prob = action_distribution.log_prob(action)
                
                # Update memory if enabled
                if self.memory_enabled and self.policy.memory_network is not None:
                    self.policy.memory_network.update(state, action, value)
                
                # Return standardized dictionary
                return {
                    'action': action,  # Return the tensor action
                    'log_prob': self.last_log_prob,
                    'value': self.last_value,
                    'action_probs': action_probs  # Include probs if needed elsewhere
                }
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            # Fallback
            random_action = torch.randint(0, self.action_dim, (1,), device=self.device)
            return {
                'action': random_action,
                'log_prob': torch.tensor(0.0, device=self.device),
                'value': torch.tensor(0.0, device=self.device),
                'action_probs': torch.ones(1, self.action_dim, device=self.device) / self.action_dim
            }
    
    def process_experience(self, state, action, reward, next_state, done, info=None):
        """Process experience to potentially store in episodic memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            bool: Whether the experience was stored in memory
        """
        try:
            # Get state embedding if not already available
            if 'state_embedding' in info and info['state_embedding'] is not None:
                state_embedding = info['state_embedding']
            else:
                if isinstance(state, torch.Tensor):
                    state_tensor = state
                else:
                    state_tensor = torch.tensor(state, device=self.device).float()
                    
                try:
                    state_embedding = self.policy.extract_state_embedding(state_tensor, self.hidden_state)
                except Exception as e:
                    logger.error(f"Error extracting state embedding in process_experience: {e}")
                    state_embedding = None
            
            # Check if we should store this experience
            if state_embedding is not None:
                should_store, importance = self.policy.should_store_memory(state_embedding, reward, done)
                
                if should_store:
                    # Prepare memory data
                    memory_data = {
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done,
                        'info': info
                    }
                    
                    # Store in memory
                    success = self.policy.store_memory(state_embedding, memory_data, importance)
                    
                    if success:
                        self.memory_stats["writes"] += 1
                        if importance > 0.7:
                            self.memory_stats["important_experiences"] += 1
                        
                        logger.info(f"Stored experience with importance {importance:.2f}")
                        return True
            
            # Add to short-term buffer regardless
            self.experience_buffer.append((state_embedding, {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'info': info
            }))
            
            return False
                
        except Exception as e:
            logger.error(f"Error processing experience: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_memory_stats(self):
        """Get memory statistics.
        
        Returns:
            Dict of memory statistics
        """
        # Get stats from the memory controller
        controller_stats = self.policy.get_memory_stats()
        
        # Combine with agent stats
        return {**self.memory_stats, **controller_stats}
    
    def enable_memory(self, enabled=True):
        """Enable or disable memory usage.
        
        Args:
            enabled: Whether to enable memory
        """
        self.memory_enabled = enabled
        logger.info(f"Memory usage {'enabled' if enabled else 'disabled'}")
    
    def set_memory_use_probability(self, prob):
        """Set the probability of using memory during inference.
        
        Args:
            prob: Probability of using memory (0.0 to 1.0)
        """
        self.memory_use_prob = max(0.0, min(1.0, prob))
        logger.info(f"Memory use probability set to {self.memory_use_prob}")
    
    def reset(self):
        """Reset agent state between episodes."""
        # Call parent reset method (handles LSTM state)
        super().reset()
        
        # Process experience buffer at the end of an episode
        # Store any experiences that were interesting but not immediately stored
        if len(self.experience_buffer) > 0:
            # Sort by reward to find most important
            sorted_experiences = sorted(
                self.experience_buffer, 
                key=lambda x: x[1]['reward'] if 'reward' in x[1] else 0,
                reverse=True
            )
            
            # Store the highest rewarded experience if not empty
            if len(sorted_experiences) > 0:
                best_experience = sorted_experiences[0]
                embedding, data = best_experience
                
                # Store in memory with medium importance
                self.policy.store_memory(embedding, data, importance=0.7)
                self.memory_stats["writes"] += 1
                logger.info("Stored best experience from buffer at episode end")
            
            # Clear the buffer
            self.experience_buffer.clear()
            
        logger.info("Memory agent reset for new episode") 