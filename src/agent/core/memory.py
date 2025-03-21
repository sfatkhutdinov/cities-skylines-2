"""
Memory module for PPO agent in Cities: Skylines 2.

This module implements the memory system for the PPO agent,
handling experience storage, retrieval, and processing.
"""

import torch
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Iterator
from collections import deque

logger = logging.getLogger(__name__)

class Memory:
    """Manages experience memory for the PPO agent."""
    
    def __init__(self, device: torch.device):
        """Initialize memory component.
        
        Args:
            device: Device to store tensors on
        """
        self.device = device
        
        # Store experiences
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.action_probs = []
        
        # Store computed data
        self.returns = []
        self.advantages = []
        
        # Statistics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        # Initialize empty
        self.clear()
        
        logger.info(f"Initialized memory on device: {device}")
    
    def add(self, 
            state: torch.Tensor, 
            action: Any, 
            reward: float, 
            next_state: Optional[torch.Tensor], 
            done: bool, 
            log_prob: Optional[torch.Tensor] = None, 
            value: Optional[torch.Tensor] = None,
            action_probs: Optional[torch.Tensor] = None) -> None:
        """Add an experience to memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            log_prob: Log probability of action
            value: Value estimate
            action_probs: Action probability distribution
        """
        # Log critical information about what's being added to memory
        logger.critical(f"Adding experience: action={action}, reward={reward}, done={done}")
        
        # Ensure state is a tensor on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, device=self.device).float()
        elif state.device != self.device:
            state = state.to(self.device)
        
        # Ensure next_state is a tensor on the correct device
        if next_state is not None:
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.as_tensor(next_state, device=self.device).float()
            elif next_state.device != self.device:
                next_state = next_state.to(self.device)
        
        # Store the experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state if next_state is not None else state.clone())  # Use current state as fallback
        self.dones.append(done)
        
        # Store optional data if provided
        if log_prob is not None:
            if not isinstance(log_prob, torch.Tensor):
                log_prob = torch.tensor(log_prob, device=self.device).float()
            elif log_prob.device != self.device:
                log_prob = log_prob.to(self.device)
            self.log_probs.append(log_prob)
        else:
            # Use a default value for log probability
            self.log_probs.append(torch.tensor(0.0, device=self.device))
        
        if value is not None:
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device).float()
            elif value.device != self.device:
                value = value.to(self.device)
            self.values.append(value)
        else:
            # Use a default value for value estimate
            self.values.append(torch.tensor(0.0, device=self.device))
        
        if action_probs is not None:
            if not isinstance(action_probs, torch.Tensor):
                action_probs = torch.tensor(action_probs, device=self.device).float()
            elif action_probs.device != self.device:
                action_probs = action_probs.to(self.device)
            self.action_probs.append(action_probs)
        
        # Track episode statistics
        if done:
            current_episode_reward = sum(self.rewards[-1:])
            current_episode_length = 1
            self.episode_rewards.append(current_episode_reward)
            self.episode_lengths.append(current_episode_length)
            logger.info(f"Episode complete: reward={current_episode_reward:.2f}, length={current_episode_length}")
    
    def compute_returns(self, gamma: float, gae_lambda: float) -> None:
        """Compute returns and advantages using GAE.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE factor
        """
        if len(self.states) == 0:
            logger.warning("No experiences in memory to compute returns")
            return
            
        # Log critical information about the computation
        logger.critical(f"Computing returns and advantages: gamma={gamma}, gae_lambda={gae_lambda}, experiences={len(self.states)}")
        
        # Convert lists to tensors for faster computation
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.cat(self.values).to(self.device) if all(isinstance(v, torch.Tensor) for v in self.values) else torch.tensor(self.values, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        
        # Check for NaN or Inf values in rewards and replace them with zeros
        invalid_reward_mask = torch.isnan(rewards) | torch.isinf(rewards)
        if invalid_reward_mask.any():
            logger.critical(f"Found {invalid_reward_mask.sum().item()} NaN/Inf reward values. Replacing with zeros.")
            rewards = torch.where(invalid_reward_mask, torch.zeros_like(rewards), rewards)
        
        # Check for NaN or Inf values in values and replace them with zeros
        invalid_value_mask = torch.isnan(values) | torch.isinf(values)
        if invalid_value_mask.any():
            logger.critical(f"Found {invalid_value_mask.sum().item()} NaN/Inf value estimates. Replacing with zeros.")
            values = torch.where(invalid_value_mask, torch.zeros_like(values), values)
        
        # Initialize returns and advantages
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Initialize values for GAE computation
        next_value = 0.0  # Assume zero value after end of trajectory
        next_advantage = 0.0
        
        # Compute returns and advantages in reverse order
        for t in reversed(range(len(rewards))):
            # If at the end of an episode, next values are zero
            next_non_terminal = 1.0 - dones[t]
            
            # Compute TD target for return
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            
            # Compute return using GAE
            returns[t] = rewards[t] + gamma * next_non_terminal * next_value
            
            # Compute advantage using GAE
            advantages[t] = delta + gamma * gae_lambda * next_non_terminal * next_advantage
            
            # Update next values
            next_value = values[t]
            next_advantage = advantages[t]
        
        # Normalize advantages
        if len(advantages) > 1:
            # Check for NaN or Inf values before normalization
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                logger.critical("NaN or Inf detected in advantages before normalization. Resetting affected values.")
                advantages = torch.where(torch.isnan(advantages) | torch.isinf(advantages), 
                                        torch.zeros_like(advantages), advantages)
            
            # Safely normalize with epsilon to avoid div by zero
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Final safety check for NaN/Inf values
        returns = torch.where(torch.isnan(returns) | torch.isinf(returns), 
                             torch.zeros_like(returns), returns)
        advantages = torch.where(torch.isnan(advantages) | torch.isinf(advantages), 
                                torch.zeros_like(advantages), advantages)
        
        # Store computed values and detach for gradient computation
        self.returns = returns.detach().cpu().tolist()
        self.advantages = advantages.detach().cpu().tolist()
        
        logger.critical(f"Returns computed: min={min(self.returns):.2f}, max={max(self.returns):.2f}, mean={sum(self.returns)/len(self.returns):.2f}")
        logger.critical(f"Advantages computed: min={min(self.advantages):.2f}, max={max(self.advantages):.2f}, mean={sum(self.advantages)/len(self.advantages):.2f}")
    
    def get_batch_iterator(self, batch_size: int, shuffle: bool = True) -> Iterator[Dict[str, torch.Tensor]]:
        """Get iterator for batches of experiences.
        
        Args:
            batch_size: Size of batches
            shuffle: Whether to shuffle the data
            
        Returns:
            Iterator of batches
        """
        if len(self.states) == 0:
            logger.warning("No experiences in memory to iterate over")
            return iter([])
            
        logger.critical(f"Creating batch iterator: batch_size={batch_size}, experiences={len(self.states)}, shuffle={shuffle}")
        
        # Convert lists to tensors
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=self.device)
        log_probs = torch.stack(self.log_probs) if all(isinstance(lp, torch.Tensor) for lp in self.log_probs) else torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)
        values = torch.stack(self.values) if all(isinstance(v, torch.Tensor) for v in self.values) else torch.tensor(self.values, dtype=torch.float32, device=self.device)
        
        # Check for NaN values and replace with zeros
        for tensor_name, tensor in [
            ('returns', returns), 
            ('advantages', advantages), 
            ('log_probs', log_probs), 
            ('values', values)
        ]:
            invalid_mask = torch.isnan(tensor) | torch.isinf(tensor)
            if invalid_mask.any():
                logger.critical(f"Found {invalid_mask.sum().item()} NaN/Inf values in {tensor_name}. Replacing with zeros.")
                if tensor_name == 'log_probs':
                    # For log probs, use a very small probability instead of zero
                    replacement = torch.ones_like(tensor) * -10.0  # log(~4.5e-5)
                else:
                    replacement = torch.zeros_like(tensor)
                tensor.copy_(torch.where(invalid_mask, replacement, tensor))
        
        # Create dataset size
        dataset_size = len(self.states)
        
        # Create indices
        indices = torch.randperm(dataset_size) if shuffle else torch.arange(dataset_size)
        
        # Create batches
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            # Extract batch data
            batch = {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'old_log_probs': log_probs[batch_indices],
                'returns': returns[batch_indices],
                'advantages': advantages[batch_indices],
                'old_values': values[batch_indices]
            }
            
            logger.critical(f"Yielding batch: size={len(batch_indices)}, indices={batch_indices[:5].tolist()}...")
            yield batch
    
    def clear(self) -> None:
        """Clear memory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.action_probs = []
        self.returns = []
        self.advantages = []
        
        logger.critical("Memory cleared")
    
    def size(self) -> int:
        """Get the number of experiences in memory.
        
        Returns:
            int: Number of experiences
        """
        return len(self.states)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Dict with statistics
        """
        return {
            'experiences': len(self.states),
            'avg_episode_reward': sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_episode_length': sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0,
            'total_episodes': len(self.episode_rewards)
        }
    
    def __len__(self) -> int:
        """Get the number of experiences in memory.
        
        Returns:
            int: Number of experiences
        """
        return len(self.states) 