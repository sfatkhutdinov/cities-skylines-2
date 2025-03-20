"""
Memory module for PPO agent in Cities: Skylines 2.

This module implements the experience memory component of the PPO agent,
storing experiences and providing data for training updates.
"""

import torch
import logging
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Iterator, Union
from collections import deque, namedtuple

logger = logging.getLogger(__name__)

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])

class Memory:
    """Implements the experience memory component for the PPO agent."""
    
    def __init__(self, device: torch.device, capacity: int = 10000):
        """Initialize memory component.
        
        Args:
            device: Device to store tensors on
            capacity: Maximum number of experiences to store
        """
        self.device = device
        self.capacity = capacity
        
        # Core memory storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        # Computed data
        self.returns = []
        self.advantages = []
        self.action_probs = []  # Full action probability distributions
        
        # Episode tracking
        self.current_episode = []
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        
        logger.info(f"Initialized memory with capacity {capacity}")
    
    def add(self, state: torch.Tensor, 
           action: torch.Tensor, 
           reward: float, 
           next_state: Optional[torch.Tensor], 
           done: bool, 
           log_prob: torch.Tensor, 
           value: torch.Tensor,
           action_probs: Optional[torch.Tensor] = None) -> None:
        """Add an experience to memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            log_prob: Log probability of action
            value: Value of state
            action_probs: Full action probability distribution
        """
        # Check if memory is full
        if len(self.states) >= self.capacity:
            # Remove oldest entries
            self._remove_oldest(1)
        
        # Convert to torch tensors if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        if next_state is not None and not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, device=self.device)
        if not isinstance(done, torch.Tensor):
            done = torch.tensor(done, device=self.device, dtype=torch.float32)
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob, device=self.device, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=self.device, dtype=torch.float32)
            
        # Store experience
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if next_state is not None:
            self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        
        # Store action probabilities if provided
        if action_probs is not None:
            self.action_probs.append(action_probs)
        
        # Add to current episode
        experience = Experience(state, action, reward, next_state, done, log_prob, value)
        self.current_episode.append(experience)
        
        # Check if episode ended
        if done:
            self._finish_episode()
    
    def add_batch(self, states: torch.Tensor, 
                 actions: torch.Tensor, 
                 rewards: torch.Tensor, 
                 next_states: torch.Tensor, 
                 dones: torch.Tensor, 
                 log_probs: torch.Tensor, 
                 values: torch.Tensor) -> None:
        """Add a batch of experiences to memory.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            log_probs: Batch of log probabilities
            values: Batch of state values
        """
        batch_size = len(states)
        
        # Check capacity
        if len(self.states) + batch_size > self.capacity:
            # Remove oldest entries to make room
            overflow = len(self.states) + batch_size - self.capacity
            self._remove_oldest(overflow)
            
        # Ensure all tensors are on the correct device
        if states.device != self.device:
            states = states.to(self.device)
        if actions.device != self.device:
            actions = actions.to(self.device)
        if rewards.device != self.device:
            rewards = rewards.to(self.device)
        if next_states.device != self.device:
            next_states = next_states.to(self.device)
        if dones.device != self.device:
            dones = dones.to(self.device)
        if log_probs.device != self.device:
            log_probs = log_probs.to(self.device)
        if values.device != self.device:
            values = values.to(self.device)
        
        # Add batch to memory
        for i in range(batch_size):
            self.states.append(states[i])
            self.actions.append(actions[i])
            self.rewards.append(rewards[i])
            if next_states is not None:
                self.next_states.append(next_states[i])
            self.dones.append(dones[i])
            self.log_probs.append(log_probs[i])
            self.values.append(values[i])
            
            # Update episode tracking
            if dones[i]:
                self._finish_episode()
    
    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float, next_value: Optional[torch.Tensor] = None) -> None:
        """Compute returns and advantages for the stored experiences.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            next_value: Value of the state after the last experience
        """
        # Ensure we have experiences to process
        if not self.states:
            logger.warning("No experiences to compute returns and advantages for")
            return
        
        # If next_value not provided, use zero
        if next_value is None:
            next_value = torch.zeros(1, device=self.device)
            
        # Convert rewards and dones to tensors
        rewards = torch.stack(self.rewards).to(self.device)
        dones = torch.stack(self.dones).to(self.device)
        values = torch.stack(self.values).to(self.device)
        
        # Compute GAE advantages and returns
        batch_size = len(rewards)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Initialize with next value
        next_value = next_value
        next_advantage = 0.0
        
        # Compute advantages and returns backwards
        for t in reversed(range(batch_size)):
            # For last timestep, use provided next_value
            if t == batch_size - 1:
                next_state_value = next_value
            else:
                # For other timesteps, use value of next state in batch
                next_state_value = values[t + 1]
                
            # Compute delta (TD error)
            delta = rewards[t] + gamma * next_state_value * (1 - dones[t]) - values[t]
            
            # Compute advantage (GAE)
            advantages[t] = delta + gamma * gae_lambda * next_advantage * (1 - dones[t])
            next_advantage = advantages[t]
            
            # Compute returns
            returns[t] = advantages[t] + values[t]
            
        # Store computed returns and advantages
        self.returns = list(returns)
        self.advantages = list(advantages)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            self.advantages = list(advantages)
    
    def get_batch_iterator(self, batch_size: int, shuffle: bool = True) -> Iterator[Dict[str, torch.Tensor]]:
        """Get iterator over batches of experiences.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle experiences
            
        Returns:
            Iterator over batches
        """
        # Check if we have data to return
        if not self.states or len(self.states) == 0:
            logger.warning("No experiences in memory")
            return iter([])
            
        # Get indices of all experiences
        indices = list(range(len(self.states)))
        
        # Shuffle if requested
        if shuffle:
            np.random.shuffle(indices)
            
        # Ensure returns and advantages are computed
        if not self.returns or len(self.returns) != len(self.states):
            logger.warning("Returns not computed, computing with default parameters")
            self.compute_returns_and_advantages(0.99, 0.95)
            
        # Create batches
        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # Collect batch data
            states_batch = torch.stack([self.states[i] for i in batch_indices])
            actions_batch = torch.stack([self.actions[i] for i in batch_indices])
            old_log_probs_batch = torch.stack([self.log_probs[i] for i in batch_indices])
            returns_batch = torch.stack([self.returns[i] for i in batch_indices])
            advantages_batch = torch.stack([self.advantages[i] for i in batch_indices])
            values_batch = torch.stack([self.values[i] for i in batch_indices])
            
            # Build batch dict
            batch = {
                'states': states_batch,
                'actions': actions_batch,
                'old_log_probs': old_log_probs_batch,
                'returns': returns_batch,
                'advantages': advantages_batch,
                'values': values_batch
            }
            
            # Add action_probs if available
            if self.action_probs and len(self.action_probs) == len(self.states):
                batch['old_action_probs'] = torch.stack([self.action_probs[i] for i in batch_indices])
                
            yield batch
    
    def _remove_oldest(self, count: int) -> None:
        """Remove oldest experiences from memory.
        
        Args:
            count: Number of experiences to remove
        """
        if count <= 0:
            return
            
        # Ensure we don't remove more than we have
        count = min(count, len(self.states))
        
        # Remove oldest experiences
        self.states = self.states[count:]
        self.actions = self.actions[count:]
        self.rewards = self.rewards[count:]
        if self.next_states:
            self.next_states = self.next_states[count:]
        self.dones = self.dones[count:]
        self.log_probs = self.log_probs[count:]
        self.values = self.values[count:]
        
        # Remove from computed data too
        if self.returns:
            self.returns = self.returns[count:]
        if self.advantages:
            self.advantages = self.advantages[count:]
        if self.action_probs:
            self.action_probs = self.action_probs[count:]
    
    def _finish_episode(self) -> None:
        """Process the end of an episode and update statistics."""
        if not self.current_episode:
            return
            
        # Calculate episode total reward and length
        total_reward = sum(exp.reward.item() if isinstance(exp.reward, torch.Tensor) else exp.reward 
                          for exp in self.current_episode)
        episode_length = len(self.current_episode)
        
        # Update statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        # Log episode statistics
        logger.debug(f"Episode finished: length={episode_length}, reward={total_reward:.2f}")
        
        # Clear current episode
        self.current_episode = []
    
    def clear(self) -> None:
        """Clear all stored experiences."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.returns = []
        self.advantages = []
        self.action_probs = []
        self.current_episode = []
        
    def size(self) -> int:
        """Get number of experiences in memory.
        
        Returns:
            Number of experiences
        """
        return len(self.states)
    
    def get_episode_stats(self) -> Dict[str, float]:
        """Get episode statistics.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'mean_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'last_episode_reward': self.episode_rewards[-1] if self.episode_rewards else 0.0,
            'last_episode_length': self.episode_lengths[-1] if self.episode_lengths else 0,
            'num_episodes': len(self.episode_rewards)
        } 