"""
PPO agent for Cities: Skylines 2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any
from model.optimized_network import OptimizedNetwork

class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, config):
        """Initialize PPO agent.
        
        Args:
            config: Hardware and training configuration
        """
        self.config = config
        self.device = config.get_device()
        
        # Initialize network
        self.network = OptimizedNetwork(config)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )
        
        # Initialize memory buffers
        self.states = []
        self.actions = []
        self.action_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        # Add action avoidance for menu toggling
        self.menu_action_indices = []  # Will be populated with indices of actions that open menus
        self.menu_action_penalties = {}  # Maps action indices to penalties
        self.menu_penalty_decay = 0.98  # Slower decay rate (was 0.95)
        self.last_rewards = []  # Track recent rewards to detect large penalties
        self.extreme_penalty_threshold = -500.0  # Threshold for detecting extreme penalties
        
    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action using the current policy.
        
        Args:
            state (torch.Tensor): Current environment state
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (action, log_prob, value)
        """
        # Ensure state is on the correct device
        if state.device != self.device:
            state = state.to(self.device)
            
        # Get action probabilities and value
        with torch.no_grad():
            action_probs, value = self.network(state)
            
            # Apply penalties to menu-opening actions
            if self.menu_action_indices and hasattr(self, 'menu_action_penalties'):
                # Create a penalty mask (default 1.0 for all actions)
                penalty_mask = torch.ones_like(action_probs)
                
                # Apply specific penalties to known menu actions
                for action_idx, penalty in self.menu_action_penalties.items():
                    if 0 <= action_idx < len(penalty_mask):
                        # Reduce probability of this action by the penalty factor
                        penalty_mask[action_idx] = max(0.1, 1.0 - penalty)  # Don't completely eliminate
                
                # Apply the mask to reduce probabilities of problematic actions
                action_probs = action_probs * penalty_mask
                
                # Renormalize probabilities
                action_probs = action_probs / action_probs.sum()
            
        # Sample action from probability distribution
        action = torch.multinomial(action_probs, 1)
        
        # Handle tensor shape for proper indexing
        if action_probs.dim() == 1:
            log_prob = torch.log(action_probs[action[0]]).unsqueeze(0)
        else:
            log_prob = torch.log(action_probs[0, action[0]]).unsqueeze(0)
        
        # Store experience
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_probs)
        self.values.append(value)
        
        return action, log_prob, value
        
    def update(self) -> Dict[str, float]:
        """Update the agent using collected experience.
        
        Returns:
            dict: Training metrics
        """
        # Check if we have any experience to learn from
        if not self.states or not self.actions:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
            
        # Convert lists to tensors
        states = torch.cat(self.states)
        actions = torch.cat(self.actions)
        
        # Check if we have action probabilities
        if not self.action_probs:
            self._clear_memory()
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
            
        old_action_probs = torch.cat(self.action_probs)
        values = torch.cat(self.values)
        
        # Compute returns and advantages
        returns = self._compute_returns()
        
        # Check if returns is empty
        if returns.numel() == 0:
            self._clear_memory()
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
            
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.config.ppo_epochs):
            # Get current action probabilities and values
            action_probs, value_preds = self.network(states)
            
            # Compute ratio of new and old action probabilities
            ratio = action_probs / old_action_probs
            
            # Compute PPO loss terms
            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
            ).mean()
            
            value_loss = 0.5 * (returns - value_preds).pow(2).mean()
            
            # Compute entropy bonus
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=-1).mean()
            
            # Total loss
            loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * entropy
            )
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
        # Clear memory
        self._clear_memory()
        
        # Return metrics
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
        
    def _compute_returns(self) -> torch.Tensor:
        """Compute returns using GAE.
        
        Returns:
            torch.Tensor: Computed returns
        """
        # Check if we have any rewards
        if not self.rewards:
            return torch.tensor([], device=self.device)
            
        rewards = torch.tensor(self.rewards, device=self.device)
        dones = torch.tensor(self.dones, device=self.device)
        values = torch.cat(self.values)
        
        # Ensure we have at least 2 values for GAE calculation
        if len(values) < 2:
            return rewards
        
        returns = []
        gae = 0
        for r, d, v, next_v in zip(
            reversed(rewards),
            reversed(dones),
            reversed(values[:-1]),
            reversed(values[1:])
        ):
            delta = r + self.config.gamma * next_v * (1 - d) - v
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - d) * gae
            returns.insert(0, gae + v)
            
        return torch.tensor(returns, device=self.device)
        
    def _clear_memory(self):
        """Clear experience memory buffers."""
        self.states = []
        self.actions = []
        self.action_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def save(self, path: str):
        """Save agent state.
        
        Args:
            path (str): Path to save state to
        """
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        """Load agent state.
        
        Args:
            path (str): Path to load state from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
    def register_menu_action(self, action_idx: int, penalty: float = 0.5):
        """Register an action that led to a menu state to discourage its selection.
        
        Args:
            action_idx (int): Index of the action that led to a menu
            penalty (float): Penalty factor (0-1) to apply to this action's probability
        """
        if not hasattr(self, 'menu_action_indices'):
            self.menu_action_indices = []
            self.menu_action_penalties = {}
            
        # Perform quick sanity check on action index
        if action_idx < 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Invalid menu action index: {action_idx}")
            return
        
        # Add to the set of menu actions if not already there
        if action_idx not in self.menu_action_indices:
            self.menu_action_indices.append(action_idx)
        
        # Update penalty value using progressive approach
        current_penalty = self.menu_action_penalties.get(action_idx, 0.0)
        
        # Progressive penalty: more penalizing with repeated occurrences
        if current_penalty <= 0.2:
            # First occurrence - light penalty
            new_penalty = max(current_penalty, penalty * 0.5)
        elif current_penalty <= 0.5:
            # Second occurrence - medium penalty
            new_penalty = max(current_penalty, penalty * 0.75) 
        else:
            # Repeated occurrences - full or increased penalty
            new_penalty = max(current_penalty, penalty * 1.2)  # Can exceed original penalty
            
        # Cap at 0.95 to always leave some probability
        new_penalty = min(0.95, new_penalty)
        
        # Store updated penalty
        self.menu_action_penalties[action_idx] = new_penalty
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Registered menu action {action_idx} with penalty {self.menu_action_penalties[action_idx]}")
        
        # If this is a high-penalty action, also penalize similar actions
        if new_penalty > 0.7:
            # Penalize adjacent action indices (likely similar actions)
            for adj_idx in [action_idx-1, action_idx+1]:
                if adj_idx >= 0 and adj_idx not in self.menu_action_indices:
                    # Apply a reduced penalty to adjacent actions
                    self.menu_action_indices.append(adj_idx)
                    self.menu_action_penalties[adj_idx] = min(0.3, new_penalty * 0.4)
                    logger.info(f"Added related action {adj_idx} with reduced penalty {self.menu_action_penalties[adj_idx]}")

    def decay_menu_penalties(self):
        """Gradually reduce penalties for menu actions over time."""
        if hasattr(self, 'menu_action_penalties') and self.menu_action_penalties:
            for action_idx in list(self.menu_action_penalties.keys()):
                # Decay the penalty
                self.menu_action_penalties[action_idx] *= self.menu_penalty_decay
                
                # Remove very small penalties
                if self.menu_action_penalties[action_idx] < 0.05:
                    del self.menu_action_penalties[action_idx]

    def update_from_reward(self, action_idx: int, reward: float):
        """Update action penalties based on received rewards.
        
        Args:
            action_idx (int): Index of the action that received the reward
            reward (float): The reward value
        """
        # Track the most recent rewards
        if not hasattr(self, 'last_rewards'):
            self.last_rewards = []
        
        # Keep only the last 10 rewards
        self.last_rewards.append(reward)
        if len(self.last_rewards) > 10:
            self.last_rewards.pop(0)
        
        # Check if we received an extreme negative reward (likely from menu penalty)
        if reward < self.extreme_penalty_threshold:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Extreme negative reward detected: {reward}, likely menu penalty")
            
            # Strongly penalize the action that led to this reward
            self.register_menu_action(action_idx, penalty=0.9) 