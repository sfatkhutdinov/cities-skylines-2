"""
PPO agent for Cities: Skylines 2.
"""

import logging
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any, List
from model.optimized_network import OptimizedNetwork
import os

class PPOAgent:
    """Proximal Policy Optimization agent."""
    
    def __init__(self, state_dim, action_dim, config=None):
        """Initialize PPO agent.
        
        Args:
            state_dim: Dimensions of the state space
            action_dim: Dimensions of the action space
            config: Optional configuration object
        """
        # Get device
        self.device = config.get_device() if config else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"PPOAgent using device: {self.device}")
        
        # Network parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize network
        self.network = OptimizedNetwork(state_dim, action_dim, device=self.device)
        
        # RL parameters
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE lambda parameter
        self.clip_param = 0.2  # PPO clip parameter
        self.value_coef = 0.5  # Value loss coefficient
        self.entropy_coef = 0.01  # Entropy coefficient
        self.max_grad_norm = 0.5  # Maximum gradient norm
        
        # Update parameters from config if provided
        if config:
            if hasattr(config, 'gamma'):
                self.gamma = config.gamma
            if hasattr(config, 'gae_lambda'):
                self.gae_lambda = config.gae_lambda
            if hasattr(config, 'clip_param'):
                self.clip_param = config.clip_param
            if hasattr(config, 'value_coef'):
                self.value_coef = config.value_coef
            if hasattr(config, 'entropy_coef'):
                self.entropy_coef = config.entropy_coef
            if hasattr(config, 'max_grad_norm'):
                self.max_grad_norm = config.max_grad_norm
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        
        # Experience memory
        self._clear_memory()
        
        # Menu action tracking
        self.menu_action_indices = set()
        self.menu_action_penalties = {}
        
        # Flag to indicate if agent is in training mode
        self.training = True
        
        # Add action avoidance for menu toggling
        self.menu_action_indices = []  # Will be populated with indices of actions that open menus
        self.menu_action_penalties = {}  # Maps action indices to penalties
        self.menu_penalty_decay = 0.98  # Slower decay rate (was 0.95)
        self.last_rewards = []  # Track recent rewards to detect large penalties
        self.extreme_penalty_threshold = -500.0  # Threshold for detecting extreme penalties
        
        # Keep track of the last state for continued episodes
        self.last_state = None
        
        # Running statistics for reward normalization
        self.rewards_mean = 0
        self.rewards_std = 1
        
        # Causal understanding integration
        self.use_causal_suggestions = False  # Start with policy-only, enable after 1000 steps
        self.causal_suggestion_weight = 0.3  # How much to bias towards causal suggestions
        self.steps_counter = 0
        self.reward_system = None  # Will be set externally
        
    def select_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select an action based on state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Action, log probability, and value
        """
        # Count steps
        self.steps_counter += 1
        
        # Check if we should start using causal suggestions
        if self.steps_counter == 1000 and not self.use_causal_suggestions:
            self.use_causal_suggestions = True
            logger.info("Enabled causal action suggestions for agent decision making")
        
        # Get policy distribution and value
        with torch.no_grad():
            policy, value = self.network(state)
            
        # Standard policy-based action selection
        dist = torch.distributions.Categorical(logits=policy)
        
        # Get causal suggestion if available and enabled
        causal_action = None
        confidence = 0.0
        
        if self.use_causal_suggestions and self.reward_system is not None:
            try:
                # Get available actions (all actions by default)
                available_actions = list(range(self.action_dim))
                
                # Get recommendation from causal system
                causal_action, confidence = self.reward_system.get_action_recommendation(
                    state, available_actions
                )
                
                # Log causal recommendation
                logger.debug(f"Causal action recommendation: {causal_action}, confidence: {confidence:.3f}")
            except Exception as e:
                logger.error(f"Error getting causal action recommendation: {e}")
                causal_action = None
        
        # Combine policy with causal suggestion if available
        if causal_action is not None and confidence > 0.2:
            # Calculate new probabilities biased towards causal suggestion
            probs = dist.probs.clone()
            
            # Scale confidence by number of actions
            bias_strength = self.causal_suggestion_weight * confidence
            
            # Increase probability of causal action
            probs[causal_action] = probs[causal_action] * (1.0 + bias_strength)
            
            # Normalize to ensure sum=1
            probs = probs / probs.sum()
            
            # Create new distribution
            biased_dist = torch.distributions.Categorical(probs=probs)
            
            # Sample from biased distribution
            action = biased_dist.sample()
            log_prob = biased_dist.log_prob(action)
        else:
            # Sample from standard policy distribution
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action, log_prob, value
        
    def update(self, experiences=None):
        """Update policy and value function.
        
        Args:
            experiences: Optional list of experiences to add to memory before updating

        Returns:
            dict: Training metrics
        """
        # If experiences are provided, add them to memory
        if experiences:
            for state, action, reward, next_state, log_prob, value, done in experiences:
                self.states.append(state)
                self.actions.append(action)
                self.rewards.append(reward)
                self.dones.append(done)
                self.values.append(value)
        
        # Check if memory is empty
        if not self.states:
            logger.warning("Cannot update with empty memory")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # Ensure all memory arrays have the same length - there might be a bug in collection
        min_length = min(len(self.states), len(self.actions), len(self.rewards), len(self.dones), len(self.values))
        if min_length < len(self.states):
            logger.warning(f"Truncating memory: States: {len(self.states)}, min length: {min_length}")
            self.states = self.states[:min_length]
            self.actions = self.actions[:min_length]
            self.rewards = self.rewards[:min_length]
            self.dones = self.dones[:min_length]
            self.values = self.values[:min_length]
            if self.action_probs:
                self.action_probs = self.action_probs[:min_length]
            
        # Convert states to a single tensor
        states = torch.stack(self.states).to(self.device)
        
        # Process actions
        action_tensors = []
        for action in self.actions:
            if isinstance(action, torch.Tensor):
                # Ensure action is on the correct device
                if action.device != self.device:
                    action = action.to(self.device)
                action_tensors.append(action.view(1))
            else:
                # Convert int to tensor
                action_tensors.append(torch.tensor([action], device=self.device))
        actions = torch.cat(action_tensors)
        
        # Process values
        values_list = []
        for val in self.values:
            if isinstance(val, torch.Tensor):
                if val.device != self.device:
                    val = val.to(self.device)
                if val.dim() == 0:
                    val = val.view(1)
                values_list.append(val)
            else:
                values_list.append(torch.tensor([val], device=self.device))
        old_values = torch.cat(values_list)
        
        # If action_probs is empty, we need to compute them now
        if not self.action_probs or len(self.action_probs) != len(self.states):
            logger.debug("Computing action probabilities")
            self.action_probs = []
            with torch.no_grad():
                for state in self.states:
                    state_tensor = torch.tensor(state, device=self.device) if not isinstance(state, torch.Tensor) else state.to(self.device)
                    if len(state_tensor.shape) == 3:  # [C, H, W]
                        state_tensor = state_tensor.unsqueeze(0)  # Add batch dimension
                    action_probs, _ = self.network(state_tensor)
                    self.action_probs.append(action_probs.squeeze(0))
        
        # Process action probabilities
        old_action_probs = torch.stack([p.to(self.device) for p in self.action_probs])
        
        # Convert rewards and dones to tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)
        
        # Check tensor sizes before normalization
        logger.debug(f"Shapes - states: {states.shape}, actions: {actions.shape}, rewards: {rewards.shape}, " + 
                     f"dones: {dones.shape}, old_values: {old_values.shape}, old_action_probs: {old_action_probs.shape}")
        
        # Normalize rewards for training stability
        if len(rewards) > 1 and rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Compute returns and advantages
        returns = self._compute_returns(rewards, dones)
        
        # Debug prints
        logger.debug(f"Returns shape: {returns.shape}, old_values shape: {old_values.shape}")
        
        # Ensure returns and old_values have the same shape
        if returns.shape != old_values.shape:
            logger.warning(f"Shape mismatch between returns and old_values! Returns: {returns.shape}, Old values: {old_values.shape}")
            min_len = min(len(returns), len(old_values))
            returns = returns[:min_len]
            old_values = old_values[:min_len]
        
        advantages = returns - old_values
        
        # Normalize advantages for training stability
        if len(advantages) > 1 and advantages.std() > 0:  
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Make sure everything is the same length
        batch_size = min(len(states), len(actions), len(returns), len(advantages), len(old_action_probs))
        states = states[:batch_size]
        actions = actions[:batch_size]
        returns = returns[:batch_size]
        advantages = advantages[:batch_size]
        old_action_probs = old_action_probs[:batch_size]
        
        # Store metrics for return
        policy_loss_val = 0
        value_loss_val = 0
        entropy_val = 0
        
        # PPO update loop
        for epoch in range(4):  # Default to 4 epochs
            # Compute action probabilities and values for this epoch
            new_action_probs, new_values = self.network(states)
            
            # Get log probabilities for the chosen actions
            action_indices = actions.unsqueeze(1)
            new_log_probs = torch.log(new_action_probs.gather(1, action_indices)).squeeze(1)
            old_log_probs = torch.log(old_action_probs.gather(1, action_indices)).squeeze(1)
            
            # Compute policy ratio and clipped ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
            
            # Compute policy loss
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            
            # Compute value loss
            value_loss = 0.5 * (returns - new_values).pow(2).mean()
            
            # Compute entropy bonus
            entropy = -(new_action_probs * torch.log(new_action_probs + 1e-10)).sum(dim=1).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Store last epoch values for metrics
            policy_loss_val = policy_loss.item()
            value_loss_val = value_loss.item()
            entropy_val = entropy.item()
        
        # Calculate metrics
        metrics = {
            'policy_loss': policy_loss_val,
            'value_loss': value_loss_val,
            'entropy': entropy_val
        }
        
        # Clear memory for next update
        self._clear_memory()
        
        return metrics
        
    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Compute returns using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards (torch.Tensor): List of rewards
            dones (torch.Tensor): List of done flags
            
        Returns:
            torch.Tensor: Computed returns
        """
        # Handle empty values case
        if not self.values:
            logger.warning("No values in memory for computing returns")
            return torch.zeros_like(rewards)
        
        # Process values to ensure they're properly shaped tensors
        value_tensors = []
        for val in self.values:
            if isinstance(val, torch.Tensor):
                # Ensure val is on the correct device
                if val.device != self.device:
                    val = val.to(self.device)
                # Handle scalar tensors
                if val.dim() == 0:
                    val = val.view(1)
                value_tensors.append(val)
            else:
                # Convert scalar value to tensor
                value_tensors.append(torch.tensor([val], device=self.device))
        
        # Concatenate value tensors
        values = torch.cat(value_tensors)
        
        # Add a final value of 0 if the last state is terminal
        if dones[-1]:
            next_value = 0
        else:
            # Use the last value or estimate it from the last state
            next_value = values[-1].item()
            
        # Create a tensor with the same dimension as values, avoiding unsqueeze mismatch
        next_value_tensor = torch.tensor([next_value], device=self.device)
        
        # Ensure values and next_value_tensor have the same dimension
        if values.dim() == 1:  # If values is 1D [batch], next_value should also be 1D
            values = torch.cat([values, next_value_tensor])
        else:
            # If values has more dimensions, reshape next_value to match
            next_value_tensor = next_value_tensor.view(*[1] * (values.dim() - 1), 1)
            values = torch.cat([values, next_value_tensor], dim=0)
        
        returns = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            # Delta is the one step TD error
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            
            # GAE recursively computes advantage by adding discounted delta terms
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            # Return is advantage plus value
            returns[t] = gae + values[t]
            
        return returns
        
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
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save core model and optimizer states
        checkpoint = {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'last_state': self.last_state.cpu() if self.last_state is not None else None,
            'running_stats': {
                'rewards_mean': self.rewards_mean,
                'rewards_std': self.rewards_std
            },
            'action_tracking': {}
        }
        
        # Save action tracking if available
        if hasattr(self, 'menu_action_indices'):
            checkpoint['action_tracking']['menu_action_indices'] = self.menu_action_indices
            checkpoint['action_tracking']['menu_action_penalties'] = self.menu_action_penalties
        
        # Save recent trajectories for continuation
        checkpoint['recent_trajectories'] = {
            'states': [s.cpu() if s is not None else None for s in self.states[-10:]],
            'actions': self.actions[-10:] if self.actions else [],
            'action_probs': self.action_probs[-10:] if self.action_probs else [],
            'values': self.values[-10:] if self.values else [],
            'rewards': self.rewards[-10:] if self.rewards else [],
            'dones': self.dones[-10:] if self.dones else []
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Agent state saved to {path}")
        
    def load(self, path: str):
        """Load agent state.
        
        Args:
            path (str): Path to load state from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load core model and optimizer states
            self.network.load_state_dict(checkpoint['network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
            # Load last state if it exists
            if checkpoint.get('last_state') is not None:
                self.last_state = checkpoint['last_state'].to(self.device)
            
            # Load running statistics
            if 'running_stats' in checkpoint:
                self.rewards_mean = checkpoint['running_stats'].get('rewards_mean', 0)
                self.rewards_std = checkpoint['running_stats'].get('rewards_std', 1)
            
            # Load action tracking
            if 'action_tracking' in checkpoint and checkpoint['action_tracking']:
                self.menu_action_indices = checkpoint['action_tracking'].get('menu_action_indices', [])
                self.menu_action_penalties = checkpoint['action_tracking'].get('menu_action_penalties', {})
            
            # Load recent trajectories
            if 'recent_trajectories' in checkpoint:
                # Only restore if we have valid trajectory data
                if checkpoint['recent_trajectories']['states']:
                    # Move states to device
                    self.states = [
                        s.to(self.device) if s is not None else None 
                        for s in checkpoint['recent_trajectories']['states']
                    ]
                    self.actions = checkpoint['recent_trajectories']['actions']
                    self.action_probs = checkpoint['recent_trajectories']['action_probs']
                    self.values = checkpoint['recent_trajectories']['values']
                    self.rewards = checkpoint['recent_trajectories']['rewards']
                    self.dones = checkpoint['recent_trajectories']['dones']
            
            logger.info(f"Agent state loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load agent state: {str(e)}")
            raise
        
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
            # Improved: Penalize similar action types (not just adjacent indices)
            # For key actions, penalize other key actions in same category
            # For mouse actions, penalize similar mouse actions
            similar_indices = self._find_similar_actions(action_idx)
            
            for sim_idx in similar_indices:
                if sim_idx >= 0 and sim_idx not in self.menu_action_indices:
                    # Apply a reduced penalty to similar actions
                    self.menu_action_indices.append(sim_idx)
                    self.menu_action_penalties[sim_idx] = min(0.3, new_penalty * 0.4)
                    logger.info(f"Added related action {sim_idx} with reduced penalty {self.menu_action_penalties[sim_idx]}")
    
    def _find_similar_actions(self, action_idx: int) -> List[int]:
        """Find actions similar to the given action index.
        
        Args:
            action_idx (int): Index of the action
            
        Returns:
            List[int]: List of similar action indices
        """
        similar_indices = []
        
        # Add adjacent action indices (likely similar actions)
        similar_indices.extend([action_idx-1, action_idx+1])
        
        # Find actions of the same type (key, mouse, etc.)
        # This is a heuristic based on common action layouts
        if 0 <= action_idx <= 14:
            # Key press actions - find other key press actions
            similar_indices.extend([i for i in range(0, 14) if abs(i - action_idx) <= 4])
        elif 15 <= action_idx <= 20:
            # Mouse actions - find other mouse actions
            similar_indices.extend([i for i in range(15, 20) if i != action_idx])
        elif 21 <= action_idx <= 32:
            # Game control keys - find related controls
            similar_indices.extend([i for i in range(21, 32) if abs(i - action_idx) <= 3])
        
        # Filter out invalid indices and duplicates
        return list(set([idx for idx in similar_indices if idx >= 0]))

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

    def set_reward_system(self, reward_system):
        """Set the reward system for causal suggestions.
        
        Args:
            reward_system: AutonomousRewardSystem instance
        """
        self.reward_system = reward_system
        logger.info("Reward system connected to agent for causal action suggestions") 