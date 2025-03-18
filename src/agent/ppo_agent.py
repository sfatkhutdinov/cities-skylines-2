"""
PPO agent for Cities: Skylines 2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any
import threading
import logging
from model.optimized_network import OptimizedNetwork
from agent.curiosity_module import IntrinsicCuriosityModule

logger = logging.getLogger(__name__)

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
        
        # Initialize intrinsic curiosity module
        self.use_curiosity = True  # Flag to enable/disable curiosity
        
        # Adaptive curiosity weight that decreases over time
        self.initial_curiosity_weight = 0.1  # Higher starting value for better exploration
        self.min_curiosity_weight = 0.001   # Minimum curiosity weight
        self.curiosity_weight = self.initial_curiosity_weight
        self.curiosity_decay_factor = 0.9999  # Decay per episode
        self.training_episodes = 0          # Episode counter for adaptive decay
        
        self.icm = IntrinsicCuriosityModule(config)
        
        # Setup optimizers
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate
        )
        
        self.curiosity_optimizer = optim.Adam(
            self.icm.parameters(),
            lr=config.learning_rate * 0.5  # Slightly lower learning rate for stability
        )
        
        # Initialize memory buffers
        self.states = []
        self.next_states = []  # Add buffer for next states (needed for curiosity)
        self.actions = []
        self.action_probs = []
        self.values = []
        self.rewards = []
        self.intrinsic_rewards = []  # Add buffer for intrinsic rewards
        self.dones = []
        
        # Add action avoidance for menu toggling
        self.menu_action_indices = []  # Will be populated with indices of actions that open menus
        self.menu_action_penalties = {}  # Maps action indices to penalties
        
        # Dynamic menu penalty decay based on training progress
        self.initial_menu_penalty_decay = 0.98
        self.min_menu_penalty_decay = 0.95  # Faster decay in later stages
        self.menu_penalty_decay = self.initial_menu_penalty_decay
        self.menu_penalty_decay_rate = 0.99999  # Very slow adjustment rate
        
        self.last_rewards = []  # Track recent rewards to detect large penalties
        self.extreme_penalty_threshold = -500.0  # Threshold for detecting extreme penalties
        
        # Add threading lock for thread safety
        self.update_lock = threading.Lock()
        self.is_updating = False
        
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
        
    def select_action_batch(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select actions for a batch of states using the current policy.
        
        This method is optimized for parallel experience collection.
        
        Args:
            states (torch.Tensor): Batch of environment states [B, C, H, W]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (actions, log_probs, values)
        """
        # Ensure states are on the correct device
        if states.device != self.device:
            states = states.to(self.device)
            
        # Get action probabilities and values for the entire batch at once
        with torch.no_grad():
            action_probs_batch, values_batch = self.network(states)
            
            # Apply penalties to menu-opening actions if needed
            if self.menu_action_indices and hasattr(self, 'menu_action_penalties'):
                # Create a penalty mask (default 1.0 for all actions)
                penalty_mask = torch.ones_like(action_probs_batch)
                
                # Apply specific penalties to known menu actions
                for action_idx, penalty in self.menu_action_penalties.items():
                    if 0 <= action_idx < penalty_mask.size(1):
                        # Reduce probability of this action by the penalty factor
                        penalty_mask[:, action_idx] = torch.max(
                            torch.tensor(0.1, device=self.device), 
                            torch.tensor(1.0 - penalty, device=self.device)
                        )
                
                # Apply the mask to reduce probabilities of problematic actions
                action_probs_batch = action_probs_batch * penalty_mask
                
                # Renormalize probabilities
                action_probs_batch = action_probs_batch / action_probs_batch.sum(dim=1, keepdim=True)
            
        # Sample actions from probability distributions
        actions = torch.multinomial(action_probs_batch, 1)
        
        # Get log probabilities for sampled actions
        log_probs = torch.log(
            torch.gather(action_probs_batch, 1, actions)
        )
        
        return actions, log_probs, values_batch
        
    def compute_intrinsic_reward(self, state, next_state, action):
        """Compute intrinsic reward using the curiosity module.
        
        Args:
            state (torch.Tensor): Current state
            next_state (torch.Tensor): Next state
            action (int or torch.Tensor): Action taken
            
        Returns:
            float: Intrinsic reward
        """
        if not self.use_curiosity:
            return 0.0
            
        # Safety check - if inputs are None, return 0
        if state is None or next_state is None:
            return 0.0
            
        # Ensure tensors are on the correct device
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(self.device)
        if isinstance(action, int):
            action = torch.tensor([action], device=self.device)
            
        # Handle batch dimension for state tensors
        if state.dim() == 3:  # [C, H, W] -> [1, C, H, W]
            state = state.unsqueeze(0)
        if next_state.dim() == 3:  # [C, H, W] -> [1, C, H, W]
            next_state = next_state.unsqueeze(0)
            
        # Compute curiosity reward with error handling
        try:
            with torch.no_grad():
                intrinsic_reward = self.icm.compute_curiosity_reward(state, next_state, action)
                
            return intrinsic_reward.item() * self.curiosity_weight
        except Exception as e:
            # Log error and return 0 reward on failure
            print(f"Error computing curiosity reward: {str(e)}")
            return 0.0
        
    def update_async(self) -> None:
        """Asynchronously update the agent in a separate thread.
        
        This allows experience collection to continue while the update is processed.
        
        Returns:
            None: Update happens asynchronously
        """
        if self.is_updating:
            logger.debug("Update already in progress, skipping")
            return
            
        # Start update in a separate thread
        self.is_updating = True
        update_thread = threading.Thread(target=self._update_thread)
        update_thread.daemon = True  # Thread will exit when main program exits
        update_thread.start()
        
    def _update_thread(self) -> None:
        """Internal method to perform the update in a separate thread."""
        try:
            with self.update_lock:
                update_results = self.update()
                logger.debug(f"Async update complete: {update_results}")
        except Exception as e:
            logger.error(f"Error in async update thread: {e}")
        finally:
            self.is_updating = False
            
    def update(self) -> Dict[str, float]:
        """Update the agent using collected experience.
        
        Returns:
            dict: Training metrics
        """
        # Increment episode counter for adaptive curiosity decay
        self.training_episodes += 1
        
        # Decay curiosity weight
        self.curiosity_weight = max(
            self.min_curiosity_weight,
            self.curiosity_weight * self.curiosity_decay_factor
        )
        
        # Update menu penalty decay rate - gradually speed up decay as training progresses
        self.menu_penalty_decay = max(
            self.min_menu_penalty_decay,
            self.menu_penalty_decay * self.menu_penalty_decay_rate
        )
        
        # Check if we have any experience to learn from
        if not self.states or not self.actions or not self.next_states:
            logger.warning("No experience to learn from")
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0, "advantage": 0}
            
        # Convert lists to tensors
        states = torch.cat(self.states)
        next_states = torch.cat(self.next_states)  # Use stored next states
        actions = torch.cat(self.actions)
        
        # Check if we have action probabilities
        if not self.action_probs:
            self._clear_memory()
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'icm_loss': 0.0}
            
        old_action_probs = torch.cat(self.action_probs)
        values = torch.cat(self.values)
        
        # Compute returns and advantages
        returns = self._compute_returns()
        
        # Check if returns is empty
        if returns.numel() == 0:
            self._clear_memory()
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'icm_loss': 0.0}
            
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update curiosity module
        icm_loss = 0.0
        if self.use_curiosity:
            intrinsic_rewards, icm_loss = self.icm(states, next_states, actions)
            
            # Update ICM
            self.curiosity_optimizer.zero_grad()
            icm_loss.backward()
            self.curiosity_optimizer.step()
        
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
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'icm_loss': icm_loss.item() if isinstance(icm_loss, torch.Tensor) else icm_loss
        }
        
        # After update, clear memory and free GPU memory
        self._clear_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # More aggressive memory cleanup
            
        return metrics
        
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
        self.next_states = []  # Clear next_states buffer
        self.actions = []
        self.action_probs = []
        self.values = []
        self.rewards = []
        self.intrinsic_rewards = []  # Clear intrinsic rewards
        self.dones = []
        
    def save(self, path: str):
        """Save agent state.
        
        Args:
            path (str): Path to save state to
        """
        torch.save({
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'icm_state': self.icm.state_dict(),  # Save ICM state
            'curiosity_optimizer_state': self.curiosity_optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        """Load agent state.
        
        Args:
            path (str): Path to load state from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Load ICM state if it exists
        if 'icm_state' in checkpoint:
            self.icm.load_state_dict(checkpoint['icm_state'])
            self.curiosity_optimizer.load_state_dict(checkpoint['curiosity_optimizer_state'])
        
    def register_menu_action(self, action_idx: int, penalty: float = 0.5):
        """Register an action that led to a menu state to discourage its selection.
        
        Args:
            action_idx (int): Index of the action that led to a menu
            penalty (float): Penalty factor (0-1) to apply to this action's probability
        """
        if not hasattr(self, 'menu_action_indices'):
            self.menu_action_indices = []
            self.menu_action_penalties = {}
        
        # Add to the set of menu actions if not already there
        if action_idx not in self.menu_action_indices:
            self.menu_action_indices.append(action_idx)
        
        # Update penalty value (use max to ensure it only increases)
        current_penalty = self.menu_action_penalties.get(action_idx, 0.0)
        self.menu_action_penalties[action_idx] = max(current_penalty, penalty)
        
        logger.info(f"Registered menu action {action_idx} with penalty {self.menu_action_penalties[action_idx]}")

    def decay_menu_penalties(self):
        """Gradually reduce penalties for menu actions over time."""
        if hasattr(self, 'menu_action_penalties') and self.menu_action_penalties:
            for action_idx in list(self.menu_action_penalties.keys()):
                # Decay the penalty
                self.menu_action_penalties[action_idx] *= self.menu_penalty_decay
                
                # Remove very small penalties
                if self.menu_action_penalties[action_idx] < 0.05:
                    del self.menu_action_penalties[action_idx]

    def update_from_reward(self, action_idx, reward):
        """Update agent's tracking of rewards for an action.
        
        Args:
            action_idx (int): Index of action taken
            reward (float): Reward received
        """
        # Store reward for tracking large penalties
        self.last_rewards.append(reward)
        if len(self.last_rewards) > 10:
            self.last_rewards.pop(0)
            
        # If we got an extremely negative reward, this might be a menu penalty
        if reward < self.extreme_penalty_threshold:
            logger.warning(f"Detected extreme penalty ({reward}) for action {action_idx}")
            self.register_menu_action(action_idx, penalty=0.9)  # Higher penalty for extreme cases
            
    def store_next_state(self, next_state):
        """Store the next state for use with the intrinsic curiosity module.
        
        Args:
            next_state (torch.Tensor): Next state observation
        """
        if next_state is not None:
            # Convert to tensor if needed
            if not isinstance(next_state, torch.Tensor):
                next_state = torch.FloatTensor(next_state)
                
            # Ensure correct device
            if next_state.device != self.device:
                next_state = next_state.to(self.device)
                
            # Handle batch dimension
            if next_state.dim() == 3:
                next_state = next_state.unsqueeze(0)
                
            # Store in buffer
            self.next_states.append(next_state) 