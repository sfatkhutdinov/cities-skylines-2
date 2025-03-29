"""
Updater module for PPO agent in Cities: Skylines 2.

This module implements the update logic for the PPO agent,
handling policy and value function optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Iterator, Union

logger = logging.getLogger(__name__)

class PPOUpdater:
    """Implements the update logic for the PPO agent."""
    
    def __init__(self, network, device: torch.device, lr: float = 3e-4):
        """Initialize updater component.
        
        Args:
            network: Neural network model
            device: Device to run computations on
            lr: Learning rate for optimizer
        """
        self.network = network
        self.device = device
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        
        # PPO parameters
        self.clip_param = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.ppo_epochs = 4
        self.batch_size = 64
        
        # Learning rate decay
        self.initial_lr = lr
        self.lr_scheduler = None
        
        # Create adaptive learning rate scheduler
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=10,
            verbose=True
        )
        
        logger.info(f"Initialized PPO updater with lr={lr}")
    
    def update(self, memory, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Dict[str, float]:
        """Update policy and value function using PPO.
        
        Args:
            memory: Experience memory
            scaler: Optional GradScaler for mixed precision
            
        Returns:
            Dict with update statistics
        """
        # Check if enough experiences are stored
        if memory.size() < self.batch_size:
            logger.warning(f"Not enough experiences for update: {memory.size()} < {self.batch_size}")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        # Track metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        # Do multiple PPO epochs
        for epoch in range(self.ppo_epochs):
            # Get batch iterator
            batch_iter = memory.get_batch_iterator(self.batch_size, shuffle=True)
            
            # Process each batch
            for batch in batch_iter:
                # Get data from batch
                states = batch['states'].to(self.device)
                actions = batch['actions'].to(self.device)
                old_log_probs = batch['old_log_probs'].to(self.device)
                returns = batch['returns'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                
                # --- AMP Handling Start ---
                # Use autocast context if scaler is provided
                autocast_context = torch.cuda.amp.autocast() if scaler else torch.no_grad() # Use dummy context if no scaler
                
                with autocast_context:
                    # Forward pass through network
                    action_probs, values = self.network(states)
                    
                    # TODO: Review log_prob calculation - assumes Categorical distribution implicitly
                    # Consider using torch.distributions directly if network output allows
                    # Calculate new log probs using Categorical distribution assumption
                    dist = torch.distributions.Categorical(probs=action_probs)
                    new_log_probs = dist.log_prob(actions)
                    dist_entropy = dist.entropy().mean()
                    
                    # Compute ratio and clipped ratio
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    
                    # Compute surrogate losses
                    surrogate1 = ratio * advantages
                    surrogate2 = clipped_ratio * advantages
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()
                    
                    # Compute value loss (standard MSE, consider clipped value loss if needed)
                    value_loss = nn.functional.mse_loss(values.squeeze(-1), returns) # Ensure value has correct shape
                    
                    # Compute total loss
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * dist_entropy
                # --- AMP Handling End ---

                # Backward pass and optimization
                self.optimizer.zero_grad()
                
                if scaler:
                    # Use scaler for backward pass
                    scaler.scale(loss).backward()
                    # Unscale gradients before clipping
                    scaler.unscale_(self.optimizer)
                    # Clip gradients
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    # Step optimizer using scaler
                    scaler.step(self.optimizer)
                    # Update scaler for next iteration
                    scaler.update()
                else:
                    # Standard backward pass and step
                    loss.backward()
                    # Clip gradients
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    # Update parameters
                    self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += dist_entropy.item()
                num_updates += 1
        
        # Compute average metrics
        avg_policy_loss = total_policy_loss / max(1, num_updates)
        avg_value_loss = total_value_loss / max(1, num_updates)
        avg_entropy = total_entropy / max(1, num_updates)
        
        # Update learning rate if using scheduler
        if self.lr_scheduler:
            # Use negative policy loss as metric for scheduler
            # (higher is better for ReduceLROnPlateau with 'max' mode)
            self.lr_scheduler.step(-avg_policy_loss)
        
        # Return statistics
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'num_updates': num_updates,
            'learning_rate': self.get_learning_rate()
        }
    
    def set_params(self, 
                  clip_param: Optional[float] = None, 
                  value_coef: Optional[float] = None, 
                  entropy_coef: Optional[float] = None,
                  max_grad_norm: Optional[float] = None,
                  lr: Optional[float] = None,
                  ppo_epochs: Optional[int] = None,
                  batch_size: Optional[int] = None) -> None:
        """Set updater parameters.
        
        Args:
            clip_param: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            lr: Learning rate
            ppo_epochs: Number of PPO epochs
            batch_size: Batch size for updates
        """
        if clip_param is not None:
            self.clip_param = clip_param
        if value_coef is not None:
            self.value_coef = value_coef
        if entropy_coef is not None:
            self.entropy_coef = entropy_coef
        if max_grad_norm is not None:
            self.max_grad_norm = max_grad_norm
        if ppo_epochs is not None:
            self.ppo_epochs = ppo_epochs
        if batch_size is not None:
            self.batch_size = batch_size
        
        # Update learning rate if changed
        if lr is not None and lr != self.get_learning_rate():
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.initial_lr = lr
            
            logger.info(f"Updated learning rate to {lr}")
    
    def get_learning_rate(self) -> float:
        """Get current learning rate.
        
        Returns:
            Current learning rate
        """
        return self.optimizer.param_groups[0]['lr']
    
    def create_lr_scheduler(self, scheduler_type: str = 'plateau', **kwargs) -> None:
        """Create learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler ('plateau', 'step', 'cosine', etc.)
            **kwargs: Additional parameters for scheduler
        """
        if scheduler_type == 'plateau':
            # ReduceLROnPlateau scheduler
            patience = kwargs.get('patience', 10)
            factor = kwargs.get('factor', 0.5)
            min_lr = kwargs.get('min_lr', 1e-6)
            
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max',  # Higher reward is better
                factor=factor,
                patience=patience,
                min_lr=min_lr,
                verbose=True
            )
            
        elif scheduler_type == 'step':
            # StepLR scheduler
            step_size = kwargs.get('step_size', 1000)
            gamma = kwargs.get('gamma', 0.9)
            
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
            
        elif scheduler_type == 'cosine':
            # CosineAnnealingLR scheduler
            t_max = kwargs.get('t_max', 10000)
            eta_min = kwargs.get('eta_min', 0)
            
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min
            )
            
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}")
            self.lr_scheduler = None
            
        logger.info(f"Created {scheduler_type} learning rate scheduler")
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for serialization.
        
        Returns:
            Dict containing state information
        """
        return {
            'optimizer': self.optimizer.state_dict(),
            'clip_param': self.clip_param,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'max_grad_norm': self.max_grad_norm,
            'ppo_epochs': self.ppo_epochs,
            'batch_size': self.batch_size,
            'initial_lr': self.initial_lr,
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        
        self.clip_param = state_dict.get('clip_param', 0.2)
        self.value_coef = state_dict.get('value_coef', 0.5)
        self.entropy_coef = state_dict.get('entropy_coef', 0.01)
        self.max_grad_norm = state_dict.get('max_grad_norm', 0.5)
        self.ppo_epochs = state_dict.get('ppo_epochs', 4)
        self.batch_size = state_dict.get('batch_size', 64)
        self.initial_lr = state_dict.get('initial_lr', 3e-4)
        
        # Load lr_scheduler if it exists
        if self.lr_scheduler and 'lr_scheduler' in state_dict and state_dict['lr_scheduler']:
            self.lr_scheduler.load_state_dict(state_dict['lr_scheduler']) 