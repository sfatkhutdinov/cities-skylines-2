"""
Trainer module for reinforcement learning in Cities: Skylines 2.

This module manages the training loop and related utilities.
"""

import torch
import numpy as np
import logging
import time
import wandb
import datetime
import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import contextlib
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .checkpointing import CheckpointManager
from .signal_handlers import is_exit_requested, request_exit
from ..config.hardware_config import HardwareConfig
from ..agent.core import PPOAgent
from ..environment.core import Environment
from ..utils.visualization import TrainingVisualizer
from ..utils.hardware_monitor import HardwareMonitor
from ..utils.performance_safeguards import PerformanceSafeguards
from ..agent.core.memory import Memory

logger = logging.getLogger(__name__)

class Trainer:
    """Handles the training process for the PPO agent."""
    
    def __init__(
        self,
        agent,
        env,
        config,
        device: torch.device = None,
        checkpoint_dir: str = "checkpoints",
        tensorboard_dir: str = "logs"
    ):
        """Initialize the trainer.
        
        Args:
            agent: The agent to train
            env: The environment to train in
            config: Training configuration
            device: The device to use for training
            checkpoint_dir: Directory to save checkpoints
            tensorboard_dir: Directory to save tensorboard logs
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up memory
        self.memory = Memory(self.device)
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # PPO hyperparameters
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_param = config.clip_param
        self.ppo_epochs = config.ppo_epochs
        self.batch_size = config.batch_size
        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        
        # Mixed precision training
        self.use_mixed_precision = config.use_mixed_precision and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=config.max_checkpoints
        )
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Initialize visualizer
        self.visualizer = TrainingVisualizer(
            log_dir=tensorboard_dir
        )
        
        # Hardware monitor
        self.hardware_monitor = HardwareMonitor() if config.monitor_hardware else None
        
        # Performance safeguards
        self.performance_safeguards = PerformanceSafeguards(
            config=config
        )
        
        # Training metrics
        self.episodes_completed = 0
        self.steps_completed = 0
        self.best_reward = float('-inf')
        
        # Training parameters
        self.start_episode = 0  # Start from episode 0
        self.num_episodes = config.num_episodes if hasattr(config, 'num_episodes') else 100
        self.max_steps = config.max_steps if hasattr(config, 'max_steps') else 1000
        self.total_steps = 0  # Initialize total steps counter
        
        # Timing
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        
        logger.critical(f"Trainer initialized on {self.device}")
        logger.critical(f"Using mixed precision: {self.use_mixed_precision}")
    
    def _create_optimizer(self):
        """Create the optimizer for the agent's policy.
        
        Returns:
            Optimizer instance
        """
        logger.critical("Creating optimizer")
        
        # Get optimizer parameters from config
        optimizer_type = self.config.optimizer
        learning_rate = self.config.learning_rate
        weight_decay = self.config.weight_decay
        
        # Get trainable parameters
        parameters = self.agent.parameters()
        
        # Create optimizer
        if optimizer_type.lower() == 'adam':
            logger.critical(f"Using Adam optimizer with lr={learning_rate}, weight_decay={weight_decay}")
            return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            logger.critical(f"Using AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
            return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = self.config.momentum if hasattr(self.config, 'momentum') else 0.9
            logger.critical(f"Using SGD optimizer with lr={learning_rate}, momentum={momentum}, weight_decay={weight_decay}")
            return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            logger.warning(f"Unknown optimizer type: {optimizer_type}, using Adam")
            return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
            
    def _create_lr_scheduler(self):
        """Create learning rate scheduler.
        
        Returns:
            Learning rate scheduler or None
        """
        if not hasattr(self.config, 'use_lr_scheduler') or not self.config.use_lr_scheduler:
            logger.critical("Not using learning rate scheduler")
            return None
            
        # Get scheduler parameters from config
        scheduler_type = self.config.scheduler_type if hasattr(self.config, 'scheduler_type') else 'step'
        
        # Create scheduler based on type
        if scheduler_type.lower() == 'step':
            step_size = self.config.lr_step_size
            gamma = self.config.lr_gamma
            logger.critical(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == 'exponential':
            gamma = self.config.lr_gamma
            logger.critical(f"Using ExponentialLR scheduler: gamma={gamma}")
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        elif scheduler_type.lower() == 'cosine':
            t_max = self.config.cosine_t_max if hasattr(self.config, 'cosine_t_max') else 100
            logger.critical(f"Using CosineAnnealingLR scheduler: T_max={t_max}")
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max)
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, not using scheduler")
            return None
    
    def _load_checkpoint(self):
        """Load checkpoint if available."""
        try:
            # Load checkpoint
            self.start_episode, self.best_reward = self.checkpoint_manager.load_checkpoint(
                self.agent, self.optimizer, self.scheduler
            )
            
            # Load episode rewards and lengths from checkpoint if available
            checkpoint_path = self.checkpoint_manager.get_checkpoint_path(self.start_episode)
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.agent.device)
                if 'episode_rewards' in checkpoint:
                    self.episode_rewards = checkpoint['episode_rewards']
                    
                    # Update visualizer metrics
                    self.visualizer.metrics["episode_rewards"] = self.episode_rewards
                
                if 'episode_lengths' in checkpoint:
                    self.episode_lengths = checkpoint['episode_lengths']
                    
                    # Update visualizer metrics
                    self.visualizer.metrics["episode_lengths"] = self.episode_lengths
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.warning("Starting training from scratch due to checkpoint loading error")
            self.start_episode = 0
            self.best_reward = float('-inf')
    
    def _save_checkpoint(self, episode: int, is_best: bool = False, is_backup: bool = False):
        """Save a checkpoint.
        
        Args:
            episode: Current episode number
            is_best: Whether this is the best performing checkpoint so far
            is_backup: Whether this is a backup checkpoint
        """
        try:
            # Create checkpoint state
            state = {
                'episode': episode,
                'agent_state': self.agent.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'best_reward': self.best_reward,
                'total_steps': self.total_steps,
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths,
                'metrics': self.visualizer.metrics
            }
            
            # Add scheduler state if present
            if self.scheduler is not None:
                state['scheduler_state'] = self.scheduler.state_dict()
            
            # Calculate mean recent reward
            mean_reward = None
            if len(self.recent_rewards) > 0:
                mean_reward = sum(self.recent_rewards) / len(self.recent_rewards)
                state['mean_reward'] = mean_reward
            
            # Save checkpoint
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                state, episode, is_best, mean_reward
            )
            
            logger.info(f"Saved checkpoint at episode {episode} to {checkpoint_path}")
            
            # Save metrics separately
            self.visualizer.save_metrics()
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def collect_trajectory(self, max_steps: int = 1000, render: bool = False) -> List:
        """Collect a trajectory of experiences.
        
        Args:
            max_steps: Maximum number of steps to collect
            render: Whether to render the environment
            
        Returns:
            List of experiences
        """
        experiences = []
        total_reward = 0
        
        # Reset environment and start new episode
        logger.critical("Resetting environment to start trajectory collection")
        observation = self.env.reset()
        
        if observation is None:
            logger.critical("Environment reset failed - returned None observation")
            return experiences
            
        logger.critical(f"Initial observation shape: {observation.shape if hasattr(observation, 'shape') else 'unknown'}")
        
        # Flag to track menu transitions
        in_menu = False
        
        # Collect steps until we reach max_steps or episode is done
        for step in range(max_steps):
            # Check for valid observation
            if observation is None:
                logger.critical(f"Invalid observation at step {step}, terminating trajectory")
                break
                
            # Convert observation to tensor
            if not isinstance(observation, torch.Tensor):
                observation = torch.tensor(observation, device=self.agent.device)
                
            # Log observation statistics
            if isinstance(observation, torch.Tensor) and observation.numel() > 0:
                logger.critical(f"Step {step}: Observation stats - shape={observation.shape}, mean={observation.float().mean().item():.4f}, max={observation.float().max().item():.4f}")
                
                # Check for NaN or infinite values
                if torch.isnan(observation).any() or torch.isinf(observation).any():
                    logger.critical(f"NaN or Inf in observation at step {step}")
                    break
            
            # Select action from policy
            action_result = self.agent.select_action(observation)
            
            # Ensure action_result contains required fields
            if isinstance(action_result, dict):
                action = action_result.get('action')
                log_prob = action_result.get('log_prob')
                value = action_result.get('value', torch.tensor(0.0, device=self.agent.device))
                action_probs = action_result.get('action_probs')
            else:
                logger.critical(f"Unexpected action result type: {type(action_result)}")
                action = action_result
                log_prob = None
                value = None
                action_probs = None
                
            # Basic validation
            if action is None:
                logger.critical("Agent returned None action, using random action")
                action = torch.randint(0, self.agent.num_actions, (1,), device=self.agent.device)
                
            logger.critical(f"Step {step}: Selected action {action}, log_prob {log_prob}")
            
            # Take step in environment
            try:
                next_observation, reward, done, info = self.env.step(action)
                logger.critical(f"Step {step}: Action executed, reward={reward}, done={done}")
                
                # Check for game crashes
                if info.get('crashed', False):
                    logger.critical("Game crash detected, terminating trajectory")
                    break
                    
                # Check for menu transitions
                if info.get('in_menu', False) and not in_menu:
                    logger.warning("Detected transition to menu state")
                    in_menu = True
                elif not info.get('in_menu', False) and in_menu:
                    logger.info("Detected return from menu state")
                    in_menu = False
                    
                # Store experience
                if self.memory:
                    self.memory.add(
                        state=observation,
                        action=action,
                        reward=reward,
                        next_state=next_observation,
                        done=done,
                        log_prob=log_prob,
                        value=value,
                        action_probs=action_probs
                    )
                    logger.critical(f"Added experience to memory: reward={reward}, done={done}")
                
                # Add to experiences list
                experiences.append({
                    'state': observation,
                    'action': action,
                    'reward': reward,
                    'next_state': next_observation,
                    'done': done,
                    'log_prob': log_prob,
                    'value': value,
                    'action_probs': action_probs,
                    'info': info
                })
                
                total_reward += reward
                observation = next_observation
                
                # Display debugging info
                if step % 10 == 0:
                    logger.info(f"Step {step}/{max_steps}: reward={reward:.2f}, total={total_reward:.2f}")
                
                # Render if requested
                if render:
                    self.env.render()
                
                # Check if episode is done
                if done:
                    logger.info(f"Episode done after {step+1} steps with total reward {total_reward:.2f}")
                    break
                    
            except Exception as e:
                logger.critical(f"Error in environment step: {e}")
                import traceback
                logger.critical(traceback.format_exc())
                break
        
        # Log trajectory statistics
        logger.info(f"Collected trajectory: {len(experiences)} steps, reward={total_reward:.2f}")
        
        return experiences
    
    def train_episode(
        self, 
        episode_num: int, 
        render: bool = False,
        max_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train for a single episode.
        
        Args:
            episode_num: Current episode number
            render: Whether to render the environment
            max_steps: Maximum steps for this episode (overrides default)
            
        Returns:
            Dict with episode metrics
        """
        logger.critical(f"===== STARTING EPISODE {episode_num} =====")
        # Check if using mock environment
        using_mock = hasattr(self.env, '_update_city_state')
        logger.critical(f"Using mock environment: {using_mock}")
        
        # Set agent to training mode
        self.agent.train()
        
        # Get max steps for this episode
        max_steps = max_steps or self.max_steps
        logger.critical(f"Episode configured for max_steps={max_steps}")
        
        # Start episode timer
        episode_start_time = time.time()
        
        # Reset performance tracking
        if self.performance_safeguards:
            self.performance_safeguards.reset()
        
        # Reset environment and get initial state
        logger.critical("About to reset environment...")
        state = self.env.reset()
        logger.critical(f"Environment reset complete. Observation shape: {state.shape if hasattr(state, 'shape') else 'unknown'}")
        
        # Initialize episode variables
        step = 0
        total_reward = 0.0
        done = False
        experiences = []
        action_counts = {}
        in_menu = False
        menu_duration = 0
        error_states = 0
        episode_stats = {}
        
        # For mixed precision training
        torch_amp_available = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
        using_mixed_precision = self.config.use_mixed_precision and torch_amp_available
        scaler = torch.cuda.amp.GradScaler() if using_mixed_precision else None
        
        logger.critical("Starting episode loop...")
        # Main episode loop
        while not done and step < max_steps:
            logger.critical(f"--- Step {step} ---")
            # Get step start time for timing
            step_start_time = time.time()
            
            # If we're using the mock environment, we'll handle potential errors directly
            observation_valid = True
            
            # Check for early termination request
            if is_exit_requested():
                logger.critical("Exit requested, ending episode early")
                break
            
            # Select action using agent policy
            logger.critical("Selecting action from policy...")
            with torch.set_grad_enabled(False):
                action, log_prob, value = self.agent.select_action(state)
            logger.critical(f"Selected action: {action}")
            
            # Track action distribution
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Execute action in environment
            logger.critical(f"Executing action {action} in environment...")
            # Force window focus before each step
            if hasattr(self.env, 'error_recovery') and hasattr(self.env.error_recovery, 'focus_game_window'):
                focus_success = self.env.error_recovery.focus_game_window()
                logger.critical(f"Pre-step window focus {'succeeded' if focus_success else 'failed'}")
                
            try:
                next_state, reward, done, info = self.env.step(action)
                logger.critical(f"Step complete. Reward: {reward}, Done: {done}")
                if info:
                    logger.critical(f"Step info: {info}")
            except Exception as e:
                logger.critical(f"ERROR during environment step: {e}")
                import traceback
                logger.critical(f"Traceback: {traceback.format_exc()}")
                # Mark observation as invalid for error handling below
                observation_valid = False
                next_state = state
                reward = -10.0
                done = True
                info = {"error": str(e)}
                error_states += 1
                logger.critical("Marked state as invalid due to exception")
            
            # Detect menu or invalid state
            if hasattr(self.env, 'check_menu_state'):
                pre_menu = in_menu
                in_menu = self.env.check_menu_state()
                if in_menu:
                    menu_duration += 1
                    logger.critical(f"Menu detected. Menu duration: {menu_duration}")
                
                # Log menu transitions
                if pre_menu != in_menu:
                    logger.critical(f"Menu transition: {pre_menu} -> {in_menu}")
            
            # Handle invalid observations
            if not observation_valid or (hasattr(self.env, 'is_observation_valid') and not self.env.is_observation_valid(next_state)):
                error_states += 1
                done = True
                reward = -10.0  # Penalty for invalid state
                logger.critical(f"Invalid observation detected. Error states: {error_states}")
                if not done:
                    # If not already done, use the last valid state
                    next_state = state
                    logger.critical("Using last valid state due to invalid observation")
            
            # Store experience for updates
            if observation_valid and (not hasattr(self.env, 'is_observation_valid') or self.env.is_observation_valid(next_state)):
                # Only store valid experiences
                if step == 0:
                    logger.critical("Storing first experience for policy update")
                experience = (state, action, reward, next_state, done, log_prob, value)
                experiences.append(experience)
                logger.critical(f"Added experience to buffer. Buffer size now: {len(experiences)}")
            
            # Update state for next iteration
            if observation_valid:
                state = next_state
            
            # Update running totals
            total_reward += reward
            
            # Increment step counter
            step += 1
            self.total_steps += 1
            
            # Calculate step time
            step_time = time.time() - step_start_time
            logger.critical(f"Step took {step_time:.3f}s")
            
            # Check for resource limits and throttle if needed
            if self.performance_safeguards and step % 10 == 0:
                self.performance_safeguards.check_limits()
                
                # Apply throttling if needed
                throttle_time = self.performance_safeguards.get_throttle_time()
                if throttle_time > 0:
                    logger.critical(f"Throttling for {throttle_time:.2f}s due to resource limits")
                    time.sleep(throttle_time)
            
            # Periodically run updates if we have enough experiences
            if len(experiences) >= self.agent.update_frequency and step % self.agent.update_frequency == 0:
                logger.critical(f"Updating policy with {len(experiences)} experiences")
                self._update_policy(experiences, scaler=scaler)
                experiences = []
                
            # Render if requested
            if render:
                self.env.render()
                
            # Monitor hardware resources if available
            if self.hardware_monitor and step % 10 == 0:
                metrics = self.hardware_monitor.get_metrics()
                
                # Update safeguards with current metrics
                if self.performance_safeguards:
                    self.performance_safeguards.update_metrics(metrics)
                    
                # Log metrics to wandb
                if self.use_wandb and step % 50 == 0:
                    # Remove non-serializable items
                    wandb_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool))}
                    wandb.log({f"hardware/{k}": v for k, v in wandb_metrics.items()})
        
        # Final update with remaining experiences
        if len(experiences) > 0:
            logger.critical(f"Final policy update with {len(experiences)} experiences")
            self._update_policy(experiences, scaler=scaler)
        
        # Calculate episode duration
        episode_duration = time.time() - episode_start_time
        
        logger.critical(f"===== EPISODE {episode_num} COMPLETED =====")
        logger.critical(f"Total steps: {step}")
        logger.critical(f"Total reward: {total_reward}")
        logger.critical(f"Duration: {episode_duration:.2f}s")
        
        # Update stats
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(step)
        self.recent_rewards.append(total_reward)
        if len(self.recent_rewards) > 100:
            self.recent_rewards.pop(0)
        
        # Update best reward
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            # Save best checkpoint
            self._save_checkpoint(episode_num, is_best=True)
        
        # Update visualizer metrics
        self.visualizer.record_episode_metrics(
            episode=episode_num,
            reward=total_reward,
            length=step
        )
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({
                "episode": episode_num,
                "reward": total_reward,
                "steps": step,
                "duration_seconds": episode_duration,
                "menu_duration": menu_duration,
                "error_states": error_states,
                "actions": action_counts,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "best_reward": self.best_reward,
                "mean_reward_100": sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0
            })
        
        # Compile episode statistics
        episode_stats = {
            "episode": episode_num,
            "reward": total_reward,
            "steps": step,
            "duration": episode_duration,
            "menu_duration": menu_duration,
            "error_states": error_states,
            "mean_reward_100": sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "best_reward": self.best_reward,
            "total_steps": self.total_steps,
            "action_distribution": action_counts
        }
        
        # Debug log
        logger.debug(f"Episode {episode_num} completed: reward={total_reward:.2f}, steps={step}, "
                    f"duration={episode_duration:.2f}s, menu_steps={menu_duration}, errors={error_states}")
        
        return episode_stats
    
    def train(self, render: bool = False) -> Dict[str, Any]:
        """Train the agent for the specified number of episodes.
        
        Args:
            render: Whether to render the environment
            
        Returns:
            Dictionary of training statistics
        """
        self.train_start_time = time.time()
        
        logger.critical(f"Starting training from episode {self.start_episode} to {self.num_episodes}")
        
        # Training loop
        for episode in range(self.start_episode, self.num_episodes):
            # Check if exit requested
            if is_exit_requested():
                logger.critical("Exit requested, stopping training loop")
                break
            
            # Train for one episode
            logger.critical(f"Calling train_episode for episode {episode}")
            stats = self.train_episode(episode, render)
            logger.critical(f"train_episode returned stats: {stats}")
            
            # Log episode results
            logger.critical(
                f"Episode {episode}: reward={stats['reward']:.2f}, steps={stats['steps']}, "
                f"mean_reward={stats['mean_reward_100']:.2f}, duration={stats['duration']:.2f}s"
            )
            
            # Generate visualizations periodically
            if episode % 10 == 0 or episode == self.num_episodes - 1:
                self.generate_visualizations()
            
            # Save checkpoint if needed
            if self.checkpoint_manager.should_save_checkpoint(episode):
                self._save_checkpoint(episode, is_best=stats['best_reward'] == stats['reward'])
            
            # Check for autosave
            if self.checkpoint_manager.should_autosave():
                logger.critical("Performing autosave...")
                self._save_checkpoint(episode, is_backup=True)
        
        # Final checkpoint
        logger.critical("Training complete, saving final checkpoint")
        self._save_checkpoint(self.num_episodes-1)
        
        # Final visualizations
        self.generate_visualizations()
        
        # Calculate training duration
        train_duration = time.time() - self.train_start_time
        hours, rem = divmod(train_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        
        # Training summary
        summary = {
            'total_episodes': self.num_episodes - self.start_episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'final_mean_reward': sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0,
            'duration': train_duration,
            'duration_formatted': f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        }
        
        logger.critical(f"Training completed in {summary['duration_formatted']}")
        logger.critical(f"Total steps: {summary['total_steps']}")
        logger.critical(f"Best reward: {summary['best_reward']}")
        logger.critical(f"Final mean reward: {summary['final_mean_reward']}")
        
        return summary
    
    def generate_visualizations(self):
        """Generate visualizations of training progress."""
        try:
            # Generate various visualizations
            self.visualizer.plot_rewards(save=True)
            self.visualizer.plot_losses(save=True)
            self.visualizer.plot_action_distribution(save=True)
            self.visualizer.plot_episode_length_vs_reward(save=True)
            self.visualizer.create_summary_dashboard(save=True)
            
            # Save metrics to CSV for further analysis
            self.visualizer.save_metrics_to_csv()
            
            logger.info("Generated training visualizations")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    def save_progress(self, episode: Optional[int] = None):
        """Save current progress.
        
        Args:
            episode: Current episode number, or None to use the last episode
        """
        if episode is None:
            episode = self.start_episode + len(self.episode_rewards) - 1
            if episode < 0:
                episode = 0
        
        self._save_checkpoint(episode, is_backup=True)
        
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.use_wandb:
                wandb.finish()
            
            # Clean up environment resources
            if hasattr(self.env, 'close'):
                self.env.close()
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def evaluate(self, num_episodes: int = 5, render: bool = True) -> Dict[str, Any]:
        """Evaluate the agent without training.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            Dictionary of evaluation statistics
        """
        logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        # Store original state
        training_mode = self.agent.training
        
        # Set agent to evaluation mode
        self.agent.eval()
        
        # Evaluation stats
        rewards = []
        lengths = []
        start_time = time.time()
        
        for i in range(num_episodes):
            # Collect trajectory without training
            _, episode_reward, episode_steps = self.collect_trajectory(render=render)
            
            rewards.append(episode_reward)
            lengths.append(episode_steps)
            
            logger.info(f"Evaluation episode {i+1}/{num_episodes}: "
                      f"reward={episode_reward:.2f}, steps={episode_steps}")
        
        # Calculate statistics
        mean_reward = sum(rewards) / len(rewards)
        std_reward = np.std(rewards)
        min_reward = min(rewards)
        max_reward = max(rewards)
        mean_length = sum(lengths) / len(lengths)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Create evaluation summary
        eval_summary = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'min_reward': min_reward,
            'max_reward': max_reward,
            'mean_length': mean_length,
            'duration': duration,
        }
        
        # Log results
        logger.info(f"Evaluation results: mean_reward={mean_reward:.2f} ± {std_reward:.2f}, "
                  f"min={min_reward:.2f}, max={max_reward:.2f}")
        
        # Save evaluation results
        try:
            eval_dir = os.path.join(self.visualizer.log_dir, "evaluations")
            os.makedirs(eval_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_file = os.path.join(eval_dir, f"eval_{timestamp}.json")
            
            with open(eval_file, "w") as f:
                json.dump(eval_summary, f, indent=2)
                
            logger.info(f"Saved evaluation results to {eval_file}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
        
        # Restore original state
        if training_mode:
            self.agent.train()
        
        return eval_summary
    
    def _update_policy(self, experiences: List[Dict]) -> Dict[str, float]:
        """Update policy based on collected experiences.
        
        Args:
            experiences: List of collected experiences
            
        Returns:
            Dictionary of statistics from the update
        """
        if not experiences:
            logger.critical("No experiences provided for policy update")
            return {
                'policy_loss': 0,
                'value_loss': 0,
                'entropy': 0,
                'total_loss': 0,
                'clip_fraction': 0,
                'approx_kl': 0
            }
            
        logger.critical(f"Updating policy with {len(experiences)} experiences")
        
        # Check that memory has been initialized
        if self.memory is None:
            logger.critical("Memory not initialized, creating temporary memory")
            self.memory = Memory(self.device)
            
        # Add experiences to memory if needed
        if self.memory.size() == 0:
            logger.critical("Memory is empty, adding experiences")
            for exp in experiences:
                self.memory.add(
                    state=exp['state'],
                    action=exp['action'],
                    reward=exp['reward'],
                    next_state=exp['next_state'],
                    done=exp['done'],
                    log_prob=exp.get('log_prob'),
                    value=exp.get('value'),
                    action_probs=exp.get('action_probs')
                )
                
        # Compute returns and advantages
        logger.critical("Computing returns and advantages")
        self.memory.compute_returns(self.gamma, self.gae_lambda)
        
        # Update policy for multiple epochs
        logger.critical(f"Starting policy update for {self.ppo_epochs} epochs")
        
        # Initialize metrics for tracking
        metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'total_loss': 0,
            'clip_fraction': 0,
            'approx_kl': 0
        }
        
        # Update for multiple epochs
        for epoch in range(self.ppo_epochs):
            logger.critical(f"Policy update epoch {epoch+1}/{self.ppo_epochs}")
            
            # Reset epoch metrics
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_entropy = 0
            epoch_total_loss = 0
            epoch_clip_fraction = 0
            epoch_approx_kl = 0
            num_batches = 0
            
            # Get batch iterator
            batch_iterator = self.memory.get_batch_iterator(self.batch_size, shuffle=True)
            
            # Process each batch
            for batch in batch_iterator:
                # Ensure all tensors are on the correct device
                for key, tensor in batch.items():
                    if tensor.device != self.device:
                        batch[key] = tensor.to(self.device)
                        
                try:
                    # Use mixed precision if enabled
                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision) if torch.cuda.is_available() and self.use_mixed_precision else contextlib.nullcontext():
                        # Forward pass through the model
                        states = batch['states']
                        actions = batch['actions']
                        old_log_probs = batch['old_log_probs']
                        returns = batch['returns']
                        advantages = batch['advantages']
                        
                        # Get current policy outputs
                        action_probs, state_values = self.agent.policy.evaluate_actions(states, actions)
                        
                        # Calculate log probabilities and entropy
                        dist = torch.distributions.Categorical(action_probs)
                        new_log_probs = dist.log_prob(actions)
                        entropy = dist.entropy().mean()
                        
                        # Calculate policy loss with clipping
                        ratio = torch.exp(new_log_probs - old_log_probs)
                        surr1 = ratio * advantages
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Calculate value loss
                        value_loss = F.mse_loss(state_values, returns)
                        
                        # Combined loss
                        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                        
                        # Calculate metrics
                        clip_fraction = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                        approx_kl = 0.5 * ((new_log_probs - old_log_probs) ** 2).mean().item()
                        
                    # Compute gradients
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Clip gradients
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), self.max_grad_norm)
                    
                    # Update parameters
                    self.optimizer.step()
                    
                    # Update metrics
                    epoch_policy_loss += policy_loss.item()
                    epoch_value_loss += value_loss.item()
                    epoch_entropy += entropy.item()
                    epoch_total_loss += loss.item()
                    epoch_clip_fraction += clip_fraction
                    epoch_approx_kl += approx_kl
                    num_batches += 1
                    
                except Exception as e:
                    logger.critical(f"Error during policy update: {e}")
                    import traceback
                    logger.critical(traceback.format_exc())
                    continue
                    
            # Average over batches
            if num_batches > 0:
                metrics['policy_loss'] += epoch_policy_loss / num_batches
                metrics['value_loss'] += epoch_value_loss / num_batches
                metrics['entropy'] += epoch_entropy / num_batches
                metrics['total_loss'] += epoch_total_loss / num_batches
                metrics['clip_fraction'] += epoch_clip_fraction / num_batches
                metrics['approx_kl'] += epoch_approx_kl / num_batches
                
        # Average over epochs
        for key in metrics:
            metrics[key] /= max(1, self.ppo_epochs)
            
        # Update learning rate if scheduler is available
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            metrics['learning_rate'] = self.lr_scheduler.get_last_lr()[0]
            
        # Clear memory after update
        self.memory.clear()
        
        logger.critical(f"Policy update complete: policy_loss={metrics['policy_loss']:.4f}, value_loss={metrics['value_loss']:.4f}")
        return metrics 