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
import torch.optim as optim
import random
from collections import deque

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
        
        # Episode tracking attributes
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_rewards = []
        self.best_reward = float('-inf')
        
        # Helper method to safely get config values
        def get_config_value(key, default_value):
            if hasattr(self.config, "get") and callable(self.config.get):
                return self.config.get(key, default_value)
            elif hasattr(self.config, key):
                return getattr(self.config, key)
            else:
                return default_value
        
        # Set up memory
        self.memory = Memory(self.device)
        
        # Set up optimizer
        self.optimizer = self._create_optimizer()
        
        # Set up learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Training parameters
        self.start_episode = 0
        self.num_episodes = get_config_value("num_episodes", 1000)
        self.total_steps = 0
        
        # Initialize metrics visualization
        run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.visualizer = TrainingVisualizer(os.path.join(tensorboard_dir, f"training_{run_timestamp}"))
        
        # PPO parameters
        self.gamma = get_config_value("gamma", 0.99)
        self.gae_lambda = get_config_value("gae_lambda", 0.95)
        self.clip_param = get_config_value("clip_param", 0.2)
        self.value_loss_coef = get_config_value("value_loss_coef", 0.5)
        self.entropy_coef = get_config_value("entropy_coef", 0.01)
        self.max_grad_norm = get_config_value("max_grad_norm", 0.5)
        self.ppo_epochs = get_config_value("ppo_epochs", 4)
        self.batch_size = get_config_value("batch_size", 64)
        self.use_mixed_precision = get_config_value("mixed_precision", False)
        
        # Hardware monitoring
        self.hardware_monitor = HardwareMonitor() if get_config_value("monitor_hardware", False) else None
        
        # Performance safeguards
        self.performance_safeguards = None
        if get_config_value("enable_performance_safeguards", False):
            try:
                self.performance_safeguards = PerformanceSafeguards(config=config)
                logger.info("Performance safeguards enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize performance safeguards: {e}")
        
        # Wandb integration
        self.use_wandb = get_config_value("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb
                wandb_config = get_config_value("wandb", {})
                wandb.init(
                    project=wandb_config.get("project", "cities-skylines-rl"),
                    entity=wandb_config.get("entity", None),
                    config=config,
                    name=wandb_config.get("run_name", None) or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags=wandb_config.get("tags", []),
                    resume=wandb_config.get("resume", False)
                )
                logger.info("Wandb tracking enabled")
            except ImportError:
                logger.warning("Wandb tracking requested but wandb not installed")
                self.use_wandb = False
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            max_checkpoints=get_config_value("max_checkpoints", 5)
        )
        
        # Load checkpoint if available
        self._load_checkpoint()
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Initialize visualizer
        self.visualizer = TrainingVisualizer(
            log_dir=tensorboard_dir
        )
        
        # Training metrics
        self.episodes_completed = 0
        self.steps_completed = 0
        self.best_reward = float('-inf')
        
        # Training parameters
        self.max_steps = get_config_value("max_steps", 1000)
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
        
        # Helper to get config values safely
        def get_config_value(key, default_value):
            if hasattr(self.config, "get") and callable(self.config.get):
                return self.config.get(key, default_value)
            elif hasattr(self.config, key):
                return getattr(self.config, key)
            else:
                return default_value
        
        # Get optimizer parameters from config
        optimizer_type = get_config_value("optimizer", "adam")
        learning_rate = get_config_value("learning_rate", 3e-4)
        weight_decay = get_config_value("weight_decay", 0.0)
        
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
            momentum = get_config_value("momentum", 0.9)
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
        # Helper to get config values safely
        def get_config_value(key, default_value):
            if hasattr(self.config, "get") and callable(self.config.get):
                return self.config.get(key, default_value)
            elif hasattr(self.config, key):
                return getattr(self.config, key)
            else:
                return default_value
                
        if not get_config_value("use_lr_scheduler", False):
            logger.critical("Not using learning rate scheduler")
            return None
            
        # Get scheduler parameters from config
        scheduler_type = get_config_value("scheduler_type", "step")
        
        # Create scheduler based on type
        if scheduler_type.lower() == 'step':
            step_size = get_config_value("lr_step_size", 100)
            gamma = get_config_value("lr_gamma", 0.9)
            logger.critical(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type.lower() == 'exponential':
            gamma = get_config_value("lr_gamma", 0.9)
            logger.critical(f"Using ExponentialLR scheduler: gamma={gamma}")
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        elif scheduler_type.lower() == 'cosine':
            t_max = get_config_value("cosine_t_max", 100)
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
                self.agent, self.optimizer, self.lr_scheduler
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
            if self.lr_scheduler is not None:
                state['scheduler_state'] = self.lr_scheduler.state_dict()
            
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
        """Train for one episode.

        Args:
            episode_num: Current episode number
            render: Whether to render the environment
            max_steps: Maximum number of steps per episode (overrides config)

        Returns:
            Dictionary containing episode statistics
        """
        if max_steps is None:
            max_steps = self.config.max_steps

        episode_start_time = time.time()
        total_reward = 0.0
        episode_steps = 0
        episode_stats = {
            "episode": episode_num,
            "reward": 0.0,
            "steps": 0,
            "duration": 0.0,
            "fps": 0.0,
            "memory_usage": 0.0,
            "action_distribution": {},
            "error_count": 0
        }

        # Reset agent state
        self.agent.reset()
        self.frame_buffer.clear()

        # Reset environment
        state = self.env.reset()
        if state is None:
            logger.error("Environment reset failed, cannot start episode.")
            return {"episode": episode_num, "reward": 0, "steps": 0, "error": "Reset failed"}

        # Initial frame buffer fill
        stacked_state = self._get_initial_stacked_observation(state)
        if stacked_state is None:
            logger.error("Could not get initial stacked observation.")
            return {"episode": episode_num, "reward": 0, "steps": 0, "error": "Initial observation failed"}

        # Main episode loop
        for step in range(max_steps):
            logger.debug(f"--- Step {step} ---")
            step_start_time = time.time()

            # Check for exit request
            if is_exit_requested():
                logger.info("Exit requested, ending episode")
                break

            # Select action using agent policy
            action_result = self.agent.select_action(stacked_state, info={'step': step})

            # Extract values from action_result dict
            if isinstance(action_result, dict):
                action = action_result.get('action')
                log_prob = action_result.get('log_prob')
                value = action_result.get('value')
                action_probs = action_result.get('action_probs')
            else:
                logger.error("Agent returned invalid action format")
                return {"episode": episode_num, "reward": 0, "steps": 0, "error": "Invalid action format"}

            # Basic validation
            if action is None:
                logger.warning("Agent returned None action, using random action")
                action = torch.randint(0, self.agent.action_dim, (1,), device=self.device)
                log_prob = torch.tensor(0.0, device=self.device)
                value = torch.tensor(0.0, device=self.device)
                action_probs = torch.ones(1, self.agent.action_dim, device=self.device) / self.agent.action_dim

            # Convert action tensor to item if needed by env.step
            action_item = action.item() if isinstance(action, torch.Tensor) else action

            # Execute action in environment
            try:
                next_state_raw, reward, done, info = self.env.step(action_item)
                success = info.get("action_success", True)
            except Exception as e:
                logger.error(f"Error during environment step: {e}")
                next_state_raw = stacked_state  # Use current state as next if error
                reward = -10.0
                done = True
                info = {"error": str(e)}
                success = False
                episode_stats["error_count"] += 1

            # Add new observation to frame buffer and get next stacked state
            self.frame_buffer.append(next_state_raw)
            next_stacked_state = self._get_stacked_observation()

            # Store experience in agent's memory
            self.agent.store_experience(
                stacked_state,
                action,  # Store the tensor action
                reward,
                next_stacked_state,
                done,
                info={'log_prob': log_prob, 'value': value, 'action_probs': action_probs, **info}
            )

            # Update state for next iteration
            stacked_state = next_stacked_state

            # Update episode statistics
            total_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # Update action distribution statistics
            action_idx = action_item
            episode_stats["action_distribution"][action_idx] = episode_stats["action_distribution"].get(action_idx, 0) + 1

            # Check if it's time to update the agent
            if self.agent.memory.size() >= self.config.update_frequency:
                logger.debug(f"Collected {self.agent.memory.size()} steps, triggering agent update.")
                update_metrics = self.agent.update()

                # Log update metrics
                if update_metrics:
                    self.writer.add_scalar('Loss/Policy', update_metrics['actor_loss'], self.total_steps)
                    self.writer.add_scalar('Loss/Value', update_metrics['critic_loss'], self.total_steps)
                    self.writer.add_scalar('Loss/Entropy', update_metrics['entropy'], self.total_steps)
                    self.visualizer.record_training_metrics(
                        update_metrics['actor_loss'],
                        update_metrics['critic_loss'],
                        update_metrics['entropy']
                    )

            # Render if requested
            if render:
                self.env.render()

            # Check for episode termination
            if done:
                logger.debug(f"Episode {episode_num} finished at step {step+1}.")
                break

            # Monitor hardware if enabled
            if self.config.monitor_hardware and self.hardware_monitor:
                self.hardware_monitor.check_resources()

            # Apply performance safeguards if enabled
            if self.performance_safeguards:
                self.performance_safeguards.check_performance()

        # Calculate episode statistics
        episode_duration = time.time() - episode_start_time
        episode_stats.update({
            "reward": total_reward,
            "steps": episode_steps,
            "duration": episode_duration,
            "fps": episode_steps / episode_duration if episode_duration > 0 else 0,
            "memory_usage": torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        })

        # Log episode statistics
        logger.info(f"===== EPISODE {episode_num} COMPLETED =====")
        logger.info(f"Total Reward: {total_reward:.2f}")
        logger.info(f"Steps: {episode_steps}")
        logger.info(f"Duration: {episode_duration:.2f}s")
        logger.info(f"FPS: {episode_stats['fps']:.2f}")
        logger.info(f"Memory Usage: {episode_stats['memory_usage']:.2f}MB")
        logger.info(f"Error Count: {episode_stats['error_count']}")

        # Save checkpoint if needed
        if episode_num % self.config.checkpoint_freq == 0:
            self._save_checkpoint(episode_num, is_best=episode_stats['reward'] == self.best_reward)

        return episode_stats

    def _get_initial_stacked_observation(self, initial_obs):
        """Create the initial stacked observation."""
        self.frame_buffer.clear()
        if initial_obs is None:
            return None
        # Ensure initial_obs is a tensor
        if not isinstance(initial_obs, torch.Tensor):
            initial_obs = torch.as_tensor(initial_obs, device=self.device).float()

        for _ in range(self.frame_stack_size):
            self.frame_buffer.append(initial_obs)
        return self._get_stacked_observation()

    def _get_stacked_observation(self):
        """Stack frames from the buffer."""
        if not self.frame_buffer:
            return None
        # Ensure all frames are tensors and on the correct device
        processed_frames = []
        for frame in self.frame_buffer:
            if frame is None:
                continue  # Skip None frames
            if not isinstance(frame, torch.Tensor):
                frame = torch.as_tensor(frame, device=self.device).float()
            elif frame.device != self.device:
                frame = frame.to(self.device)
            processed_frames.append(frame)

        if not processed_frames:
            return None

        # Stack frames along channel dimension
        try:
            # Check if all frames have the same shape
            first_shape = processed_frames[0].shape
            if not all(f.shape == first_shape for f in processed_frames):
                logger.error("Inconsistent frame shapes in buffer, cannot stack.")
                # Fallback: return the last valid frame repeated
                last_valid_frame = processed_frames[-1]
                return torch.cat([last_valid_frame] * self.frame_stack_size, dim=0)

            stacked = torch.cat(processed_frames, dim=0)  # Assumes frames are [C, H, W]
            return stacked
        except Exception as e:
            logger.error(f"Error stacking frames: {e}")
            # Fallback: return the last valid frame repeated
            last_valid_frame = processed_frames[-1]
            return torch.cat([last_valid_frame] * self.frame_stack_size, dim=0)
    
    def train(self, render: bool = False) -> Dict[str, Any]:
        """Train the agent for the specified number of episodes.
        
        Args:
            render: Whether to render the environment
            
        Returns:
            Dictionary of training statistics
        """
        self.train_start_time = time.time()
        
        logger.critical(f"Starting training from episode {self.start_episode} to {self.num_episodes}")
        
        # Ensure at least 1 episode always runs by setting end_episode to be at least start_episode + 1
        end_episode = max(self.start_episode + 1, self.num_episodes)
        
        # Training loop
        for episode in range(self.start_episode, end_episode):
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
            if episode % 10 == 0 or episode == end_episode - 1:
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
        self._save_checkpoint(end_episode - 1)
        
        # Final visualizations
        self.generate_visualizations()
        
        # Calculate training duration
        train_duration = time.time() - self.train_start_time
        hours, rem = divmod(train_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        
        # Training summary
        summary = {
            'total_episodes': end_episode - self.start_episode,
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
    
    def request_exit(self):
        """Request the trainer to exit cleanly.
        This method is called by signal handlers to request a clean exit."""
        logger.info("Exit requested for trainer")
        # Use the signal_handlers module to set the global exit flag
        request_exit()
            
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
    
    def _prepare_batches(self, states, actions, log_probs, returns, advantages, values=None):
        """Prepare minibatches for policy update.
        
        Args:
            states: State batch
            actions: Action batch
            log_probs: Log probability batch
            returns: Returns batch
            advantages: Advantages batch
            values: Old value predictions batch
            
        Returns:
            List of batches
        """
        # Get agent properties
        use_lstm = False
        sequence_length = 0
        batch_size = 64  # Default batch size
        
        if hasattr(self, 'agent'):
            if hasattr(self.agent, 'use_lstm'):
                use_lstm = self.agent.use_lstm
            if hasattr(self, "sequence_length"):
                sequence_length = self.sequence_length
            elif use_lstm:
                sequence_length = 16  # Default for LSTM
            if hasattr(self, "batch_size"):
                batch_size = self.batch_size
        
        # Convert to tensors if they aren't already (safely)
        if not isinstance(states, torch.Tensor):
            try:
                # If it's a list of tensors, stack them
                if isinstance(states[0], torch.Tensor):
                    states = torch.stack(states)
                else:
                    states = torch.tensor(states, device=self.device)
            except (ValueError, TypeError):
                # Handle potential errors when converting complex structures
                logger.critical(f"Error converting states to tensor, shape: {np.shape(states)}")
                # Convert states to a flat batch if needed
                if hasattr(states, '__len__') and hasattr(states[0], 'shape'):
                    states = torch.cat([s.reshape(1, -1) for s in states], dim=0).to(self.device)
        
        if not isinstance(actions, torch.Tensor):
            try:
                # If it's a list of tensors, stack them
                if isinstance(actions[0], torch.Tensor):
                    actions = torch.stack(actions)
                else:
                    actions = torch.tensor(actions, device=self.device)
            except (ValueError, TypeError):
                logger.critical(f"Error converting actions to tensor, shape: {np.shape(actions)}")
                actions = torch.tensor([a.item() if hasattr(a, 'item') else a for a in actions], device=self.device)
        
        if not isinstance(log_probs, torch.Tensor):
            try:
                # If it's a list of tensors, stack them
                if isinstance(log_probs[0], torch.Tensor):
                    log_probs = torch.stack(log_probs)
                else:
                    log_probs = torch.tensor(log_probs, device=self.device)
            except (ValueError, TypeError):
                logger.critical(f"Error converting log_probs to tensor, shape: {np.shape(log_probs)}")
                log_probs = torch.tensor([lp.item() if hasattr(lp, 'item') else float(lp) for lp in log_probs], device=self.device)
        
        if not isinstance(returns, torch.Tensor):
            try:
                # If it's a list of tensors, stack them
                if isinstance(returns[0], torch.Tensor):
                    returns = torch.stack(returns)
                else:
                    returns = torch.tensor(returns, device=self.device)
            except (ValueError, TypeError):
                logger.critical(f"Error converting returns to tensor, shape: {np.shape(returns)}")
                returns = torch.tensor([r.item() if hasattr(r, 'item') else float(r) for r in returns], device=self.device)
        
        if not isinstance(advantages, torch.Tensor):
            try:
                # If it's a list of tensors, stack them
                if isinstance(advantages[0], torch.Tensor):
                    advantages = torch.stack(advantages)
                else:
                    advantages = torch.tensor(advantages, device=self.device)
            except (ValueError, TypeError):
                logger.critical(f"Error converting advantages to tensor, shape: {np.shape(advantages)}")
                advantages = torch.tensor([a.item() if hasattr(a, 'item') else float(a) for a in advantages], device=self.device)
        
        if values is not None and not isinstance(values, torch.Tensor):
            try:
                # If it's a list of tensors, stack them
                if isinstance(values[0], torch.Tensor):
                    values = torch.stack(values)
                else:
                    values = torch.tensor(values, device=self.device)
            except (ValueError, TypeError):
                logger.critical(f"Error converting values to tensor, shape: {np.shape(values)}")
                values = torch.tensor([v.item() if hasattr(v, 'item') else float(v) for v in values], device=self.device)
            
        # Create batches based on whether we're using LSTM or not
        if use_lstm:
            logger.info(f"Preparing sequence-based batches with sequence length {sequence_length}")
            return self._prepare_sequence_batches(
                states, actions, log_probs, returns, advantages, values,
                sequence_length, batch_size
            )
        else:
            logger.info("Preparing regular mini-batches")
            # Regular mini-batches without sequence consideration
            dataset = torch.utils.data.TensorDataset(
                states, actions, log_probs, returns, advantages,
                values if values is not None else torch.zeros_like(returns)
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            return list(dataloader)
    
    def _prepare_sequence_batches(self, states, actions, log_probs, returns, advantages, values, 
                                  sequence_length, batch_size):
        """Prepare batches that respect sequence boundaries for LSTM training.
        
        Args:
            states: State tensor
            actions: Action tensor
            log_probs: Log probability tensor
            returns: Returns tensor
            advantages: Advantages tensor
            values: Values tensor
            sequence_length: Length of sequences for LSTM
            batch_size: Number of sequences per batch
            
        Returns:
            List of sequence batches
        """
        total_steps = states.size(0)
        logger.info(f"Preparing sequence batches from {total_steps} steps")
        
        # Determine starting indices for sequences
        num_sequences = total_steps // sequence_length
        if num_sequences == 0:
            # Not enough data for even one sequence
            logger.warning(f"Not enough data for sequence batching, falling back to standard batching")
            dataset = torch.utils.data.TensorDataset(
                states, actions, log_probs, returns, advantages,
                values if values is not None else torch.zeros_like(returns)
            )
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
            return list(dataloader)
            
        # Create sequence indices that respect episode boundaries
        sequence_starts = []
        for seq_idx in range(num_sequences):
            start_idx = seq_idx * sequence_length
            sequence_starts.append(start_idx)
            
        # Shuffle the sequence starts
        random.shuffle(sequence_starts)
        
        # Create batches of sequences
        batches = []
        num_batches = (len(sequence_starts) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            # Get the sequence starts for this batch
            batch_starts = sequence_starts[batch_idx * batch_size:
                                          min((batch_idx + 1) * batch_size, len(sequence_starts))]
            
            batch_states = []
            batch_actions = []
            batch_log_probs = []
            batch_returns = []
            batch_advantages = []
            batch_values = []
            
            # Collect sequences for this batch
            for start_idx in batch_starts:
                end_idx = min(start_idx + sequence_length, total_steps)
                batch_states.append(states[start_idx:end_idx])
                batch_actions.append(actions[start_idx:end_idx])
                batch_log_probs.append(log_probs[start_idx:end_idx])
                batch_returns.append(returns[start_idx:end_idx])
                batch_advantages.append(advantages[start_idx:end_idx])
                if values is not None:
                    batch_values.append(values[start_idx:end_idx])
                    
            # Pad sequences if necessary
            for i in range(len(batch_states)):
                if batch_states[i].size(0) < sequence_length:
                    padding_size = sequence_length - batch_states[i].size(0)
                    if len(batch_states[i].shape) > 1:
                        # For multidimensional states
                        padding = torch.zeros(padding_size, *batch_states[i].shape[1:], 
                                             device=self.device)
                    else:
                        # For 1D states
                        padding = torch.zeros(padding_size, device=self.device)
                    
                    batch_states[i] = torch.cat([batch_states[i], padding])
                    batch_actions[i] = torch.cat([batch_actions[i], torch.zeros(padding_size, device=self.device)])
                    batch_log_probs[i] = torch.cat([batch_log_probs[i], torch.zeros(padding_size, device=self.device)])
                    batch_returns[i] = torch.cat([batch_returns[i], torch.zeros(padding_size, device=self.device)])
                    batch_advantages[i] = torch.cat([batch_advantages[i], torch.zeros(padding_size, device=self.device)])
                    if values is not None:
                        batch_values[i] = torch.cat([batch_values[i], torch.zeros(padding_size, device=self.device)])
            
            # Stack sequences into batch tensors
            batch_states = torch.stack(batch_states)  # [batch_size, sequence_length, ...]
            batch_actions = torch.stack(batch_actions)
            batch_log_probs = torch.stack(batch_log_probs)
            batch_returns = torch.stack(batch_returns)
            batch_advantages = torch.stack(batch_advantages)
            if values is not None:
                batch_values = torch.stack(batch_values)
            else:
                batch_values = torch.zeros_like(batch_returns)
                
            # Add to batches
            batches.append((batch_states, batch_actions, batch_log_probs, 
                           batch_returns, batch_advantages, batch_values))
                
        logger.info(f"Created {len(batches)} sequence batches with size {batch_size}")
        return batches
    
    def update_policy(self, states, actions, log_probs, returns, advantages, values=None):
        """Update policy using the collected experiences.
        
        Args:
            states: States from experience buffer
            actions: Actions from experience buffer  
            log_probs: Log probabilities from experience buffer
            returns: Returns from experience buffer
            advantages: Advantages from experience buffer
            values: Values from experience buffer (optional)
            
        Returns:
            Dict of training statistics
        """
        # Get training hyperparameters
        # Try to get values from config based on its type
        clip_param = 0.2  # Default value
        value_coef = 0.5  # Default value
        entropy_coef = 0.01  # Default value
        max_grad_norm = 0.5  # Default value
        num_epochs = 10  # Default value
        
        if hasattr(self.config, "get"):
            # Config is a dict-like object
            clip_param = self.config.get("training", {}).get("clip_param", 0.2)
            value_coef = self.config.get("training", {}).get("value_coef", 0.5)
            entropy_coef = self.config.get("training", {}).get("entropy_coef", 0.01)
            max_grad_norm = self.config.get("training", {}).get("max_grad_norm", 0.5)
            num_epochs = self.config.get("training", {}).get("update_epochs", 10)
        else:
            # Config is likely a HardwareConfig object
            # Try to access attributes directly
            if hasattr(self, "clip_param"):
                clip_param = self.clip_param
            if hasattr(self, "value_coef"):
                value_coef = self.value_coef
            if hasattr(self, "entropy_coef"):
                entropy_coef = self.entropy_coef
            if hasattr(self, "max_grad_norm"):
                max_grad_norm = self.max_grad_norm
            if hasattr(self, "num_epochs"):
                num_epochs = self.num_epochs
            elif hasattr(self, "update_epochs"):
                num_epochs = self.update_epochs

        # Prepare batches for update
        batches = self._prepare_batches(states, actions, log_probs, returns, advantages, values)
        
        # Track metrics
        metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'approx_kl': 0,
            'clip_fraction': 0,
            'explained_variance': 0,
            'update_time': 0,
        }
        
        # Begin timing update
        update_start = time.time()
        
        # Run multiple epochs of training
        use_lstm = False  # Default value
        if hasattr(self.config, "get"):
            use_lstm = self.config.get("model", {}).get("use_lstm", False)
        else:
            # Config is likely a HardwareConfig object
            if hasattr(self.agent, "use_lstm"):
                use_lstm = self.agent.use_lstm
        
        for epoch in range(num_epochs):
            # Process each batch
            for batch in batches:
                batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages, batch_values = batch
                
                # Initialize LSTM hidden state for sequence processing
                hidden_state = None
                
                # Handle batch shape depending on whether we're using sequences
                if use_lstm:
                    # For sequence batches, we have [batch_size, sequence_length, ...] shape
                    batch_size, sequence_length = batch_states.shape[0], batch_states.shape[1]
                    
                    # Process entire sequences
                    self.agent.optimizer.zero_grad()
                    
                    # Reshape tensors to process in batch
                    batch_loss = 0
                    batch_policy_loss = 0
                    batch_value_loss = 0
                    batch_entropy = 0
                    
                    # Get initial hidden state
                    h0 = torch.zeros(1, batch_size, self.agent.policy.lstm_hidden_size, device=self.device)
                    c0 = torch.zeros(1, batch_size, self.agent.policy.lstm_hidden_size, device=self.device)
                    hidden_state = (h0, c0)
                    
                    # Process each step in the sequence
                    for step_idx in range(sequence_length):
                        step_states = batch_states[:, step_idx]
                        step_actions = batch_actions[:, step_idx]
                        step_old_log_probs = batch_old_log_probs[:, step_idx]
                        step_returns = batch_returns[:, step_idx]
                        step_advantages = batch_advantages[:, step_idx]
                        
                        # Forward pass with LSTM state
                        action_probs, values, hidden_state = self.agent.policy(step_states, hidden_state)
                        
                        # Calculate new log probs and entropy
                        dist = torch.distributions.Categorical(action_probs)
                        new_log_probs = dist.log_prob(step_actions)
                        entropy = dist.entropy().mean()
                        
                        # PPO loss calculation
                        ratio = torch.exp(new_log_probs - step_old_log_probs)
                        surr1 = ratio * step_advantages
                        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * step_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        # Ensure values and batch_returns have the same shape for MSE loss
                        if values.shape != batch_returns.shape:
                            # If batch_returns is a vector but values is a 2D tensor with second dim = 1
                            if len(values.shape) == 2 and values.shape[1] == 1 and len(batch_returns.shape) == 1:
                                # Reshape batch_returns to match values
                                batch_returns = batch_returns.unsqueeze(1)
                            # If values is a vector but batch_returns is a 2D tensor with second dim = 1
                            elif len(batch_returns.shape) == 2 and batch_returns.shape[1] == 1 and len(values.shape) == 1:
                                # Reshape values to match batch_returns
                                values = values.unsqueeze(1)
                        
                        value_loss = F.mse_loss(values, batch_returns)
                        
                        # Total loss
                        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                        
                        # Accumulate batch loss
                        batch_loss += loss
                        batch_policy_loss += policy_loss.item()
                        batch_value_loss += value_loss.item()
                        batch_entropy += entropy.item()
                    
                    # Average losses over sequence steps
                    batch_loss /= sequence_length
                    batch_policy_loss /= sequence_length
                    batch_value_loss /= sequence_length
                    batch_entropy /= sequence_length
                    
                    # Backward pass and update
                    batch_loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), max_grad_norm)
                    self.agent.optimizer.step()
                    
                    # Update metrics
                    metrics['policy_loss'] += batch_policy_loss / num_epochs
                    metrics['value_loss'] += batch_value_loss / num_epochs
                    metrics['entropy'] += batch_entropy / num_epochs
                    
                else:
                    # Standard non-sequence processing (original method)
                    self.agent.optimizer.zero_grad()
                    
                    # Forward pass
                    action_probs, values, _ = self.agent.policy(batch_states)
                    dist = torch.distributions.Categorical(action_probs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                    
                    # Ratio for PPO
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    
                    # Clipped surrogate objective
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    # Ensure values and batch_returns have the same shape for MSE loss
                    if values.shape != batch_returns.shape:
                        # If batch_returns is a vector but values is a 2D tensor with second dim = 1
                        if len(values.shape) == 2 and values.shape[1] == 1 and len(batch_returns.shape) == 1:
                            # Reshape batch_returns to match values
                            batch_returns = batch_returns.unsqueeze(1)
                        # If values is a vector but batch_returns is a 2D tensor with second dim = 1
                        elif len(batch_returns.shape) == 2 and batch_returns.shape[1] == 1 and len(values.shape) == 1:
                            # Reshape values to match batch_returns
                            values = values.unsqueeze(1)
                    
                    value_loss = F.mse_loss(values, batch_returns)
                    
                    # Total loss
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                    
                    # Backward pass and optimize
                    loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), max_grad_norm)
                    self.agent.optimizer.step()
                    
                    # Update metrics
                    metrics['policy_loss'] += policy_loss.item() / num_epochs
                    metrics['value_loss'] += value_loss.item() / num_epochs
                    metrics['entropy'] += entropy.item() / num_epochs
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    metrics['approx_kl'] += approx_kl / num_epochs
                    metrics['clip_fraction'] += ((ratio - 1.0).abs() > clip_param).float().mean().item() / num_epochs
        
        # Calculate explained variance
        if values is not None:
            var_y = torch.var(returns)
            if var_y > 0:
                explained_var = 1 - torch.var(returns - values) / var_y
            else:
                explained_var = torch.tensor(0.0, device=self.device)
            metrics['explained_variance'] = explained_var.item()
            
        # Record update time
        metrics['update_time'] = time.time() - update_start
        
        return metrics 