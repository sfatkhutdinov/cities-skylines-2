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

from .checkpointing import CheckpointManager
from .signal_handlers import is_exit_requested, request_exit
from ..config.hardware_config import HardwareConfig
from ..agent.core import PPOAgent
from ..environment.core import Environment
from ..utils.visualization import TrainingVisualizer
from ..utils.hardware_monitor import HardwareMonitor
from ..utils.performance_safeguards import PerformanceSafeguards

logger = logging.getLogger(__name__)

class Trainer:
    """Manages the training process for reinforcement learning."""
    
    def __init__(
        self,
        agent: PPOAgent,
        env: Environment,
        config: HardwareConfig,
        config_dict: Dict[str, Any],
        hardware_monitor: Optional[HardwareMonitor] = None,
        performance_safeguards: Optional[PerformanceSafeguards] = None
    ):
        """Initialize the trainer.
        
        Args:
            agent: The PPO agent
            env: The environment to interact with
            config: Hardware configuration
            config_dict: Training configuration dictionary
            hardware_monitor: Optional hardware monitor instance
            performance_safeguards: Optional performance safeguards instance
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.args = config_dict
        self.hardware_monitor = hardware_monitor
        self.performance_safeguards = performance_safeguards
        
        # Extract training parameters
        self.num_episodes = config_dict.get('num_episodes', 1000)
        self.max_steps = config_dict.get('max_steps', 1000)
        self.checkpoint_dir = config_dict.get('checkpoint_dir', 'checkpoints')
        self.use_wandb = config_dict.get('use_wandb', False)
        
        # Initialize checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_freq=config_dict.get('checkpoint_freq', 100),
            autosave_interval=config_dict.get('autosave_interval', 15),
            backup_checkpoints=config_dict.get('backup_checkpoints', 5),
            max_checkpoints=config_dict.get('max_checkpoints', 10),
            max_disk_usage_gb=config_dict.get('max_disk_usage_gb', 5.0),
            use_best=config_dict.get('use_best', False),
            fresh_start=config_dict.get('fresh_start', False)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=config_dict.get('learning_rate', 1e-4)
        )
        
        # Optional learning rate scheduler
        self.scheduler = None
        if config_dict.get('use_lr_scheduler', False):
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config_dict.get('lr_step_size', 100),
                gamma=config_dict.get('lr_gamma', 0.9)
            )
        
        # Training stats
        self.start_episode = 0
        self.best_reward = float('-inf')
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.recent_rewards = []
        self.train_start_time = None
        
        # Initialize visualizer
        log_dir = config_dict.get('log_dir', os.path.join(self.checkpoint_dir, 'logs'))
        self.visualizer = TrainingVisualizer(log_dir=log_dir)
        
        # Load checkpoint if available
        self._load_checkpoint()
        
        # Initialize Weights & Biases logging
        if self.use_wandb:
            self._init_wandb(config_dict)
    
    def _init_wandb(self, args: Dict[str, Any]):
        """Initialize Weights & Biases logging.
        
        Args:
            args: Training arguments
        """
        try:
            # Configure wandb
            wandb_project = args.get('wandb_project', 'cities-skylines-rl')
            wandb_name = args.get('wandb_name', f"train_{time.strftime('%Y%m%d_%H%M%S')}")
            
            # Get some device information for logging
            device_info = {
                "device": str(self.agent.device),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            }
            
            # Initialize wandb
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                config={
                    **vars(args),
                    **device_info,
                    "agent_type": type(self.agent).__name__,
                    "environment_type": type(self.env).__name__,
                    "resume": self.start_episode > 0
                }
            )
            
            logger.info(f"Initialized Weights & Biases logging with project '{wandb_project}', run '{wandb_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize Weights & Biases: {e}")
            logger.warning("Continuing without Weights & Biases logging")
            self.use_wandb = False
    
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
    
    def collect_trajectory(
        self, 
        render: bool = False
    ) -> Tuple[List[Tuple], float, int]:
        """Collect a trajectory from the environment.
        
        Args:
            render: Whether to render the environment
            
        Returns:
            Tuple of (experiences, total_reward, steps)
        """
        state = self.env.reset()
        
        experiences = []
        done = False
        total_reward = 0
        steps = 0
        
        # For tracking menu transitions
        in_menu = False
        menu_transition_count = 0
        consecutive_menu_steps = 0
        
        # For tracking game crashes
        game_crash_wait_count = 0
        max_crash_wait_steps = 60  # Maximum steps to wait for game restart
        
        for step in range(self.max_steps):
            # Check if exit requested
            if is_exit_requested():
                logger.info("Exit requested during trajectory collection, stopping early")
                break
                
            # Select action
            action, log_prob, value = self.agent.select_action(state)
            
            # Record action for visualization
            self.visualizer.record_action_count(action)
            
            # Check if we're in a menu before taking action
            if hasattr(self.env, 'check_menu_state'):
                pre_action_in_menu = self.env.check_menu_state()
                
                if pre_action_in_menu:
                    consecutive_menu_steps += 1
                    
                    # If stuck in menu for too long, try recovery
                    if consecutive_menu_steps >= 3:
                        logger.info(f"Stuck in menu for {consecutive_menu_steps} steps, attempting recovery")
                        try:
                            self.env.input_simulator.handle_menu_recovery(retries=2)
                        except Exception as e:
                            logger.error(f"Menu recovery failed: {e}")
                        
                        # Re-check menu state after recovery attempt
                        in_menu = self.env.check_menu_state()
                        if not in_menu:
                            logger.info("Successfully recovered from menu state")
                            consecutive_menu_steps = 0
                else:
                    consecutive_menu_steps = 0
            
            # Take action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store in experience buffer
            experience = (state, action, reward, next_state, done, log_prob, value)
            experiences.append(experience)
            
            # Update state and counters
            state = next_state
            total_reward += reward
            steps += 1
            
            # Check for menu transitions if supported
            if hasattr(self.env, 'check_menu_state'):
                post_action_in_menu = self.env.check_menu_state()
                
                # Detect menu transitions
                if post_action_in_menu != in_menu:
                    menu_transition_count += 1
                    logger.debug(f"Menu transition detected: {in_menu} -> {post_action_in_menu}")
                
                in_menu = post_action_in_menu
            
            # Check for game crashes if supported
            if hasattr(self.env, 'check_game_running') and hasattr(self.env, 'restart_game'):
                game_running = self.env.check_game_running()
                
                if not game_running:
                    game_crash_wait_count += 1
                    logger.warning(f"Game appears to have crashed or is not responding. Waiting ({game_crash_wait_count}/{max_crash_wait_steps})")
                    
                    # After waiting a bit, try to restart
                    if game_crash_wait_count >= max_crash_wait_steps:
                        logger.warning("Attempting to restart the game")
                        try:
                            self.env.restart_game()
                            # Reset counters and state
                            game_crash_wait_count = 0
                            state = self.env.reset()
                        except Exception as e:
                            logger.error(f"Failed to restart game: {e}")
                else:
                    game_crash_wait_count = 0
            
            if render:
                self.env.render()
                
            if done:
                break
                
        return experiences, total_reward, steps
    
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
        # Check if using mock environment
        using_mock = hasattr(self.env, '_update_city_state')
        
        # Set agent to training mode
        self.agent.train()
        
        # Get max steps for this episode
        max_steps = max_steps or self.max_steps
        
        # Start episode timer
        episode_start_time = time.time()
        
        # Reset performance tracking
        if self.performance_safeguards:
            self.performance_safeguards.reset()
        
        # Reset environment and get initial state
        state = self.env.reset()
        
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
        
        # Main episode loop
        while not done and step < max_steps:
            # Get step start time for timing
            step_start_time = time.time()
            
            # If we're using the mock environment, we'll handle potential errors directly
            observation_valid = True
            
            # Check for early termination request
            if is_exit_requested():
                logger.info("Exit requested, ending episode early")
                break
            
            # Select action using agent policy
            with torch.set_grad_enabled(False):
                action, log_prob, value = self.agent.select_action(state)
            
            # Track action distribution
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Execute action in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Check for invalid observation
            if not torch.isfinite(next_state).all():
                logger.warning(f"Episode {episode_num}, step {step}: Invalid observation detected")
                observation_valid = False
                error_states += 1
                
                # If using mock environment, we can recover with the previous state
                if using_mock:
                    next_state = state
                    reward = -1.0  # Penalty for invalid state
                
                # In real environment, we'll rely on the error recovery system
                # which should have already handled the issue
            
            # Handle menu detection (if available in info)
            current_in_menu = info.get('in_menu', False)
            if current_in_menu:
                menu_duration += 1
                
                # Penalize menu actions in agent (if supported)
                if hasattr(self.agent, 'register_menu_action') and not in_menu:
                    # First frame of menu detection
                    self.agent.register_menu_action(action)
                
                # Update menu state
                if not in_menu:
                    logger.debug(f"Episode {episode_num}, step {step}: Menu detected")
                in_menu = True
            else:
                if in_menu:
                    logger.debug(f"Episode {episode_num}, step {step}: Exited menu after {menu_duration} steps")
                    menu_duration = 0
                in_menu = False
            
            # Store experience if observation is valid
            if observation_valid:
                experiences.append((state, action, reward, next_state, done, log_prob, value))
                
                # Update total reward
                total_reward += reward
                
                # Update state
                state = next_state
            
            # Increment step counter
            step += 1
            self.total_steps += 1
            
            # Calculate step time
            step_time = time.time() - step_start_time
            
            # Check for resource limits and throttle if needed
            if self.performance_safeguards and step % 10 == 0:
                self.performance_safeguards.check_limits()
                
                # Apply throttling if needed
                throttle_time = self.performance_safeguards.get_throttle_time()
                if throttle_time > 0:
                    logger.debug(f"Throttling for {throttle_time:.2f}s due to resource limits")
                    time.sleep(throttle_time)
            
            # Periodically run updates if we have enough experiences
            if len(experiences) >= self.agent.update_frequency and step % self.agent.update_frequency == 0:
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
            self._update_policy(experiences, scaler=scaler)
        
        # Calculate episode duration
        episode_duration = time.time() - episode_start_time
        
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
        self.visualizer.update(
            episode_num=episode_num,
            episode_reward=total_reward,
            episode_length=step,
            total_steps=self.total_steps,
            action_distribution=action_counts
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
        
        logger.info(f"Starting training from episode {self.start_episode} to {self.num_episodes}")
        
        # Training loop
        for episode in range(self.start_episode, self.num_episodes):
            # Check if exit requested
            if is_exit_requested():
                logger.info("Exit requested, stopping training loop")
                break
            
            # Train for one episode
            stats = self.train_episode(episode, render)
            
            # Log episode results
            logger.info(
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
                logger.info("Performing autosave...")
                self._save_checkpoint(episode, is_backup=True)
        
        # Final checkpoint
        logger.info("Training complete, saving final checkpoint")
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
        
        logger.info(f"Training completed in {summary['duration_formatted']}")
        logger.info(f"Total steps: {summary['total_steps']}")
        logger.info(f"Best reward: {summary['best_reward']}")
        logger.info(f"Final mean reward: {summary['final_mean_reward']}")
        
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
    
    def _update_policy(self, experiences, scaler=None):
        """Update policy based on collected experiences.
        
        Args:
            experiences: List of experience tuples
            scaler: Optional grad scaler for mixed precision training
        """
        # Extract states, actions, rewards, etc. from experiences
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []
        values = []
        
        for exp in experiences:
            state, action, reward, next_state, done, log_prob, value = exp
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)
        
        # Convert to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long, device=self.agent.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.agent.device)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.agent.device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        
        # Calculate returns and advantages
        returns = self._compute_returns(rewards, dones, values)
        advantages = self._compute_advantages(returns, values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # With mixed precision if available
        using_mixed_precision = scaler is not None
        
        # Create mini-batches for update
        batch_size = min(self.config.batch_size, len(states))
        indices = torch.randperm(len(states))
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Track losses for logging
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        # Number of batches
        n_batches = (len(states) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            # Get batch indices
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            
            # Extract batch data
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_log_probs = log_probs[batch_indices]
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]
            
            # Forward pass - with or without mixed precision
            if using_mixed_precision:
                with torch.cuda.amp.autocast():
                    # Get new action distributions and values
                    new_action_dists, new_values = self.agent.network(batch_states)
                    
                    # Calculate new log probs
                    new_log_probs = new_action_dists.log_prob(batch_actions)
                    
                    # Calculate policy loss (PPO objective)
                    ratio = torch.exp(new_log_probs - batch_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.agent.clip_range, 1.0 + self.agent.clip_range) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Calculate value loss
                    value_loss = ((new_values - batch_returns) ** 2).mean()
                    
                    # Calculate entropy bonus
                    entropy_loss = -new_action_dists.entropy().mean()
                    
                    # Calculate total loss
                    loss = policy_loss + self.agent.value_coef * value_loss + self.agent.entropy_coef * entropy_loss
                
                # Backpropagate with scaler
                scaler.scale(loss).backward()
            else:
                # Standard forward pass
                new_action_dists, new_values = self.agent.network(batch_states)
                
                # Calculate new log probs
                new_log_probs = new_action_dists.log_prob(batch_actions)
                
                # Calculate policy loss (PPO objective)
                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.agent.clip_range, 1.0 + self.agent.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = ((new_values - batch_returns) ** 2).mean()
                
                # Calculate entropy bonus
                entropy_loss = -new_action_dists.entropy().mean()
                
                # Calculate total loss
                loss = policy_loss + self.agent.value_coef * value_loss + self.agent.entropy_coef * entropy_loss
                
                # Standard backprop
                loss.backward()
            
            # Track losses for logging
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
        
        # Apply gradients with or without scaler
        if using_mixed_precision:
            # Unscale to enable gradient clipping
            scaler.unscale_(self.optimizer)
            
            # Clip gradients using L2 norm
            torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.agent.max_grad_norm)
            
            # Step with scaler
            scaler.step(self.optimizer)
            scaler.update()
        else:
            # Standard gradient clipping
            torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.agent.max_grad_norm)
            
            # Standard optimizer step
            self.optimizer.step()
        
        # Step learning rate scheduler if available
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update visualizer metrics
        self.visualizer.update_training_metrics(
            actor_loss=sum(policy_losses) / len(policy_losses) if policy_losses else 0.0,
            critic_loss=sum(value_losses) / len(value_losses) if value_losses else 0.0,
            entropy=sum(entropy_losses) / len(entropy_losses) if entropy_losses else 0.0
        )
        
        # Return update statistics
        return {
            'actor_loss': sum(policy_losses) / len(policy_losses) if policy_losses else 0.0,
            'critic_loss': sum(value_losses) / len(value_losses) if value_losses else 0.0,
            'entropy': sum(entropy_losses) / len(entropy_losses) if entropy_losses else 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
    def _compute_returns(self, rewards, dones, values):
        """Compute discounted returns for a trajectory.
        
        Args:
            rewards: Tensor of rewards
            dones: Tensor of done flags
            values: Tensor of value predictions
            
        Returns:
            Tensor of discounted returns
        """
        returns = torch.zeros_like(rewards)
        next_value = 0
        
        # Compute returns in reverse order
        for t in reversed(range(len(rewards))):
            next_value = rewards[t] + self.agent.gamma * next_value * (1 - dones[t])
            returns[t] = next_value
            
        return returns
    
    def _compute_advantages(self, returns, values):
        """Compute advantages for a trajectory.
        
        Args:
            returns: Tensor of discounted returns
            values: Tensor of value predictions
            
        Returns:
            Tensor of advantages
        """
        # Simple advantage computation: returns - values
        advantages = returns - values
        return advantages 