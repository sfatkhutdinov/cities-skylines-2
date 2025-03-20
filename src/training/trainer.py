"""
Trainer module for Cities: Skylines 2 agent.

This module handles the core training loop for the reinforcement learning agent.
"""

import torch
import numpy as np
import logging
import time
import wandb
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import os

from .checkpointing import CheckpointManager
from .signal_handlers import is_exit_requested, request_exit
from ..environment.game_env import CitiesEnvironment
from ..agent.ppo_agent import PPOAgent
from ..config.hardware_config import HardwareConfig

logger = logging.getLogger(__name__)

class Trainer:
    """Handles the training of the reinforcement learning agent."""
    
    def __init__(
        self,
        agent: PPOAgent,
        env: CitiesEnvironment,
        config: HardwareConfig,
        args: Dict[str, Any]
    ):
        """Initialize the trainer.
        
        Args:
            agent: The agent to train
            env: The environment to train in
            config: Hardware configuration
            args: Training arguments
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.args = args
        
        # Extract training parameters
        self.num_episodes = args.get('num_episodes', 1000)
        self.max_steps = args.get('max_steps', 1000)
        self.checkpoint_dir = args.get('checkpoint_dir', 'checkpoints')
        self.use_wandb = args.get('use_wandb', False)
        
        # Initialize checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.checkpoint_dir,
            checkpoint_freq=args.get('checkpoint_freq', 100),
            autosave_interval=args.get('autosave_interval', 15),
            backup_checkpoints=args.get('backup_checkpoints', 5),
            max_checkpoints=args.get('max_checkpoints', 10),
            max_disk_usage_gb=args.get('max_disk_usage_gb', 5.0),
            use_best=args.get('use_best', False),
            fresh_start=args.get('fresh_start', False)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=args.get('learning_rate', 1e-4)
        )
        
        # Optional learning rate scheduler
        self.scheduler = None
        if args.get('use_lr_scheduler', False):
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.get('lr_step_size', 100),
                gamma=args.get('lr_gamma', 0.9)
            )
        
        # Training stats
        self.start_episode = 0
        self.best_reward = float('-inf')
        self.total_steps = 0
        self.episode_rewards = []
        self.recent_rewards = []
        self.train_start_time = None
        
        # Load checkpoint if available
        self._load_checkpoint()
        
        # Initialize Weights & Biases logging
        if self.use_wandb:
            self._init_wandb(args)
    
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
                'episode_rewards': self.episode_rewards
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
            
            # Execute action in environment
            next_state, reward, done, info = self.env.step(action.item())
            
            # Check if game has crashed
            if info.get("game_crashed", False):
                logger.warning(f"Game crash detected (wait count: {game_crash_wait_count}/{max_crash_wait_steps})")
                game_crash_wait_count += 1
                
                # If waiting too long for restart, end the episode
                if game_crash_wait_count >= max_crash_wait_steps:
                    logger.error("Max wait time for game restart exceeded. Ending episode.")
                    done = True
                    break
                    
                # Sleep to reduce CPU usage while waiting
                time.sleep(3.0)
                
                # Skip storing this experience and continue waiting
                continue
            else:
                # Reset crash wait counter if game is running
                game_crash_wait_count = 0
            
            # Check if we just entered or exited a menu
            menu_detected = info.get("menu_detected", False)
            
            # Handle menu transition for learning (if we have menu detection)
            if hasattr(self.env, 'check_menu_state') and not pre_action_in_menu and menu_detected:
                # We just entered a menu
                menu_transition_count += 1
                logger.info(f"Action {action.item()} caused menu transition (count: {menu_transition_count})")
                
                # Register this action as a menu-opening action with the agent
                if hasattr(self.agent, 'register_menu_action'):
                    self.agent.register_menu_action(action.item(), penalty=0.7)
            
            # Store experience
            experiences.append((state, action, reward, next_state, log_prob, value, done))
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            if render:
                self.env.render()
                
            if done:
                break
                
        return experiences, total_reward, steps
    
    def train_episode(self, episode: int, render: bool = False) -> Dict[str, Any]:
        """Train for one episode.
        
        Args:
            episode: Current episode number
            render: Whether to render the environment
            
        Returns:
            Dictionary of episode statistics
        """
        # Start timing the episode
        episode_start_time = time.time()
        
        # Collect trajectory
        experiences, episode_reward, episode_steps = self.collect_trajectory(render)
        
        # Update total steps
        self.total_steps += episode_steps
        
        # Save the reward
        self.episode_rewards.append(episode_reward)
        self.recent_rewards = self.episode_rewards[-100:]  # Keep last 100 rewards
        
        # Check if this is a new best reward
        is_best = False
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            is_best = True
            logger.info(f"New best reward: {self.best_reward}")
        
        # Calculate mean recent reward
        mean_reward = sum(self.recent_rewards) / len(self.recent_rewards)
        
        # Only update the policy if we have enough experiences
        update_policy = len(experiences) >= self.args.get('min_experiences', 10)
        
        # Statistics to return
        stats = {
            'episode': episode,
            'reward': episode_reward,
            'steps': episode_steps,
            'mean_reward': mean_reward,
            'best_reward': self.best_reward,
            'experiences': len(experiences),
            'update_policy': update_policy,
            'is_best': is_best
        }
        
        # Update the policy if we have enough experiences
        if update_policy:
            try:
                # Update the agent
                update_stats = self.agent.update(experiences)
                stats.update(update_stats)
                
                # Step the scheduler if used
                if self.scheduler is not None:
                    self.scheduler.step()
                    stats['learning_rate'] = self.scheduler.get_last_lr()[0]
                
            except Exception as e:
                logger.error(f"Error updating policy: {e}")
                stats['policy_update_error'] = str(e)
        
        # Calculate episode duration
        episode_duration = time.time() - episode_start_time
        stats['duration'] = episode_duration
        
        # Calculate FPS
        if episode_steps > 0:
            fps = episode_steps / episode_duration
            stats['fps'] = fps
        
        # Log to wandb if enabled
        if self.use_wandb:
            try:
                wandb.log(stats)
            except Exception as e:
                logger.error(f"Error logging to wandb: {e}")
        
        return stats
    
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
                f"mean_reward={stats['mean_reward']:.2f}, duration={stats['duration']:.2f}s"
            )
            
            # Save checkpoint if needed
            if self.checkpoint_manager.should_save_checkpoint(episode):
                self._save_checkpoint(episode, is_best=stats['is_best'])
            
            # Check for autosave
            if self.checkpoint_manager.should_autosave():
                logger.info("Performing autosave...")
                self._save_checkpoint(episode, is_backup=True)
        
        # Final checkpoint
        logger.info("Training complete, saving final checkpoint")
        self._save_checkpoint(self.num_episodes-1)
        
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