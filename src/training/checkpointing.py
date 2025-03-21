"""
Checkpoint management for training.

This module provides functionality for saving and loading checkpoints during training.
"""

import torch
import os
import logging
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import glob
import re
from datetime import datetime
import numpy as np

from src.utils import get_checkpoints_dir, get_logs_dir, get_path, ensure_dir_exists

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages the saving and loading of model checkpoints during training."""
    
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 100,
        autosave_interval: int = 15,
        backup_checkpoints: int = 5,
        max_checkpoints: int = 10,
        max_disk_usage_gb: float = 5.0,
        use_best: bool = False,
        fresh_start: bool = False,
    ):
        """Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints, defaults to 'checkpoints' in project root
            checkpoint_freq: Episodes between checkpoints
            autosave_interval: Minutes between auto-saves
            backup_checkpoints: Number of backup checkpoints to keep
            max_checkpoints: Max regular checkpoints to keep
            max_disk_usage_gb: Maximum disk usage in GB for logs and checkpoints
            use_best: Resume training from best checkpoint instead of latest
            fresh_start: Force starting training from scratch
        """
        if checkpoint_dir is None:
            self.checkpoint_dir = get_checkpoints_dir()
        else:
            self.checkpoint_dir = get_path(checkpoint_dir)
            ensure_dir_exists(self.checkpoint_dir)
            
        self.checkpoint_freq = checkpoint_freq
        self.autosave_interval = autosave_interval
        self.backup_checkpoints = backup_checkpoints
        self.max_checkpoints = max_checkpoints
        self.max_disk_usage_gb = max_disk_usage_gb
        self.use_best = use_best
        self.fresh_start = fresh_start
        
        # Track checkpoint times
        self.last_checkpoint_time = time.time()
        self.last_autosave_time = time.time()
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint file in the checkpoint directory.
        
        Returns:
            Path to the latest checkpoint file, or None if no checkpoints found
        """
        if not self.checkpoint_dir.exists() or not self.checkpoint_dir.is_dir():
            return None
            
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            return None
            
        # Extract episode numbers and find the highest
        episodes = []
        for cp_file in checkpoint_files:
            try:
                episode = int(cp_file.stem.split('_')[1])
                episodes.append((episode, cp_file))
            except (IndexError, ValueError):
                logger.warning(f"Invalid checkpoint filename format: {cp_file}")
                
        if not episodes:
            return None
            
        # Return the file with the highest episode number
        latest_episode, latest_file = max(episodes, key=lambda x: x[0])
        logger.info(f"Found latest checkpoint at episode {latest_episode}: {latest_file}")
        return latest_file
    
    def find_best_checkpoint(self) -> Optional[Path]:
        """Find the best performing checkpoint based on reward.
        
        Returns:
            Path to the best checkpoint file, or None if no checkpoints found
        """
        if not self.checkpoint_dir.exists() or not self.checkpoint_dir.is_dir():
            return None
            
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*_reward_*.pt"))
        if not checkpoint_files:
            return None
            
        # Extract episode numbers and rewards
        checkpoints = []
        for cp_file in checkpoint_files:
            try:
                # Parse filename of format checkpoint_<episode>_reward_<reward>.pt
                parts = cp_file.stem.split('_')
                episode = int(parts[1])
                reward = float(parts[3])
                checkpoints.append((episode, reward, cp_file))
            except (IndexError, ValueError):
                logger.warning(f"Invalid checkpoint filename format: {cp_file}")
                
        if not checkpoints:
            return None
            
        # Return the file with the highest reward
        _, best_reward, best_file = max(checkpoints, key=lambda x: x[1])
        logger.info(f"Found best checkpoint with reward {best_reward}: {best_file}")
        return best_file
    
    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        episode: int,
        is_best: bool = False,
        mean_reward: Optional[float] = None,
    ) -> Path:
        """Save a checkpoint.
        
        Args:
            state_dict: State dictionary to save
            episode: Current episode number
            is_best: Whether this is the best performing checkpoint so far
            mean_reward: Mean reward for this checkpoint
            
        Returns:
            Path to the saved checkpoint file
        """
        # Check disk usage before saving
        self._manage_disk_usage()
        
        # Create filename
        if mean_reward is not None:
            filename = f"checkpoint_{episode}_reward_{mean_reward:.2f}.pt"
        else:
            filename = f"checkpoint_{episode}.pt"
        
        filepath = self.checkpoint_dir / filename
        
        # Save the checkpoint
        try:
            torch.save(state_dict, filepath)
            logger.info(f"Saved checkpoint to {filepath}")
            
            # Update last checkpoint time
            self.last_checkpoint_time = time.time()
            
            # If this is a "best" checkpoint, create a copy or symlink
            if is_best:
                best_path = self.checkpoint_dir / "checkpoint_best.pt"
                shutil.copy2(filepath, best_path)
                logger.info(f"Saved best checkpoint to {best_path}")
                
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return filepath
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, agent, optimizer=None, scheduler=None) -> Tuple[int, float]:
        """Load a checkpoint for the given agent and optimizer.
        
        Args:
            agent: Agent to load checkpoint for
            optimizer: Optional optimizer to load state for
            scheduler: Optional learning rate scheduler to load state for
            
        Returns:
            Tuple of (start_episode, best_reward)
        """
        start_episode = 0
        best_reward = float('-inf')
        
        # Skip loading if fresh start is requested
        if self.fresh_start:
            logger.info("Fresh start requested, skipping checkpoint loading")
            return start_episode, best_reward
        
        # Determine which checkpoint to load
        if self.use_best:
            checkpoint_path = self.find_best_checkpoint()
        else:
            checkpoint_path = self.find_latest_checkpoint()
            
        if checkpoint_path is None:
            logger.info("No checkpoints found, starting from scratch")
            return start_episode, best_reward
            
        # Load the checkpoint
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=agent.device)
            
            # Load agent state
            agent.load_state_dict(checkpoint['agent_state'])
            
            # Load optimizer state if provided
            if optimizer is not None and 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                logger.info("Loaded optimizer state")
                
            # Load scheduler state if provided
            if scheduler is not None and 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
                logger.info("Loaded scheduler state")
            
            # Get episode and best reward
            start_episode = checkpoint.get('episode', 0) + 1
            best_reward = checkpoint.get('best_reward', float('-inf'))
            
            logger.info(f"Resuming from episode {start_episode} with best reward {best_reward}")
            return start_episode, best_reward
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("Starting from scratch due to checkpoint loading failure")
            return 0, float('-inf')
    
    def should_save_checkpoint(self, episode: int) -> bool:
        """Check if we should save a regular checkpoint based on frequency.
        
        Args:
            episode: Current episode number
            
        Returns:
            Whether to save a checkpoint
        """
        return episode > 0 and episode % self.checkpoint_freq == 0
    
    def should_autosave(self) -> bool:
        """Check if we should perform an autosave based on time.
        
        Returns:
            Whether to perform an autosave
        """
        if self.autosave_interval <= 0:
            return False
            
        elapsed = time.time() - self.last_autosave_time
        return elapsed >= self.autosave_interval * 60  # Convert minutes to seconds
    
    def get_checkpoint_path(self, episode: int) -> Optional[Path]:
        """Get the path to a checkpoint for a specific episode.
        
        Args:
            episode: Episode number to find checkpoint for
            
        Returns:
            Path to checkpoint or None if not found
        """
        if not self.checkpoint_dir.exists() or not self.checkpoint_dir.is_dir():
            return None
            
        # Try exact match first
        exact_match = list(self.checkpoint_dir.glob(f"checkpoint_{episode}_*.pt"))
        if exact_match:
            return exact_match[0]
            
        # Try finding closest episode number if exact match not found
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            return None
            
        # Parse episode numbers
        episodes = []
        for cp_file in checkpoint_files:
            try:
                ep = int(cp_file.stem.split('_')[1])
                episodes.append((ep, cp_file))
            except (IndexError, ValueError):
                continue
                
        if not episodes:
            return None
            
        # Find closest episode number
        closest_ep, closest_file = min(episodes, key=lambda x: abs(x[0] - episode))
        logger.debug(f"Found closest checkpoint at episode {closest_ep}: {closest_file}")
        return closest_file
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to prevent disk space issues."""
        try:
            # Get all regular checkpoints
            checkpoint_files = sorted(
                self.checkpoint_dir.glob("checkpoint_*.pt"),
                key=os.path.getmtime
            )
            
            # Filter out best checkpoints and backup checkpoints
            backup_pattern = "checkpoint_*_backup_*.pt"
            best_pattern = "checkpoint_best.pt"
            regular_checkpoints = [
                f for f in checkpoint_files 
                if not f.match(backup_pattern) and not f.match(best_pattern)
            ]
            
            # If we have more than max_checkpoints, remove the oldest ones
            if len(regular_checkpoints) > self.max_checkpoints:
                for old_checkpoint in regular_checkpoints[:-self.max_checkpoints]:
                    try:
                        os.remove(old_checkpoint)
                        logger.info(f"Removed old checkpoint: {old_checkpoint}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")
            
            # Handle backup checkpoints separately
            backup_checkpoints = sorted(
                self.checkpoint_dir.glob("checkpoint_*_backup_*.pt"),
                key=os.path.getmtime
            )
            
            if len(backup_checkpoints) > self.backup_checkpoints:
                for old_backup in backup_checkpoints[:-self.backup_checkpoints]:
                    try:
                        os.remove(old_backup)
                        logger.info(f"Removed old backup checkpoint: {old_backup}")
                    except Exception as e:
                        logger.warning(f"Failed to remove old backup checkpoint {old_backup}: {e}")
                        
        except Exception as e:
            logger.error(f"Error during checkpoint cleanup: {e}")
    
    def _manage_disk_usage(self):
        """Manage disk usage for logs and checkpoints.
        
        Cleans up old checkpoints and logs if they exceed the specified disk usage.
        """
        # Check total size of checkpoints
        checkpoint_size_gb = 0
        if self.checkpoint_dir.exists():
            checkpoint_size_gb = sum(
                os.path.getsize(f) for f in self.checkpoint_dir.glob("**/*") if os.path.isfile(f)
            ) / (1024 ** 3)
        
        # Check total size of logs
        logs_dir = get_logs_dir()
        log_size_gb = 0
        if logs_dir.exists():
            log_size_gb = sum(
                os.path.getsize(f) for f in logs_dir.glob("**/*") if os.path.isfile(f)
            ) / (1024 ** 3)
        
        total_size_gb = checkpoint_size_gb + log_size_gb
        logger.debug(
            f"Current disk usage: {total_size_gb:.2f} GB "
            f"(checkpoints: {checkpoint_size_gb:.2f} GB, logs: {log_size_gb:.2f} GB)"
        )
        
        # Clean up logs if they're using significant space
        if logs_dir.exists() and log_size_gb > 0.5:
            log_files = sorted(
                logs_dir.glob("*.log"),
                key=lambda f: os.path.getmtime(f)
            )
            
            # Keep the 5 most recent logs, remove others
            if len(log_files) > 5:
                old_logs = log_files[:-5]
                for old_log in old_logs:
                    try:
                        logger.info(f"Removing old log file: {old_log}")
                        os.remove(old_log)
                    except Exception as e:
                        logger.warning(f"Failed to remove old log file {old_log}: {e}")
        
        # If we're approaching the limit, take action
        if total_size_gb > self.max_disk_usage_gb * 0.9:
            logger.warning(f"Approaching max disk usage ({total_size_gb:.2f}/{self.max_disk_usage_gb} GB)")
            
            # First, remove some old checkpoints
            self._cleanup_old_checkpoints()
            
            # Check if we're still over the threshold after cleanup
            current_checkpoint_size_gb = sum(
                os.path.getsize(f) for f in self.checkpoint_dir.glob("**/*") if os.path.isfile(f)
            ) / (1024 ** 3)
            
            if current_checkpoint_size_gb + log_size_gb > self.max_disk_usage_gb:
                logger.warning("Still over disk usage threshold after cleanup. Consider increasing limit.")
            
        except Exception as e:
            logger.error(f"Error checking disk usage: {e}") 