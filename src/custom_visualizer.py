"""
Simple training visualizer for Cities Skylines 2 agent.

This module provides basic visualization functionality for tracking
and plotting training metrics.
"""

import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class Visualizer:
    """Simple visualizer for agent training metrics."""
    
    def __init__(self, log_dir: Path):
        """Initialize the visualizer.
        
        Args:
            log_dir: Directory to save visualizations
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization subdirectory
        self.viz_dir = self.log_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.rewards = []
        self.episode_lengths = []
        self.losses = []
        self.action_counts = {}
        
        logger.info(f"Visualizer initialized, saving to {self.viz_dir}")
        
    def log_episode_metrics(self, episode: int, reward: float, length: int, 
                          losses: Optional[Dict[str, float]] = None,
                          actions: Optional[List[int]] = None):
        """Log metrics for an episode.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            length: Episode length (number of steps)
            losses: Optional dictionary of loss values
            actions: Optional list of actions taken in episode
        """
        self.rewards.append(reward)
        self.episode_lengths.append(length)
        
        if losses:
            self.losses.append(losses)
        
        if actions:
            for action in actions:
                self.action_counts[action] = self.action_counts.get(action, 0) + 1
        
        # Log to console
        logger.info(f"Episode {episode}: reward={reward:.4f}, length={length}")
    
    def generate_visualizations(self, rewards: Optional[List[float]] = None,
                              episode_lengths: Optional[List[int]] = None):
        """Generate visualizations for training metrics.
        
        Args:
            rewards: Optional list of episode rewards
            episode_lengths: Optional list of episode lengths
        """
        if rewards is None:
            rewards = self.rewards
        
        if episode_lengths is None:
            episode_lengths = self.episode_lengths
        
        if not rewards:
            logger.warning("No rewards data to visualize")
            return
        
        # Create rewards plot
        self._create_rewards_plot(rewards)
        
        # Create episode length plot if available
        if episode_lengths:
            self._create_episode_length_plot(episode_lengths)
            
            # Create episode length vs reward plot
            if len(rewards) == len(episode_lengths):
                self._create_length_vs_reward_plot(episode_lengths, rewards)
        
        # Create loss plot if available
        if self.losses:
            self._create_loss_plot()
        
        # Create action counts plot if available
        if self.action_counts:
            self._create_action_counts_plot()
    
    def _create_rewards_plot(self, rewards: List[float]):
        """Create a plot of episode rewards.
        
        Args:
            rewards: List of episode rewards
        """
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, 'b-', label='Reward')
        
        # Add moving average if enough points
        if len(rewards) >= 5:
            window_size = min(10, len(rewards) // 2)
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(moving_avg)), moving_avg, 'r-', label=f'Moving Avg (w={window_size})')
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plt.savefig(self.viz_dir / "rewards.png")
        plt.close()
        
        logger.info(f"Rewards plot saved to {self.viz_dir / 'rewards.png'}")
    
    def _create_episode_length_plot(self, episode_lengths: List[int]):
        """Create a plot of episode lengths.
        
        Args:
            episode_lengths: List of episode lengths
        """
        plt.figure(figsize=(10, 6))
        plt.plot(episode_lengths, 'g-', label='Length')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Lengths')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plt.savefig(self.viz_dir / "episode_lengths.png")
        plt.close()
        
        logger.info(f"Episode length plot saved to {self.viz_dir / 'episode_lengths.png'}")
    
    def _create_length_vs_reward_plot(self, episode_lengths: List[int], rewards: List[float]):
        """Create a plot of episode length vs reward.
        
        Args:
            episode_lengths: List of episode lengths
            rewards: List of episode rewards
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(episode_lengths, rewards, c=range(len(rewards)), cmap='viridis', alpha=0.7)
        plt.colorbar(label='Episode')
        plt.xlabel('Episode Length (steps)')
        plt.ylabel('Total Reward')
        plt.title('Episode Length vs Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plt.savefig(self.viz_dir / "episode_length_vs_reward.png")
        plt.close()
        
        logger.info(f"Episode length vs reward plot saved to {self.viz_dir / 'episode_length_vs_reward.png'}")
    
    def _create_loss_plot(self):
        """Create a plot of training losses."""
        if not self.losses:
            logger.warning("No loss data to plot")
            return
            
        # Extract loss types and values
        loss_types = self.losses[0].keys()
        episodes = range(len(self.losses))
        
        plt.figure(figsize=(12, 6))
        
        for loss_type in loss_types:
            loss_values = [loss.get(loss_type, 0) for loss in self.losses]
            plt.plot(episodes, loss_values, label=loss_type)
        
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        plt.savefig(self.viz_dir / "losses.png")
        plt.close()
        
        logger.info(f"Loss plot saved to {self.viz_dir / 'losses.png'}")
    
    def _create_action_counts_plot(self):
        """Create a plot of action counts."""
        if not self.action_counts:
            logger.warning("No action count data to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        actions = list(self.action_counts.keys())
        counts = list(self.action_counts.values())
        
        # Sort by action ID
        sorted_indices = np.argsort(actions)
        sorted_actions = [actions[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]
        
        plt.bar(sorted_actions, sorted_counts)
        plt.xlabel('Action ID')
        plt.ylabel('Count')
        plt.title('Action Counts')
        plt.xticks(sorted_actions)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Save plot
        plt.savefig(self.viz_dir / "action_counts.png")
        plt.close()
        
        logger.info(f"Action counts plot saved to {self.viz_dir / 'action_counts.png'}")
    
    def generate_dashboard(self, rewards: Optional[List[float]] = None, 
                         episode_lengths: Optional[List[int]] = None):
        """Generate a dashboard with all visualizations.
        
        Args:
            rewards: Optional list of episode rewards
            episode_lengths: Optional list of episode lengths
        """
        if rewards is None:
            rewards = self.rewards
        
        if episode_lengths is None:
            episode_lengths = self.episode_lengths
        
        if not rewards:
            logger.warning("No rewards data for dashboard")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Rewards subplot
        plt.subplot(2, 2, 1)
        plt.plot(rewards, 'b-', label='Reward')
        if len(rewards) >= 5:
            window_size = min(10, len(rewards) // 2)
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, window_size-1+len(moving_avg)), moving_avg, 'r-', label=f'MA (w={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Episode length subplot
        plt.subplot(2, 2, 2)
        if episode_lengths:
            plt.plot(episode_lengths, 'g-')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.title('Episode Lengths')
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, 'No episode length data', horizontalalignment='center', verticalalignment='center')
            plt.title('Episode Lengths')
        
        # Length vs reward subplot
        plt.subplot(2, 2, 3)
        if episode_lengths and len(rewards) == len(episode_lengths):
            plt.scatter(episode_lengths, rewards, c=range(len(rewards)), cmap='viridis', alpha=0.7)
            plt.colorbar(label='Episode')
            plt.xlabel('Episode Length')
            plt.ylabel('Total Reward')
            plt.title('Length vs Reward')
            plt.grid(True, linestyle='--', alpha=0.7)
        else:
            plt.text(0.5, 0.5, 'Insufficient data', horizontalalignment='center', verticalalignment='center')
            plt.title('Length vs Reward')
        
        # Action counts subplot
        plt.subplot(2, 2, 4)
        if self.action_counts:
            actions = list(self.action_counts.keys())
            counts = list(self.action_counts.values())
            sorted_indices = np.argsort(actions)
            sorted_actions = [actions[i] for i in sorted_indices]
            sorted_counts = [counts[i] for i in sorted_indices]
            plt.bar(sorted_actions, sorted_counts)
            plt.xlabel('Action ID')
            plt.ylabel('Count')
            plt.title('Action Counts')
            plt.xticks(sorted_actions)
            plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        else:
            plt.text(0.5, 0.5, 'No action count data', horizontalalignment='center', verticalalignment='center')
            plt.title('Action Counts')
        
        plt.tight_layout()
        
        # Save dashboard
        plt.savefig(self.viz_dir / "dashboard.png")
        plt.close()
        
        logger.info(f"Dashboard saved to {self.viz_dir / 'dashboard.png'}")
        
        # Also save metrics to CSV
        try:
            import pandas as pd
            metrics = {
                'episode': list(range(len(rewards))),
                'reward': rewards
            }
            
            if episode_lengths and len(rewards) == len(episode_lengths):
                metrics['length'] = episode_lengths
                
            df = pd.DataFrame(metrics)
            df.to_csv(self.viz_dir / "metrics.csv", index=False)
            logger.info(f"Metrics saved to CSV: {self.viz_dir / 'metrics.csv'}")
        except ImportError:
            logger.warning("Pandas not available, skipping CSV export") 