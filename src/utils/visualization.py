#!/usr/bin/env python3
"""
Visualization utilities for training.

This module provides classes and functions for visualizing training metrics.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
from typing import Dict, List, Optional, Union, Tuple, Any


class TrainingVisualizer:
    """Class for visualizing training metrics."""
    
    def __init__(self, log_dir: str, output_dir: Optional[str] = None):
        """Initialize the training visualizer.
        
        Args:
            log_dir: Directory containing log files
            output_dir: Directory to save visualizations (defaults to log_dir/visualizations)
        """
        self.log_dir = log_dir
        self.output_dir = output_dir or os.path.join(log_dir, "visualizations")
        self.metrics = defaultdict(list)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def update_metrics(self, metrics_dict: Dict[str, Any]):
        """Update metrics with new values.
        
        Args:
            metrics_dict: Dictionary of metrics to update
        """
        for key, value in metrics_dict.items():
            if isinstance(value, list):
                self.metrics[key].extend(value)
            else:
                self.metrics[key].append(value)
    
    def record_episode_metrics(self, episode: int, reward: float, length: int):
        """Record metrics for a completed episode.
        
        Args:
            episode: Episode number
            reward: Episode reward
            length: Episode length
        """
        self.metrics["episode_rewards"].append(reward)
        self.metrics["episode_lengths"].append(length)
        self.metrics["episodes"].append(episode)
    
    def record_training_metrics(self, actor_loss: float, critic_loss: float, entropy: float):
        """Record training metrics.
        
        Args:
            actor_loss: Actor loss
            critic_loss: Critic loss
            entropy: Entropy
        """
        self.metrics["actor_losses"].append(float(actor_loss))
        self.metrics["critic_losses"].append(float(critic_loss))
        self.metrics["entropies"].append(float(entropy))
    
    def record_action_count(self, action: int):
        """Record an action taken by the agent.
        
        Args:
            action: Action taken
        """
        if "action_counts" not in self.metrics:
            self.metrics["action_counts"] = {}
        
        action_str = str(action)
        if action_str in self.metrics["action_counts"]:
            self.metrics["action_counts"][action_str] += 1
        else:
            self.metrics["action_counts"][action_str] = 1
    
    def save_metrics(self, filename: str = "training_log.json"):
        """Save metrics to a JSON file.
        
        Args:
            filename: Name of the file to save metrics to
        """
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            # Convert any numpy values to Python types
            clean_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, list):
                    clean_metrics[key] = [float(x) if isinstance(x, (np.float32, np.float64)) else x for x in value]
                elif isinstance(value, dict):
                    clean_metrics[key] = {k: int(v) if isinstance(v, (np.int32, np.int64)) else v 
                                        for k, v in value.items()}
                else:
                    clean_metrics[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
            
            with open(filepath, "w") as f:
                json.dump(clean_metrics, f, indent=2)
            
            logging.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save metrics: {e}")
    
    def save_metrics_to_csv(self, filename: str = "metrics.csv"):
        """Save metrics to a CSV file.
        
        Args:
            filename: Name of the file to save metrics to
        """
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Convert metrics to DataFrame
            data = {}
            
            # Basic episode metrics
            num_episodes = len(self.metrics["episode_rewards"])
            data["episode"] = list(range(1, num_episodes + 1))
            data["reward"] = self.metrics["episode_rewards"]
            data["length"] = self.metrics["episode_lengths"]
            
            # Add losses if available
            if "actor_losses" in self.metrics and len(self.metrics["actor_losses"]) > 0:
                # If we have fewer losses than episodes (because updates happen less frequently)
                # we need to handle that by creating NaN values
                num_losses = len(self.metrics["actor_losses"])
                loss_data = [np.nan] * num_episodes
                
                # Fill in the available losses
                if num_losses <= num_episodes:
                    loss_data[-num_losses:] = self.metrics["actor_losses"]
                else:
                    # If we have more losses than episodes, just take the last num_episodes losses
                    loss_data = self.metrics["actor_losses"][-num_episodes:]
                
                data["actor_loss"] = loss_data
                
                # Do the same for critic losses and entropy
                if "critic_losses" in self.metrics and len(self.metrics["critic_losses"]) > 0:
                    num_losses = len(self.metrics["critic_losses"])
                    loss_data = [np.nan] * num_episodes
                    if num_losses <= num_episodes:
                        loss_data[-num_losses:] = self.metrics["critic_losses"]
                    else:
                        loss_data = self.metrics["critic_losses"][-num_episodes:]
                    data["critic_loss"] = loss_data
                
                if "entropies" in self.metrics and len(self.metrics["entropies"]) > 0:
                    num_losses = len(self.metrics["entropies"])
                    loss_data = [np.nan] * num_episodes
                    if num_losses <= num_episodes:
                        loss_data[-num_losses:] = self.metrics["entropies"]
                    else:
                        loss_data = self.metrics["entropies"][-num_episodes:]
                    data["entropy"] = loss_data
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
            logging.info(f"Metrics saved to CSV: {filepath}")
        except Exception as e:
            logging.error(f"Failed to save metrics to CSV: {e}")
    
    def plot_rewards(self, window_size: int = 10, 
                    save: bool = True, 
                    filename: str = "rewards.png") -> Optional[plt.Figure]:
        """Plot episode rewards.
        
        Args:
            window_size: Window size for moving average
            save: Whether to save the plot
            filename: Name of the file to save the plot to
            
        Returns:
            Matplotlib figure if successful, None otherwise
        """
        try:
            if "episode_rewards" not in self.metrics or len(self.metrics["episode_rewards"]) < 2:
                logging.warning("Not enough reward data to plot")
                return None
            
            rewards = self.metrics["episode_rewards"]
            episodes = list(range(1, len(rewards) + 1))
            
            # Calculate moving average
            if window_size > 0 and len(rewards) > window_size:
                moving_avg = []
                for i in range(len(rewards)):
                    if i < window_size:
                        moving_avg.append(np.mean(rewards[:i+1]))
                    else:
                        moving_avg.append(np.mean(rewards[i-window_size+1:i+1]))
            else:
                moving_avg = rewards
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot rewards and moving average
            ax.plot(episodes, rewards, "b-", alpha=0.3, label="Episode Reward")
            ax.plot(episodes, moving_avg, "r-", label=f"{window_size}-Episode Moving Avg")
            
            # Add labels and title
            ax.set_xlabel("Episode")
            ax.set_ylabel("Reward")
            ax.set_title("Episode Rewards")
            ax.legend()
            ax.grid(alpha=0.3)
            
            # Add stats as text in the corner
            stats_text = (
                f"Total Episodes: {len(rewards)}\n"
                f"Latest Reward: {rewards[-1]:.2f}\n"
                f"Moving Avg: {moving_avg[-1]:.2f}\n"
                f"Max Reward: {max(rewards):.2f}\n"
                f"Min Reward: {min(rewards):.2f}\n"
                f"Avg Reward: {np.mean(rewards):.2f}"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment="top", fontsize=9, 
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            # Save if requested
            if save:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                logging.info(f"Rewards plot saved to {filepath}")
            
            return fig
        except Exception as e:
            logging.error(f"Failed to plot rewards: {e}")
            return None
    
    def plot_losses(self, save: bool = True, 
                  filename: str = "losses.png") -> Optional[plt.Figure]:
        """Plot training losses.
        
        Args:
            save: Whether to save the plot
            filename: Name of the file to save the plot to
            
        Returns:
            Matplotlib figure if successful, None otherwise
        """
        try:
            if "actor_losses" not in self.metrics or len(self.metrics["actor_losses"]) < 2:
                logging.warning("Not enough loss data to plot")
                return None
            
            actor_losses = self.metrics["actor_losses"]
            critic_losses = self.metrics.get("critic_losses", [])
            entropies = self.metrics.get("entropies", [])
            
            updates = list(range(1, len(actor_losses) + 1))
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
            
            # Plot actor losses
            axes[0].plot(updates, actor_losses, "b-")
            axes[0].set_ylabel("Actor Loss")
            axes[0].set_title("Actor Loss")
            axes[0].grid(alpha=0.3)
            
            # Plot critic losses if available
            if critic_losses:
                axes[1].plot(updates, critic_losses, "r-")
                axes[1].set_ylabel("Critic Loss")
                axes[1].set_title("Critic Loss")
                axes[1].grid(alpha=0.3)
            else:
                axes[1].set_visible(False)
            
            # Plot entropy if available
            if entropies:
                axes[2].plot(updates, entropies, "g-")
                axes[2].set_ylabel("Entropy")
                axes[2].set_title("Entropy")
                axes[2].grid(alpha=0.3)
            else:
                axes[2].set_visible(False)
            
            # Set x-axis label
            axes[-1].set_xlabel("Update")
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if requested
            if save:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                logging.info(f"Losses plot saved to {filepath}")
            
            return fig
        except Exception as e:
            logging.error(f"Failed to plot losses: {e}")
            return None
    
    def plot_action_distribution(self, action_counts: Optional[Dict[str, int]] = None,
                                save: bool = True, 
                                filename: str = "action_distribution.png") -> Optional[plt.Figure]:
        """Plot action distribution.
        
        Args:
            action_counts: Dictionary of action counts (if None, uses metrics["action_counts"])
            save: Whether to save the plot
            filename: Name of the file to save the plot to
            
        Returns:
            Matplotlib figure if successful, None otherwise
        """
        try:
            # Use provided action_counts or get from metrics
            counts = action_counts or self.metrics.get("action_counts", {})
            
            if not counts:
                logging.warning("No action counts data to plot")
                return None
            
            # Sort by action index
            actions = sorted(counts.keys(), key=lambda x: int(x) if x.isdigit() else float("inf"))
            counts_sorted = [counts[a] for a in actions]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar chart
            bars = ax.bar(actions, counts_sorted)
            
            # Add counts as text on top of bars
            for bar, count in zip(bars, counts_sorted):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                       str(count), ha="center", va="bottom")
            
            # Add labels and title
            ax.set_xlabel("Action")
            ax.set_ylabel("Count")
            ax.set_title("Action Distribution")
            
            # Set x-ticks explicitly to ensure all actions are shown
            ax.set_xticks(list(range(len(actions))))
            ax.set_xticklabels(actions)
            
            # Add grid
            ax.grid(axis="y", alpha=0.3)
            
            # Save if requested
            if save:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                logging.info(f"Action distribution plot saved to {filepath}")
            
            return fig
        except Exception as e:
            logging.error(f"Failed to plot action distribution: {e}")
            return None
    
    def plot_episode_length_vs_reward(self, save: bool = True, 
                                    filename: str = "episode_length_vs_reward.png") -> Optional[plt.Figure]:
        """Plot episode length vs reward.
        
        Args:
            save: Whether to save the plot
            filename: Name of the file to save the plot to
            
        Returns:
            Matplotlib figure if successful, None otherwise
        """
        try:
            if ("episode_rewards" not in self.metrics or 
                "episode_lengths" not in self.metrics or 
                len(self.metrics["episode_rewards"]) < 2):
                logging.warning("Not enough data to plot episode length vs reward")
                return None
            
            rewards = self.metrics["episode_rewards"]
            lengths = self.metrics["episode_lengths"]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create scatter plot
            scatter = ax.scatter(lengths, rewards, alpha=0.6, c=range(len(rewards)), cmap="viridis")
            
            # Add labels and title
            ax.set_xlabel("Episode Length")
            ax.set_ylabel("Reward")
            ax.set_title("Episode Length vs Reward")
            
            # Add colorbar to show episode number
            cbar = plt.colorbar(scatter)
            cbar.set_label("Episode")
            
            # Add grid
            ax.grid(alpha=0.3)
            
            # Add trend line
            if len(rewards) > 2:
                z = np.polyfit(lengths, rewards, 1)
                p = np.poly1d(z)
                ax.plot(sorted(lengths), p(sorted(lengths)), "r--", alpha=0.8)
                
                # Add correlation coefficient
                correlation = np.corrcoef(lengths, rewards)[0, 1]
                ax.text(0.02, 0.98, f"Correlation: {correlation:.2f}", transform=ax.transAxes, 
                       verticalalignment="top", fontsize=9, 
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            # Save if requested
            if save:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                logging.info(f"Episode length vs reward plot saved to {filepath}")
            
            return fig
        except Exception as e:
            logging.error(f"Failed to plot episode length vs reward: {e}")
            return None
    
    def create_summary_dashboard(self, save: bool = True, 
                               filename: str = "dashboard.png") -> Optional[plt.Figure]:
        """Create a summary dashboard with multiple plots.
        
        Args:
            save: Whether to save the plot
            filename: Name of the file to save the plot to
            
        Returns:
            Matplotlib figure if successful, None otherwise
        """
        try:
            if "episode_rewards" not in self.metrics or len(self.metrics["episode_rewards"]) < 2:
                logging.warning("Not enough data to create dashboard")
                return None
            
            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Set up grid for subplots
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # 1. Rewards plot
            ax1 = fig.add_subplot(gs[0, 0])
            rewards = self.metrics["episode_rewards"]
            episodes = list(range(1, len(rewards) + 1))
            
            # Calculate moving average
            window_size = min(10, len(rewards))
            moving_avg = []
            for i in range(len(rewards)):
                if i < window_size:
                    moving_avg.append(np.mean(rewards[:i+1]))
                else:
                    moving_avg.append(np.mean(rewards[i-window_size+1:i+1]))
            
            ax1.plot(episodes, rewards, "b-", alpha=0.3, label="Episode Reward")
            ax1.plot(episodes, moving_avg, "r-", label=f"{window_size}-Episode Moving Avg")
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.set_title("Episode Rewards")
            ax1.legend(loc="lower right")
            ax1.grid(alpha=0.3)
            
            # 2. Episode length plot
            ax2 = fig.add_subplot(gs[0, 1])
            lengths = self.metrics["episode_lengths"]
            
            # Calculate moving average for lengths
            moving_avg_lengths = []
            for i in range(len(lengths)):
                if i < window_size:
                    moving_avg_lengths.append(np.mean(lengths[:i+1]))
                else:
                    moving_avg_lengths.append(np.mean(lengths[i-window_size+1:i+1]))
            
            ax2.plot(episodes, lengths, "g-", alpha=0.3, label="Episode Length")
            ax2.plot(episodes, moving_avg_lengths, "m-", label=f"{window_size}-Episode Moving Avg")
            ax2.set_xlabel("Episode")
            ax2.set_ylabel("Length")
            ax2.set_title("Episode Lengths")
            ax2.legend(loc="lower right")
            ax2.grid(alpha=0.3)
            
            # 3. Action distribution
            ax3 = fig.add_subplot(gs[1, 0])
            counts = self.metrics.get("action_counts", {})
            
            if counts:
                # Sort by action index
                actions = sorted(counts.keys(), key=lambda x: int(x) if x.isdigit() else float("inf"))
                counts_sorted = [counts[a] for a in actions]
                
                bars = ax3.bar(actions, counts_sorted)
                
                # Add counts as text on top of bars
                for bar, count in zip(bars, counts_sorted):
                    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                           str(count), ha="center", va="bottom", fontsize=8)
                
                ax3.set_xlabel("Action")
                ax3.set_ylabel("Count")
                ax3.set_title("Action Distribution")
                ax3.grid(axis="y", alpha=0.3)
            else:
                ax3.text(0.5, 0.5, "No action data available", 
                        ha="center", va="center", fontsize=12)
                ax3.set_title("Action Distribution")
                ax3.axis("off")
            
            # 4. Episode length vs reward
            ax4 = fig.add_subplot(gs[1, 1])
            if len(rewards) > 2 and len(lengths) == len(rewards):
                scatter = ax4.scatter(lengths, rewards, alpha=0.6, c=range(len(rewards)), cmap="viridis")
                ax4.set_xlabel("Episode Length")
                ax4.set_ylabel("Reward")
                ax4.set_title("Episode Length vs Reward")
                
                # Add colorbar to show episode number
                cbar = plt.colorbar(scatter, ax=ax4)
                cbar.set_label("Episode")
                
                # Add grid
                ax4.grid(alpha=0.3)
                
                # Add trend line
                z = np.polyfit(lengths, rewards, 1)
                p = np.poly1d(z)
                ax4.plot(sorted(lengths), p(sorted(lengths)), "r--", alpha=0.8)
                
                # Add correlation coefficient
                correlation = np.corrcoef(lengths, rewards)[0, 1]
                ax4.text(0.02, 0.98, f"Correlation: {correlation:.2f}", transform=ax4.transAxes, 
                        verticalalignment="top", fontsize=9, 
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            else:
                ax4.text(0.5, 0.5, "Not enough data for scatter plot", 
                        ha="center", va="center", fontsize=12)
                ax4.set_title("Episode Length vs Reward")
                ax4.axis("off")
            
            # Add summary statistics
            plt.figtext(0.5, 0.01, 
                       f"Summary: {len(rewards)} Episodes, " 
                       f"Latest Reward: {rewards[-1]:.2f}, "
                       f"Moving Avg: {moving_avg[-1]:.2f}, "
                       f"Max Reward: {max(rewards):.2f}, "
                       f"Avg Reward: {np.mean(rewards):.2f}",
                       ha="center", fontsize=10,
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            
            # Save if requested
            if save:
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                logging.info(f"Dashboard saved to {filepath}")
            
            return fig
        except Exception as e:
            logging.error(f"Failed to create dashboard: {e}")
            return None 