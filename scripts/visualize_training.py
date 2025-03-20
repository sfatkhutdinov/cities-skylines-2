#!/usr/bin/env python3
"""
Script to visualize training results.

This script loads training data from checkpoints and generates visualizations.
"""

import argparse
import logging
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.visualization import TrainingVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory containing checkpoints")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save visualizations (defaults to checkpoint_dir/visualizations)")
    parser.add_argument("--log_file", type=str, default=None, help="Specific log file to visualize")
    parser.add_argument("--checkpoint_file", type=str, default=None, help="Specific checkpoint file to visualize")
    parser.add_argument("--show", action="store_true", help="Show plots instead of saving them")
    parser.add_argument("--rewards", action="store_true", help="Plot rewards")
    parser.add_argument("--losses", action="store_true", help="Plot losses")
    parser.add_argument("--actions", action="store_true", help="Plot action distribution")
    parser.add_argument("--scatter", action="store_true", help="Plot episode length vs reward scatter plot")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--window", type=int, default=10, help="Window size for moving average")
    return parser.parse_args()


def load_checkpoint_data(checkpoint_path):
    """Load data from checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        return checkpoint
    except Exception as e:
        logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def load_log_data(log_path):
    """Load data from log file."""
    try:
        with open(log_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load log file {log_path}: {e}")
        return None


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in the checkpoint directory."""
    try:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") and not f.endswith("_best.pt")]
        if not checkpoint_files:
            return None
        
        # Extract episode numbers from filenames
        episodes = []
        for filename in checkpoint_files:
            try:
                # Extract episode number, assuming format "checkpoint_EPISODE.pt"
                episode = int(filename.split("_")[1].split(".")[0])
                episodes.append((episode, os.path.join(checkpoint_dir, filename)))
            except (IndexError, ValueError):
                continue
        
        if not episodes:
            return None
        
        # Find the highest episode number
        latest_episode, latest_file = max(episodes, key=lambda x: x[0])
        return latest_file
    except Exception as e:
        logging.error(f"Failed to find latest checkpoint: {e}")
        return None


def main():
    """Main function to visualize training results."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Determine checkpoint file
    checkpoint_path = None
    if args.checkpoint_file:
        checkpoint_path = args.checkpoint_file
    else:
        latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if latest_checkpoint:
            checkpoint_path = latest_checkpoint
            logging.info(f"Using latest checkpoint: {checkpoint_path}")
        else:
            logging.error(f"No checkpoint found in {args.checkpoint_dir}")
            return 1
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint_data(checkpoint_path)
    if not checkpoint_data:
        return 1
    
    # Initialize visualizer
    output_dir = args.output_dir or os.path.join(args.checkpoint_dir, "visualizations")
    visualizer = TrainingVisualizer(
        log_dir=args.checkpoint_dir,
        output_dir=output_dir
    )
    
    # Load metrics from checkpoint
    metrics = checkpoint_data.get("metrics", {})
    if not metrics:
        logging.warning("No metrics found in checkpoint")
        
        # Try to reconstruct basic metrics from checkpoint
        episode_rewards = checkpoint_data.get("episode_rewards", [])
        episode_lengths = checkpoint_data.get("episode_lengths", [])
        
        if episode_rewards:
            metrics["episode_rewards"] = episode_rewards
        if episode_lengths:
            metrics["episode_lengths"] = episode_lengths
    
    # Update visualizer metrics
    for key, values in metrics.items():
        visualizer.metrics[key] = values
    
    # Load action counts
    action_counts = checkpoint_data.get("action_counts", {})
    
    # Generate visualizations
    created_visualizations = False
    
    # Determine which plots to generate
    generate_all = args.all
    generate_rewards = args.rewards or generate_all
    generate_losses = args.losses or generate_all
    generate_actions = args.actions or generate_all
    generate_scatter = args.scatter or generate_all
    
    # If no specific plots were requested, generate all
    if not any([generate_rewards, generate_losses, generate_actions, generate_scatter]):
        generate_all = True
        generate_rewards = generate_losses = generate_actions = generate_scatter = True
    
    # Generate plots
    if generate_rewards:
        reward_fig = visualizer.plot_rewards(window_size=args.window, save=not args.show)
        if reward_fig:
            created_visualizations = True
            if args.show:
                plt.figure(reward_fig.number)
                plt.show()
    
    if generate_losses:
        loss_fig = visualizer.plot_losses(save=not args.show)
        if loss_fig:
            created_visualizations = True
            if args.show:
                plt.figure(loss_fig.number)
                plt.show()
    
    if generate_actions and action_counts:
        action_fig = visualizer.plot_action_distribution(action_counts, save=not args.show)
        if action_fig:
            created_visualizations = True
            if args.show:
                plt.figure(action_fig.number)
                plt.show()
    
    if generate_scatter:
        scatter_fig = visualizer.plot_episode_length_vs_reward(save=not args.show)
        if scatter_fig:
            created_visualizations = True
            if args.show:
                plt.figure(scatter_fig.number)
                plt.show()
    
    # Save metrics to CSV
    if created_visualizations and not args.show:
        visualizer.save_metrics_to_csv()
        logging.info(f"Visualizations saved to {output_dir}")
    elif not created_visualizations:
        logging.warning("No visualizations were created")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 