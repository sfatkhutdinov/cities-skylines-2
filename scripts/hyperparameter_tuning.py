#!/usr/bin/env python3
"""
Script for hyperparameter tuning of the agent.

This script implements a simple grid search and random search for hyperparameters,
evaluating the performance on the Cities Skylines 2 environment.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple, Any
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import itertools

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.core import PPOAgent
from src.environment.core import Environment
from src.training.trainer import Trainer
from src.config.hardware_config import HardwareConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for the agent")
    parser.add_argument("--output", type=str, default="hyperparameter_results", help="Output directory for results")
    parser.add_argument("--method", type=str, choices=["grid", "random"], default="random", help="Search method")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials for random search")
    parser.add_argument("--mock", action="store_true", help="Use mock environment")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per trial")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per epoch")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    parser.add_argument("--eval_episodes", type=int, default=3, help="Evaluation episodes per trial")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel trials to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations of results")
    return parser.parse_args()


def get_grid_hyperparameters():
    """Get grid of hyperparameters to search.
    
    Returns:
        Dictionary of hyperparameter name to list of values to try
    """
    return {
        # PPO hyperparameters
        "lr": [1e-4, 3e-4, 1e-3],
        "gamma": [0.9, 0.95, 0.99],
        "lambda_gae": [0.9, 0.95],
        "clip_ratio": [0.1, 0.2, 0.3],
        "critic_coef": [0.5, 1.0],
        "entropy_coef": [0.0, 0.01, 0.05],
        
        # Network architecture
        "hidden_size": [64, 128, 256],
        "actor_layers": [1, 2],
        "critic_layers": [1, 2],
        
        # Training parameters
        "batch_size": [64, 128, 256],
        "update_epochs": [3, 5, 10],
    }


def get_random_hyperparameters(num_trials=10, seed=42):
    """Generate random hyperparameter combinations.
    
    Args:
        num_trials: Number of random trials to generate
        seed: Random seed
        
    Returns:
        List of dictionaries with hyperparameter combinations
    """
    random.seed(seed)
    np.random.seed(seed)
    
    trials = []
    
    # Define hyperparameter ranges
    param_ranges = {
        # PPO hyperparameters
        "lr": (1e-5, 1e-2, "log"),
        "gamma": (0.8, 0.999, "linear"),
        "lambda_gae": (0.8, 0.99, "linear"),
        "clip_ratio": (0.05, 0.4, "linear"),
        "critic_coef": (0.1, 2.0, "linear"),
        "entropy_coef": (0.0, 0.1, "linear"),
        
        # Network architecture
        "hidden_size": ([32, 64, 128, 256, 512], None, "choice"),
        "actor_layers": ([1, 2, 3], None, "choice"),
        "critic_layers": ([1, 2, 3], None, "choice"),
        
        # Training parameters
        "batch_size": ([32, 64, 128, 256, 512], None, "choice"),
        "update_epochs": ([1, 3, 5, 8, 10, 15], None, "choice"),
    }
    
    for _ in range(num_trials):
        trial_params = {}
        
        for param, (range_min, range_max, dist) in param_ranges.items():
            if dist == "log":
                trial_params[param] = np.exp(np.random.uniform(np.log(range_min), np.log(range_max)))
            elif dist == "linear":
                trial_params[param] = np.random.uniform(range_min, range_max)
            elif dist == "choice":
                trial_params[param] = random.choice(range_min)
                
            # Round numerical values for readability
            if param in ["lr", "gamma", "lambda_gae", "clip_ratio", "critic_coef", "entropy_coef"]:
                trial_params[param] = float(np.round(trial_params[param], 6))
        
        trials.append(trial_params)
    
    return trials


def evaluate_agent(agent, env, num_episodes=5, max_steps=100):
    """Evaluate agent performance.
    
    Args:
        agent: The agent to evaluate
        env: Environment to evaluate in
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        for _ in range(max_steps):
            action, _ = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
                
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
    }


def run_trial(trial_id, hyperparams, args):
    """Run a single trial with the given hyperparameters.
    
    Args:
        trial_id: ID of the trial
        hyperparams: Dictionary of hyperparameters to use
        args: Command line arguments
        
    Returns:
        Dictionary of results including hyperparameters and evaluation metrics
    """
    trial_start_time = time.time()
    
    # Set up logging for this trial
    log_file = os.path.join(args.output, f"trial_{trial_id}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure file handler for logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    
    # Get the root logger and add the file handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logging.info(f"Starting trial {trial_id}")
    logging.info(f"Hyperparameters: {hyperparams}")
    
    # Set up hardware configuration
    config = HardwareConfig()
    
    # Create environment
    env = Environment(config=config, mock_mode=args.mock)
    
    # Create agent with the trial hyperparameters
    agent = PPOAgent(
        state_dim=env.observation_shape,
        action_dim=env.action_space.n,
        hidden_size=hyperparams["hidden_size"],
        actor_layers=hyperparams["actor_layers"],
        critic_layers=hyperparams["critic_layers"],
        lr=hyperparams["lr"],
        gamma=hyperparams["gamma"],
        lambda_gae=hyperparams["lambda_gae"],
        clip_ratio=hyperparams["clip_ratio"],
        critic_coef=hyperparams["critic_coef"],
        entropy_coef=hyperparams["entropy_coef"],
        device=config.get_device()
    )
    
    # Create trainer
    trainer = Trainer(
        agent=agent,
        env=env,
        batch_size=hyperparams["batch_size"],
        update_epochs=hyperparams["update_epochs"],
        log_dir=os.path.join(args.output, f"trial_{trial_id}")
    )
    
    # Train the agent
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        trainer.train(num_episodes=args.episodes, max_steps=args.steps)
    
    training_time = time.time() - training_start_time
    
    # Evaluate the agent
    eval_metrics = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=args.eval_episodes,
        max_steps=args.steps
    )
    
    # Save agent checkpoint
    checkpoint_path = os.path.join(args.output, f"trial_{trial_id}", "checkpoint.pt")
    trainer.save_checkpoint(checkpoint_path)
    
    # Calculate total time
    total_time = time.time() - trial_start_time
    
    # Combine results
    results = {
        "trial_id": trial_id,
        "hyperparameters": hyperparams,
        "evaluation": eval_metrics,
        "training_time": training_time,
        "total_time": total_time,
    }
    
    # Save individual trial results
    results_path = os.path.join(args.output, f"trial_{trial_id}", "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Trial {trial_id} completed")
    logging.info(f"Evaluation metrics: {eval_metrics}")
    
    # Remove the file handler
    root_logger.removeHandler(file_handler)
    file_handler.close()
    
    return results


def run_grid_search(args):
    """Run grid search for hyperparameters.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of dictionaries with trial results
    """
    # Get hyperparameter grid
    param_grid = get_grid_hyperparameters()
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(itertools.product(*[param_grid[name] for name in param_names]))
    
    # Convert to list of dictionaries
    param_dicts = []
    for values in param_values:
        param_dict = {name: value for name, value in zip(param_names, values)}
        param_dicts.append(param_dict)
    
    logging.info(f"Grid search: {len(param_dicts)} combinations to evaluate")
    
    # Run trials
    all_results = []
    
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        future_to_params = {
            executor.submit(run_trial, i, params, args): (i, params)
            for i, params in enumerate(param_dicts)
        }
        
        for future in future_to_params:
            trial_id, params = future_to_params[future]
            try:
                result = future.result()
                all_results.append(result)
                logging.info(f"Completed trial {trial_id}: "
                            f"reward={result['evaluation']['mean_reward']:.2f}")
            except Exception as e:
                logging.error(f"Trial {trial_id} failed: {e}")
    
    return all_results


def run_random_search(args):
    """Run random search for hyperparameters.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of dictionaries with trial results
    """
    # Generate random hyperparameter combinations
    param_dicts = get_random_hyperparameters(num_trials=args.trials, seed=args.seed)
    
    logging.info(f"Random search: {len(param_dicts)} trials to evaluate")
    
    # Run trials
    all_results = []
    
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        future_to_params = {
            executor.submit(run_trial, i, params, args): (i, params)
            for i, params in enumerate(param_dicts)
        }
        
        for future in future_to_params:
            trial_id, params = future_to_params[future]
            try:
                result = future.result()
                all_results.append(result)
                logging.info(f"Completed trial {trial_id}: "
                            f"reward={result['evaluation']['mean_reward']:.2f}")
            except Exception as e:
                logging.error(f"Trial {trial_id} failed: {e}")
    
    return all_results


def analyze_results(results, output_dir):
    """Analyze and save hyperparameter tuning results.
    
    Args:
        results: List of dictionaries with trial results
        output_dir: Directory to save analysis results
    """
    # Extract data for analysis
    data = []
    
    for result in results:
        row = {
            "trial_id": result["trial_id"],
            "mean_reward": result["evaluation"]["mean_reward"],
            "std_reward": result["evaluation"]["std_reward"],
            "training_time": result["training_time"],
        }
        
        # Add hyperparameters
        for param, value in result["hyperparameters"].items():
            row[param] = value
        
        data.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by mean reward
    df_sorted = df.sort_values("mean_reward", ascending=False)
    
    # Save sorted results
    df_sorted.to_csv(os.path.join(output_dir, "sorted_results.csv"), index=False)
    
    # Get best hyperparameters
    best_trial = df_sorted.iloc[0]
    best_hyperparams = {col: best_trial[col] for col in df_sorted.columns 
                       if col not in ["trial_id", "mean_reward", "std_reward", "training_time"]}
    
    # Save best hyperparameters
    with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
        json.dump(best_hyperparams, f, indent=2)
    
    logging.info(f"Best trial: {best_trial['trial_id']}")
    logging.info(f"Best mean reward: {best_trial['mean_reward']:.4f}")
    logging.info(f"Best hyperparameters: {best_hyperparams}")
    
    return df_sorted, best_hyperparams


def visualize_results(df, output_dir):
    """Generate visualizations of hyperparameter tuning results.
    
    Args:
        df: DataFrame with sorted results
        output_dir: Directory to save visualizations
    """
    # Create a directory for visualizations
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. Distribution of rewards
    plt.figure(figsize=(10, 6))
    plt.hist(df["mean_reward"], bins=20, alpha=0.7, color="blue")
    plt.axvline(df["mean_reward"].max(), color="red", linestyle="--", 
                label=f"Best: {df['mean_reward'].max():.4f}")
    plt.xlabel("Mean Reward")
    plt.ylabel("Count")
    plt.title("Distribution of Mean Rewards")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(vis_dir, "reward_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Parameter importance: correlation with reward
    params = [col for col in df.columns 
              if col not in ["trial_id", "mean_reward", "std_reward", "training_time"]]
    
    correlations = {}
    for param in params:
        if df[param].nunique() > 1:  # Only calculate if parameter varies
            correlations[param] = df[param].corr(df["mean_reward"])
    
    # Sort by absolute correlation
    correlations = {k: v for k, v in sorted(correlations.items(), 
                                           key=lambda item: abs(item[1]), 
                                           reverse=True)}
    
    plt.figure(figsize=(12, 6))
    plt.bar(correlations.keys(), correlations.values())
    plt.xlabel("Hyperparameter")
    plt.ylabel("Correlation with Mean Reward")
    plt.title("Hyperparameter Importance")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "parameter_importance.png"), dpi=300)
    plt.close()
    
    # 3. Pairwise scatter plots for top parameters
    # Select top parameters by correlation
    top_params = list(correlations.keys())[:min(4, len(correlations))]
    
    # Create pairwise scatter plots
    for i, param1 in enumerate(top_params):
        for param2 in top_params[i+1:]:
            plt.figure(figsize=(8, 6))
            plt.scatter(df[param1], df[param2], c=df["mean_reward"], cmap="viridis", 
                       alpha=0.7, s=50)
            plt.colorbar(label="Mean Reward")
            plt.xlabel(param1)
            plt.ylabel(param2)
            plt.title(f"{param1} vs {param2}")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(vis_dir, f"{param1}_vs_{param2}.png"), dpi=300)
            plt.close()
    
    # 4. Learning curve for best trial
    best_trial_id = df.iloc[0]["trial_id"]
    best_log_dir = os.path.join(output_dir, f"trial_{best_trial_id}")
    
    try:
        # Load training log for best trial
        log_file = os.path.join(best_log_dir, "training_log.json")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                training_log = json.load(f)
            
            # Extract episode rewards
            episodes = list(range(1, len(training_log["episode_rewards"]) + 1))
            rewards = training_log["episode_rewards"]
            
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, rewards, marker="o")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Learning Curve for Best Trial")
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(vis_dir, "best_learning_curve.png"), dpi=300)
            plt.close()
    except Exception as e:
        logging.warning(f"Could not generate learning curve: {e}")
    
    # 5. Parameter parallel coordinates plot
    # Normalize parameters for parallel coordinates
    df_norm = df.copy()
    for param in params:
        if df[param].nunique() > 1:  # Only normalize if parameter varies
            df_norm[param] = (df[param] - df[param].min()) / (df[param].max() - df[param].min())
    
    # Sort by mean reward
    df_norm = df_norm.sort_values("mean_reward")
    
    # Select a subset of trials for clarity
    num_to_show = min(20, len(df_norm))
    step = len(df_norm) // num_to_show
    indices = list(range(0, len(df_norm), step))
    if len(df_norm) - 1 not in indices:
        indices.append(len(df_norm) - 1)  # Add best trial
    
    df_subset = df_norm.iloc[indices].copy()
    
    # Create parallel coordinates plot
    plt.figure(figsize=(14, 8))
    
    # Plot each parameter as a vertical axis
    x = list(range(len(params)))
    for i, (_, row) in enumerate(df_subset.iterrows()):
        color = plt.cm.viridis(i / len(df_subset))
        y = [row[param] for param in params]
        plt.plot(x, y, "-o", color=color, alpha=0.5, 
                label=f"Trial {row['trial_id']}" if i == len(df_subset)-1 else None)
    
    # Plot best trial with distinct color
    best_row = df_subset.iloc[-1]
    plt.plot(x, [best_row[param] for param in params], "-o", 
            color="red", linewidth=2, label=f"Best Trial {best_row['trial_id']}")
    
    plt.xticks(x, params, rotation=45, ha="right")
    plt.ylabel("Normalized Parameter Value")
    plt.title("Parallel Coordinates Plot of Hyperparameters")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "parallel_coordinates.png"), dpi=300)
    plt.close()


def main():
    """Main function for hyperparameter tuning."""
    args = parse_args()
    
    # Set up output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output, "tuning.log")),
            logging.StreamHandler()
        ]
    )
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    try:
        # Run search based on method
        if args.method == "grid":
            results = run_grid_search(args)
        else:  # random search
            results = run_random_search(args)
        
        # Save all results
        with open(os.path.join(args.output, "all_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        # Analyze results
        df_sorted, best_hyperparams = analyze_results(results, args.output)
        
        # Visualize results if requested
        if args.visualize:
            visualize_results(df_sorted, args.output)
        
        logging.info("Hyperparameter tuning completed successfully")
        return 0
    except Exception as e:
        logging.error(f"Hyperparameter tuning failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 