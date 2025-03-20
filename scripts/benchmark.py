#!/usr/bin/env python3
"""
Script to benchmark the agent and environment performance.

This script measures the performance of the agent and environment in terms of
frames per second, memory usage, and other metrics.
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import psutil
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.core import PPOAgent
from src.environment.core import Environment
from src.config.hardware_config import HardwareConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark agent and environment performance")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to load")
    parser.add_argument("--mock", action="store_true", help="Use mock environment")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization of results")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU is available")
    return parser.parse_args()


def measure_fps(agent, env, num_steps: int = 100) -> float:
    """Measure frames per second for environment and agent interaction.
    
    Args:
        agent: The agent
        env: The environment
        num_steps: Number of steps to measure
        
    Returns:
        Frames per second
    """
    obs = env.reset()
    
    # Warm up
    for _ in range(10):
        action, _ = agent.select_action(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    # Measure
    start_time = time.time()
    
    for _ in range(num_steps):
        action, _ = agent.select_action(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    fps = num_steps / elapsed_time
    return fps


def measure_memory_usage(agent, env, num_steps: int = 100) -> Dict[str, float]:
    """Measure memory usage during agent-environment interaction.
    
    Args:
        agent: The agent
        env: The environment
        num_steps: Number of steps to measure
        
    Returns:
        Dictionary of memory usage metrics
    """
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Run agent-environment interaction
    obs = env.reset()
    for _ in range(num_steps):
        action, _ = agent.select_action(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    # Get final memory usage
    final_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Get peak memory usage
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        peak_memory_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    else:
        peak_memory_gpu = 0
    
    return {
        "initial_memory_mb": initial_memory,
        "final_memory_mb": final_memory,
        "memory_increase_mb": final_memory - initial_memory,
        "peak_memory_gpu_mb": peak_memory_gpu
    }


def measure_inference_time(agent, env, num_steps: int = 100) -> Dict[str, float]:
    """Measure inference time for agent and environment components.
    
    Args:
        agent: The agent
        env: The environment
        num_steps: Number of steps to measure
        
    Returns:
        Dictionary of inference time metrics
    """
    obs = env.reset()
    
    # Warm up
    for _ in range(10):
        action, _ = agent.select_action(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    # Measure agent inference time
    agent_times = []
    for _ in range(num_steps):
        start_time = time.time()
        action, _ = agent.select_action(obs)
        agent_times.append(time.time() - start_time)
        
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    # Measure environment step time
    env_times = []
    obs = env.reset()
    for _ in range(num_steps):
        action, _ = agent.select_action(obs)
        
        start_time = time.time()
        obs, _, done, _ = env.step(action)
        env_times.append(time.time() - start_time)
        
        if done:
            obs = env.reset()
    
    return {
        "agent_inference_ms": np.mean(agent_times) * 1000,
        "agent_inference_std_ms": np.std(agent_times) * 1000,
        "env_step_ms": np.mean(env_times) * 1000,
        "env_step_std_ms": np.std(env_times) * 1000,
        "total_inference_ms": np.mean(agent_times) * 1000 + np.mean(env_times) * 1000
    }


def run_benchmark(args):
    """Run the benchmark with the specified arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of benchmark results
    """
    logging.info("Setting up environment and agent")
    
    # Configure hardware
    config = HardwareConfig()
    if args.cpu:
        config.device = "cpu"
    
    # Create environment
    env = Environment(config=config, mock_mode=args.mock)
    
    # Create or load agent
    if args.checkpoint:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=config.get_device())
            agent_state = checkpoint.get("agent_state", None)
            
            if agent_state:
                # Create agent with same architecture
                agent = PPOAgent(
                    state_dim=env.observation_shape,
                    action_dim=env.action_space.n,
                    device=config.get_device()
                )
                
                # Load state
                agent.load_state_dict(agent_state)
                logging.info(f"Loaded agent from {args.checkpoint}")
            else:
                raise ValueError("Checkpoint does not contain agent state")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            logging.info("Creating new agent")
            agent = PPOAgent(
                state_dim=env.observation_shape,
                action_dim=env.action_space.n,
                device=config.get_device()
            )
    else:
        # Create new agent
        agent = PPOAgent(
            state_dim=env.observation_shape,
            action_dim=env.action_space.n,
            device=config.get_device()
        )
    
    # Run benchmarks
    results = {}
    
    # Measure FPS
    logging.info("Measuring FPS")
    fps = measure_fps(agent, env, num_steps=args.max_steps)
    results["fps"] = fps
    logging.info(f"FPS: {fps:.2f}")
    
    # Measure memory usage
    logging.info("Measuring memory usage")
    memory_metrics = measure_memory_usage(agent, env, num_steps=args.max_steps)
    results.update(memory_metrics)
    logging.info(f"Memory increase: {memory_metrics['memory_increase_mb']:.2f} MB")
    
    # Measure inference time
    logging.info("Measuring inference time")
    inference_metrics = measure_inference_time(agent, env, num_steps=args.max_steps)
    results.update(inference_metrics)
    logging.info(f"Agent inference time: {inference_metrics['agent_inference_ms']:.2f} ms")
    logging.info(f"Environment step time: {inference_metrics['env_step_ms']:.2f} ms")
    
    # Run episodes
    logging.info(f"Running {args.num_episodes} episodes with {args.max_steps} steps each")
    
    episode_lengths = []
    episode_rewards = []
    episode_times = []
    action_counts = {}
    
    for episode in range(args.num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        start_time = time.time()
        
        for step in range(args.max_steps):
            action, _ = agent.select_action(obs)
            
            # Update action counts
            action_counts[action] = action_counts.get(action, 0) + 1
            
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        end_time = time.time()
        episode_time = end_time - start_time
        
        episode_lengths.append(episode_length)
        episode_rewards.append(episode_reward)
        episode_times.append(episode_time)
        
        logging.info(f"Episode {episode+1}/{args.num_episodes}: "
                    f"Reward={episode_reward:.2f}, "
                    f"Length={episode_length}, "
                    f"Time={episode_time:.2f}s")
    
    # Add episode metrics to results
    results["episode_lengths"] = episode_lengths
    results["episode_rewards"] = episode_rewards
    results["episode_times"] = episode_times
    results["mean_episode_length"] = np.mean(episode_lengths)
    results["mean_episode_reward"] = np.mean(episode_rewards)
    results["mean_episode_time"] = np.mean(episode_times)
    results["action_counts"] = action_counts
    
    # Add system info
    results["system_info"] = {
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "device": config.device,
        "has_cuda": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    
    # Add benchmark parameters
    results["benchmark_params"] = {
        "num_episodes": args.num_episodes,
        "max_steps": args.max_steps,
        "mock_mode": args.mock,
        "checkpoint": args.checkpoint
    }
    
    return results


def visualize_results(results, output_path="benchmark_visualization.png"):
    """Visualize benchmark results.
    
    Args:
        results: Benchmark results
        output_path: Path to save visualization
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("Benchmark Results", fontsize=16)
    
    # Create subplot grid
    gs = fig.add_gridspec(3, 3)
    
    # 1. FPS and timing metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(["FPS"], [results["fps"]], color="blue")
    ax1.set_ylabel("Frames per second")
    ax1.set_title("Performance")
    
    # 2. Inference time breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(["Agent", "Environment"], 
            [results["agent_inference_ms"], results["env_step_ms"]], 
            color=["green", "orange"])
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Inference Time")
    
    # 3. Memory usage
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(["Initial", "Final", "Increase", "GPU Peak"], 
            [results["initial_memory_mb"], results["final_memory_mb"], 
             results["memory_increase_mb"], results["peak_memory_gpu_mb"]], 
            color=["blue", "red", "purple", "orange"])
    ax3.set_ylabel("Memory (MB)")
    ax3.set_title("Memory Usage")
    
    # 4. Episode rewards
    ax4 = fig.add_subplot(gs[1, :2])
    episodes = range(1, len(results["episode_rewards"]) + 1)
    ax4.plot(episodes, results["episode_rewards"], marker="o")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Reward")
    ax4.set_title("Episode Rewards")
    ax4.grid(True, alpha=0.3)
    
    # 5. Action distribution
    ax5 = fig.add_subplot(gs[1, 2])
    actions = list(results["action_counts"].keys())
    counts = list(results["action_counts"].values())
    ax5.bar(actions, counts)
    ax5.set_xlabel("Action")
    ax5.set_ylabel("Count")
    ax5.set_title("Action Distribution")
    
    # 6. System info
    ax6 = fig.add_subplot(gs[2, :])
    sys_info = results["system_info"]
    benchmark_params = results["benchmark_params"]
    system_text = (
        f"System Info:\n"
        f"CPU Count: {sys_info['cpu_count']}\n"
        f"Memory: {sys_info['memory_total_gb']:.2f} GB\n"
        f"Device: {sys_info['device']}\n"
        f"CUDA Available: {sys_info['has_cuda']}\n"
        f"CUDA Device: {sys_info['cuda_device']}\n\n"
        f"Benchmark Parameters:\n"
        f"Episodes: {benchmark_params['num_episodes']}\n"
        f"Max Steps: {benchmark_params['max_steps']}\n"
        f"Mock Mode: {benchmark_params['mock_mode']}\n"
        f"Checkpoint: {benchmark_params['checkpoint']}"
    )
    ax6.text(0.5, 0.5, system_text, ha="center", va="center", fontsize=10)
    ax6.axis("off")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    logging.info(f"Visualization saved to {output_path}")


def main():
    """Main function to run benchmark."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Run benchmark
        results = run_benchmark(args)
        
        # Save results
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logging.info(f"Benchmark results saved to {args.output}")
        
        # Visualize results if requested
        if args.visualize:
            visualize_results(results, output_path=os.path.splitext(args.output)[0] + ".png")
        
        return 0
    except Exception as e:
        logging.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 