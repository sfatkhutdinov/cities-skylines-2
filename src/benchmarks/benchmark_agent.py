"""
Benchmark script for evaluating agent performance.

This script uses the mock environment to benchmark agent performance
across different configurations and scenarios.
"""

import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.environment.mock_environment import MockEnvironment
from src.agent.ppo_agent import PPOAgent
from src.config.hardware_config import HardwareConfig
from src.config.training_config import TrainingConfig
from src.monitoring.hardware_monitor import HardwareMonitor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark agent performance")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output directory name")
    parser.add_argument("--gpu", action="store_true", help="Force use of GPU if available")
    parser.add_argument("--cpu", action="store_true", help="Force use of CPU")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    return parser.parse_args()

def benchmark_agent(
    agent: PPOAgent,
    env: MockEnvironment,
    num_episodes: int,
    max_steps: int,
    hardware_monitor: Optional[HardwareMonitor] = None
) -> Dict[str, Any]:
    """Benchmark agent performance.
    
    Args:
        agent: The agent to benchmark
        env: The environment to use
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        hardware_monitor: Optional hardware monitor
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "rewards": [],
        "episode_lengths": [],
        "success_rate": 0.0,
        "crashes": 0,
        "freezes": 0,
        "menu_encounters": 0,
        "time_per_step": [],
        "hardware_metrics": [],
        "final_city_states": []
    }
    
    total_steps = 0
    successful_episodes = 0
    
    # Start hardware monitoring if available
    if hardware_monitor:
        hardware_monitor.start()
    
    for episode in range(num_episodes):
        print(f"Starting episode {episode+1}/{num_episodes}")
        observation = env.reset()
        
        episode_reward = 0
        done = False
        steps = 0
        episode_crashes = 0
        episode_freezes = 0
        episode_menus = 0
        
        while not done and steps < max_steps:
            step_start_time = time.time()
            
            # Get action from agent
            action = agent.select_action(observation)
            
            # Take step in environment
            next_observation, reward, done, info = env.step(action)
            
            # Record hardware metrics if monitoring
            if hardware_monitor:
                metrics = hardware_monitor.get_metrics()
                results["hardware_metrics"].append(metrics)
            
            # Track step time
            step_time = time.time() - step_start_time
            results["time_per_step"].append(step_time)
            
            # Track error conditions
            if done and 'error' in info and info['error'] == 'crash':
                episode_crashes += 1
                results["crashes"] += 1
                observation = env.reset()
                done = False  # Continue episode after reset
            
            elif 'frozen' in info and info['frozen']:
                episode_freezes += 1
                results["freezes"] += 1
            
            elif info.get('in_menu', False):
                episode_menus += 1
                results["menu_encounters"] += 1
            
            # Update state
            observation = next_observation
            episode_reward += reward
            steps += 1
            total_steps += 1
            
            # Print progress periodically
            if steps % 50 == 0:
                print(f"  Step {steps}, Reward: {episode_reward:.2f}")
                print(f"  City state: Pop={info['population']}, Budget={info['budget']}")
        
        # Record episode results
        results["rewards"].append(episode_reward)
        results["episode_lengths"].append(steps)
        
        # Record final city state
        results["final_city_states"].append({
            "population": info.get('population', 0),
            "budget": info.get('budget', 0),
            "happiness": info.get('happiness', 0),
            "traffic": info.get('traffic', 0),
            "pollution": info.get('pollution', 0)
        })
        
        # Check if episode was successful (didn't terminate early due to errors)
        if steps >= max_steps or (done and 'error' not in info):
            successful_episodes += 1
        
        print(f"Episode {episode+1} completed:")
        print(f"  Steps: {steps}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Crashes: {episode_crashes}")
        print(f"  Freezes: {episode_freezes}")
        print(f"  Menu encounters: {episode_menus}")
    
    # Stop hardware monitoring
    if hardware_monitor:
        hardware_monitor.stop()
    
    # Calculate success rate
    results["success_rate"] = successful_episodes / num_episodes * 100
    
    # Calculate averages
    results["avg_reward"] = np.mean(results["rewards"])
    results["avg_episode_length"] = np.mean(results["episode_lengths"])
    results["avg_time_per_step"] = np.mean(results["time_per_step"])
    
    return results

def plot_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Plot benchmark results.
    
    Args:
        results: Benchmark results
        output_dir: Output directory
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(results["rewards"])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig(output_dir / "rewards.png")
    plt.close()
    
    # Plot episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(results["episode_lengths"])
    plt.title("Episode Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.savefig(output_dir / "episode_lengths.png")
    plt.close()
    
    # Plot step times
    plt.figure(figsize=(10, 6))
    plt.plot(results["time_per_step"])
    plt.title("Step Processing Times")
    plt.xlabel("Step")
    plt.ylabel("Time (s)")
    plt.savefig(output_dir / "step_times.png")
    plt.close()
    
    # Plot city metrics
    if results["final_city_states"]:
        metrics = ["population", "budget", "happiness", "traffic", "pollution"]
        plt.figure(figsize=(12, 8))
        
        for metric in metrics:
            values = [state.get(metric, 0) for state in results["final_city_states"]]
            
            # Normalize values for comparison
            if metric in ["population", "budget"]:
                if max(values) > 0:
                    values = [v / max(values) for v in values]
            
            plt.plot(values, label=metric.capitalize())
        
        plt.title("City Metrics Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Normalized Value")
        plt.legend()
        plt.savefig(output_dir / "city_metrics.png")
        plt.close()
    
    # Plot hardware metrics if available
    if results["hardware_metrics"]:
        metrics = ["cpu_percent", "memory_percent", "gpu_utilization", "gpu_memory_used"]
        
        plt.figure(figsize=(12, 8))
        
        for metric in metrics:
            values = [hw.get(metric, 0) for hw in results["hardware_metrics"]]
            if any(values):  # Only plot if we have values
                plt.plot(values, label=metric.replace('_', ' ').title())
        
        plt.title("Hardware Utilization")
        plt.xlabel("Step")
        plt.ylabel("Percent / MB")
        plt.legend()
        plt.savefig(output_dir / "hardware_metrics.png")
        plt.close()

def save_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save benchmark results to disk.
    
    Args:
        results: Benchmark results
        output_dir: Output directory
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create a simplified version of the results for JSON serialization
    serializable_results = {
        "timestamp": datetime.now().isoformat(),
        "avg_reward": float(results["avg_reward"]),
        "avg_episode_length": float(results["avg_episode_length"]),
        "avg_time_per_step": float(results["avg_time_per_step"]),
        "success_rate": float(results["success_rate"]),
        "crashes": results["crashes"],
        "freezes": results["freezes"],
        "menu_encounters": results["menu_encounters"],
        "rewards": [float(r) for r in results["rewards"]],
        "episode_lengths": [int(l) for l in results["episode_lengths"]],
        "time_per_step": [float(t) for t in results["time_per_step"][:100]],  # Save first 100 for brevity
        "final_city_states": results["final_city_states"],
        # Filter out complex hardware metrics for JSON serialization
        "hardware_metrics_summary": {
            "cpu_percent_avg": np.mean([m.get("cpu_percent", 0) for m in results["hardware_metrics"]]) if results["hardware_metrics"] else 0,
            "memory_percent_avg": np.mean([m.get("memory_percent", 0) for m in results["hardware_metrics"]]) if results["hardware_metrics"] else 0,
            "gpu_utilization_avg": np.mean([m.get("gpu_utilization", 0) for m in results["hardware_metrics"]]) if results["hardware_metrics"] else 0,
            "gpu_memory_used_avg": np.mean([m.get("gpu_memory_used", 0) for m in results["hardware_metrics"]]) if results["hardware_metrics"] else 0
        }
    }
    
    # Save as JSON
    with open(output_dir / "results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    # Save a summary as text
    with open(output_dir / "summary.txt", "w") as f:
        f.write("Benchmark Results Summary\n")
        f.write("========================\n\n")
        f.write(f"Timestamp: {serializable_results['timestamp']}\n\n")
        f.write(f"Average Reward: {serializable_results['avg_reward']:.2f}\n")
        f.write(f"Average Episode Length: {serializable_results['avg_episode_length']:.2f} steps\n")
        f.write(f"Average Time per Step: {serializable_results['avg_time_per_step'] * 1000:.2f} ms\n")
        f.write(f"Success Rate: {serializable_results['success_rate']:.2f}%\n\n")
        f.write(f"Total Crashes: {serializable_results['crashes']}\n")
        f.write(f"Total Freezes: {serializable_results['freezes']}\n")
        f.write(f"Total Menu Encounters: {serializable_results['menu_encounters']}\n\n")
        
        f.write("Hardware Utilization:\n")
        f.write(f"  CPU: {serializable_results['hardware_metrics_summary']['cpu_percent_avg']:.2f}%\n")
        f.write(f"  Memory: {serializable_results['hardware_metrics_summary']['memory_percent_avg']:.2f}%\n")
        f.write(f"  GPU: {serializable_results['hardware_metrics_summary']['gpu_utilization_avg']:.2f}%\n")
        f.write(f"  GPU Memory: {serializable_results['hardware_metrics_summary']['gpu_memory_used_avg']:.2f} MB\n")

def main():
    """Main function to run the benchmark."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config_data = json.load(f)
        hardware_config = HardwareConfig(**config_data.get("hardware", {}))
        training_config = TrainingConfig(**config_data.get("training", {}))
    else:
        # Use default configuration with command line overrides
        hardware_config = HardwareConfig(
            use_gpu=args.gpu,
            use_cpu=args.cpu,
            use_mixed_precision=args.mixed_precision
        )
        training_config = TrainingConfig()
    
    # Initialize hardware monitor
    hardware_monitor = HardwareMonitor(hardware_config)
    
    # Initialize environment
    env = MockEnvironment(
        config=hardware_config,
        max_steps=args.steps,
        # Reduce error probabilities for benchmark
        crash_probability=0.01,
        freeze_probability=0.02,
        menu_probability=0.05
    )
    
    # Initialize agent
    agent = PPOAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        hardware_config=hardware_config,
        training_config=training_config
    )
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "output" / args.output / timestamp
    
    print("Starting benchmark with the following configuration:")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps per episode: {args.steps}")
    print(f"  Device: {hardware_config.get_device()}")
    print(f"  Mixed precision: {hardware_config.use_mixed_precision}")
    print(f"  Output directory: {output_dir}")
    
    # Run benchmark
    results = benchmark_agent(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        max_steps=args.steps,
        hardware_monitor=hardware_monitor
    )
    
    # Plot and save results
    plot_results(results, output_dir)
    save_results(results, output_dir)
    
    print("\nBenchmark completed:")
    print(f"  Average reward: {results['avg_reward']:.2f}")
    print(f"  Success rate: {results['success_rate']:.2f}%")
    print(f"  Average time per step: {results['avg_time_per_step'] * 1000:.2f} ms")
    print(f"  Results saved to {output_dir}")

if __name__ == "__main__":
    main() 