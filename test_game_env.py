"""
Test script for the Cities: Skylines 2 game environment with GPU acceleration.
This will load the environment in mock mode to test GPU-accelerated operations.
"""

import sys
import os
import time
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import directly from module paths to avoid circular dependencies
try:
    from config.hardware_config import HardwareConfig
    from environment.game_env import CitiesEnvironment
    print("Successfully imported modules")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def test_game_env():
    """Test the game environment with GPU acceleration in mock mode."""
    print("\n=== Game Environment Test with GPU Acceleration ===")
    
    # Create hardware config with GPU acceleration
    config = HardwareConfig()
    print(f"GPU acceleration enabled: {config.use_cuda}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Device: {config.device}")
    
    # Initialize environment in mock mode
    print("\nInitializing environment in mock mode...")
    start_time = time.time()
    env = CitiesEnvironment(config=config, mock_mode=True)
    end_time = time.time()
    print(f"Environment initialization time: {(end_time - start_time)*1000:.2f} ms")
    
    # Reset environment
    print("\nResetting environment...")
    start_time = time.time()
    frame = env.reset()
    end_time = time.time()
    print(f"Environment reset time: {(end_time - start_time)*1000:.2f} ms")
    
    # Report frame information
    print(f"Frame shape: {frame.shape}")
    print(f"Frame device: {frame.device}")
    print(f"Frame dtype: {frame.dtype}")
    
    # Test a few steps
    print("\nTaking steps in the environment...")
    rewards = []
    step_times = []
    
    for i in range(10):
        # Take a random action
        action = np.random.randint(0, env.num_actions)
        
        # Measure step time
        start_time = time.time()
        next_frame, reward, done, info = env.step(action)
        end_time = time.time()
        
        step_time = (end_time - start_time)*1000
        step_times.append(step_time)
        rewards.append(reward)
        
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Step time={step_time:.2f} ms")
    
    # Report performance
    print(f"\nAverage step time: {np.mean(step_times):.2f} ms")
    print(f"Average reward: {np.mean(rewards):.4f}")
    
    # Test visual metrics estimator
    print("\nTesting visual metrics estimation...")
    start_time = time.time()
    population, metrics = env.visual_estimator.estimate_population(next_frame)
    end_time = time.time()
    
    print(f"Population estimation time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Estimated population: {population}")
    
    # Report GPU memory usage
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        print(f"Current: {torch.cuda.memory_allocated() / 1e9:.4f} GB")
        print(f"Maximum: {torch.cuda.max_memory_allocated() / 1e9:.4f} GB")
    
    # Clean up
    env.close()
    print("\nEnvironment closed. Test completed successfully.")

if __name__ == "__main__":
    test_game_env() 