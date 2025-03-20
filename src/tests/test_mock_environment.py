"""
Test script for the MockEnvironment class.

This script demonstrates how to use the mock environment for testing
the reinforcement learning agent without requiring the actual game.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.environment.mock_environment import MockEnvironment
from src.config.hardware_config import HardwareConfig

def test_basic_functionality():
    """Test basic functionality of the mock environment."""
    # Initialize with default config
    env = MockEnvironment()
    
    # Test reset
    observation = env.reset()
    assert isinstance(observation, torch.Tensor), "Observation should be a tensor"
    assert observation.shape == (3, 240, 320), "Observation has wrong shape"
    
    # Test step
    action = 0  # Take first action
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, torch.Tensor), "Observation should be a tensor"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(done, bool), "Done should be a boolean"
    assert isinstance(info, dict), "Info should be a dictionary"
    
    # Test render
    frame = env.render()
    assert frame.shape == (240, 320, 3), "Rendered frame has wrong shape"
    
    # Check menu state
    menu_state = env.check_menu_state()
    assert isinstance(menu_state, bool), "Menu state should be a boolean"
    
    # Check game running
    game_running = env.check_game_running()
    assert isinstance(game_running, bool), "Game running should be a boolean"
    
    # Close environment
    env.close()
    print("Basic functionality tests passed!")

def test_episode():
    """Test running a complete episode."""
    max_steps = 100
    env = MockEnvironment(max_steps=max_steps)
    
    observation = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    # Run until episode completion
    while not done:
        # Take random action
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Print progress every 10 steps
        if steps % 10 == 0:
            print(f"Step {steps}, Reward: {reward:.2f}, Total: {total_reward:.2f}")
            print(f"City state: Pop={info['population']}, Budget={info['budget']}")
            
        # Force early termination if taking too long
        if steps >= 120:
            print("Episode taking too long, terminating...")
            break
    
    print(f"Episode completed after {steps} steps with total reward {total_reward:.2f}")
    print("Episode test passed!")

def test_error_conditions():
    """Test error conditions (crash, freeze, menus)."""
    # Create environment with high probabilities for testing errors
    env = MockEnvironment(
        max_steps=50,
        crash_probability=0.1,
        freeze_probability=0.2,
        menu_probability=0.2
    )
    
    observation = env.reset()
    
    # Run for a few steps and monitor issues
    crashes = 0
    freezes = 0
    menus = 0
    
    for i in range(100):
        # Take random action
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        
        # Check for different conditions
        if done and 'error' in info and info['error'] == 'crash':
            crashes += 1
            observation = env.reset()  # Restart after crash
            print(f"Detected crash at step {i}, restarting...")
            
        elif 'frozen' in info and info['frozen']:
            freezes += 1
            print(f"Detected freeze at step {i}, duration: {info['freeze_duration']}")
            
        elif info['in_menu']:
            menus += 1
            print(f"Detected menu at step {i}")
        
        if done:
            observation = env.reset()
    
    print(f"Error condition test completed with {crashes} crashes, {freezes} freezes, and {menus} menus")
    print("Error condition test passed!")

def test_visualization():
    """Test visualization of the environment."""
    env = MockEnvironment()
    observation = env.reset()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # Collect frames for different actions
    frames = []
    for action in [0, 5, 10, 15]:
        observation, _, _, _ = env.step(action)
        frames.append(env.render())
    
    # Generate a menu frame
    env.in_menu = True
    frames.append(env._generate_menu_frame().permute(1, 2, 0).cpu().numpy())
    
    # Generate a crash frame (black screen)
    black_screen = np.zeros((240, 320, 3), dtype=np.uint8)
    frames.append(black_screen)
    
    # Plot frames
    titles = [
        "Road Action", "Residential Action", 
        "Commercial Action", "Services Action",
        "Menu Screen", "Crash Screen"
    ]
    
    for i, (frame, title) in enumerate(zip(frames, titles)):
        axes[i].imshow(frame)
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "mock_environment_visualization.png")
    
    plt.close()
    print(f"Visualization saved to {output_dir / 'mock_environment_visualization.png'}")
    print("Visualization test passed!")

def run_all_tests():
    """Run all tests."""
    print("Testing MockEnvironment...")
    test_basic_functionality()
    test_episode()
    test_error_conditions()
    test_visualization()
    print("All tests passed!")

if __name__ == "__main__":
    run_all_tests() 