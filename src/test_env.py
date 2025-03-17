"""
Test script for Cities: Skylines 2 environment.
"""

from environment.game_env import CitiesEnvironment
from agent.ppo_agent import PPOAgent
from config.hardware_config import HardwareConfig
from environment.input_simulator import InputSimulator

def main():
    # Initialize hardware config
    config = HardwareConfig()
    
    # Initialize environment
    env = CitiesEnvironment(config)
    
    # Initialize agent
    agent = PPOAgent(config)
    
    # Test input simulator
    input_sim = InputSimulator()
    if not input_sim.ensure_game_window_focused():
        print("Could not focus Cities: Skylines II window")
        return
        
    print("Game window found and focused successfully")
    
    # Test basic environment functionality
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(5):
        action, _, _ = agent.select_action(obs)
        obs, reward, done, info = env.step(action.item())
        print(f"Step {i}: Reward = {reward}, Done = {done}")
        
    env.close()
    
if __name__ == "__main__":
    main() 