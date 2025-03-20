"""
Mock environment for Cities: Skylines 2 agent.

This module provides a simulated environment for testing the agent without
requiring the actual game to be running.
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, Tuple, List, Any, Optional
import random
import gymnasium as gym
from gymnasium import spaces

from src.config.hardware_config import HardwareConfig

logger = logging.getLogger(__name__)

class MockEnvironment:
    """Mock environment that simulates a simplified version of Cities: Skylines 2."""
    
    def __init__(self, config: Optional[HardwareConfig] = None, **kwargs):
        """Initialize the mock environment.
        
        Args:
            config: Hardware configuration
            **kwargs: Additional configuration options
        """
        self.config = config or HardwareConfig()
        self.device = self.config.get_device()
        self.dtype = self.config.get_dtype()
        
        # Configuration
        self.frame_height = kwargs.get('frame_height', 240)
        self.frame_width = kwargs.get('frame_width', 320)
        self.frame_channels = kwargs.get('frame_channels', 3)
        self.num_actions = kwargs.get('num_actions', 20)
        self.max_steps = kwargs.get('max_steps', 1000)
        
        # State variables
        self.steps = 0
        self.last_action = None
        self.current_frame = None
        self.in_menu = False
        self.menu_probability = 0.02  # Probability of entering a menu
        self.crash_probability = 0.005  # Probability of simulated crash
        self.freeze_probability = 0.01  # Probability of simulated freeze
        self.is_frozen = False
        self.freeze_duration = 0
        
        # City state variables (simplified simulation)
        self.population = 1000
        self.budget = 10000
        self.happiness = 0.5
        self.traffic = 0.3
        self.pollution = 0.2
        
        # Set up observation and action spaces
        self.observation_space = spaces.Box(
            low=0, 
            high=255, 
            shape=(self.frame_channels, self.frame_height, self.frame_width), 
            dtype=np.uint8
        )
        self.action_space = spaces.Discrete(self.num_actions)
        
        # Generate initial city layout
        self.city_grid = np.zeros((50, 50), dtype=np.int32)
        self._generate_random_city()
        
        logger.info("Initialized mock environment")
    
    def reset(self) -> torch.Tensor:
        """Reset the environment.
        
        Returns:
            torch.Tensor: Initial observation
        """
        logger.debug("Resetting mock environment")
        self.steps = 0
        self.last_action = None
        self.in_menu = False
        self.is_frozen = False
        self.freeze_duration = 0
        
        # Reset city state
        self.population = 1000
        self.budget = 10000
        self.happiness = 0.5
        self.traffic = 0.3
        self.pollution = 0.2
        
        # Generate new city layout
        self._generate_random_city()
        
        # Generate initial frame
        self.current_frame = self._generate_frame()
        
        return self.current_frame
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Execute an action in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple containing (observation, reward, done, info)
        """
        self.steps += 1
        self.last_action = action
        
        # Check for simulated crash
        if random.random() < self.crash_probability:
            logger.info("Simulating game crash")
            # Return black screen and negative reward
            black_screen = torch.zeros(
                (self.frame_channels, self.frame_height, self.frame_width),
                dtype=self.dtype, device=self.device
            )
            return black_screen, -5.0, True, {'error': 'crash', 'steps': self.steps}
        
        # Check for simulated freeze
        if random.random() < self.freeze_probability and not self.is_frozen:
            logger.info("Simulating game freeze")
            self.is_frozen = True
            self.freeze_duration = random.randint(3, 10)  # Freeze for 3-10 steps
        
        # Handle frozen state
        if self.is_frozen:
            self.freeze_duration -= 1
            if self.freeze_duration <= 0:
                logger.info("Game unfrozen")
                self.is_frozen = False
            else:
                # Return same frame during freeze
                return self.current_frame, -0.1, False, {
                    'steps': self.steps, 
                    'frozen': True,
                    'freeze_duration': self.freeze_duration
                }
        
        # Determine if menu appears
        if random.random() < self.menu_probability and not self.in_menu:
            self.in_menu = True
            logger.debug("Entering menu")
        
        # Handle menu interaction
        if self.in_menu:
            # Allow exit from menu with certain actions
            if action in [0, 1, 2, 3]:  # Basic navigation actions
                self.in_menu = random.random() < 0.3  # 30% chance to exit menu
                reward = -0.1
            else:
                reward = -0.2  # Penalty for wrong menu action
            
            # Generate menu observation
            observation = self._generate_menu_frame()
        else:
            # Update city state based on action
            reward = self._update_city_state(action)
            
            # Generate game observation
            observation = self._generate_frame()
        
        # Update current frame
        self.current_frame = observation
        
        # Check if episode is done
        done = self.steps >= self.max_steps or self.budget <= 0
        
        # Create info dictionary
        info = {
            'steps': self.steps,
            'in_menu': self.in_menu,
            'population': self.population,
            'budget': self.budget,
            'happiness': self.happiness,
            'traffic': self.traffic,
            'pollution': self.pollution
        }
        
        return observation, reward, done, info
    
    def render(self, mode='rgb_array'):
        """Render the current state.
        
        Args:
            mode: Rendering mode
            
        Returns:
            RGB array of the current frame
        """
        if self.current_frame is None:
            return None
            
        # Convert tensor to numpy array for visualization
        if isinstance(self.current_frame, torch.Tensor):
            frame = self.current_frame.cpu().numpy()
            frame = np.transpose(frame, (1, 2, 0))  # CHW to HWC
            return frame
        return self.current_frame
    
    def close(self):
        """Clean up resources."""
        logger.debug("Closing mock environment")
        # No resources to clean up in the mock environment
    
    def check_menu_state(self) -> bool:
        """Check if currently in a menu.
        
        Returns:
            bool: True if in menu, False otherwise
        """
        return self.in_menu
    
    def check_game_running(self) -> bool:
        """Check if the game is running.
        
        Returns:
            bool: True if game is running, False if crashed
        """
        # Mock environment is always running unless explicitly crashed
        return not (random.random() < self.crash_probability)
    
    def restart_game(self) -> bool:
        """Simulate restarting the game.
        
        Returns:
            bool: True if restart successful
        """
        logger.info("Simulating game restart")
        time.sleep(1)  # Simulate restart time
        self.reset()
        return True
    
    def _generate_random_city(self):
        """Generate a random city layout."""
        # Clear grid
        self.city_grid.fill(0)
        
        # Add random roads
        for _ in range(10):
            start_x = random.randint(0, 49)
            start_y = random.randint(0, 49)
            length = random.randint(5, 20)
            direction = random.choice(['h', 'v'])
            
            if direction == 'h':
                for i in range(min(length, 50 - start_x)):
                    self.city_grid[start_y, start_x + i] = 1  # Road
            else:
                for i in range(min(length, 50 - start_y)):
                    self.city_grid[start_y + i, start_x] = 1  # Road
        
        # Add random buildings
        for _ in range(30):
            x = random.randint(0, 49)
            y = random.randint(0, 49)
            building_type = random.randint(2, 5)  # Different building types
            size = random.randint(1, 3)
            
            for i in range(min(size, 50 - y)):
                for j in range(min(size, 50 - x)):
                    if self.city_grid[y + i, x + j] == 0:  # Only place on empty space
                        self.city_grid[y + i, x + j] = building_type
    
    def _generate_frame(self) -> torch.Tensor:
        """Generate a game frame based on current state.
        
        Returns:
            torch.Tensor: Generated observation
        """
        # Create base frame
        frame = np.zeros((self.frame_height, self.frame_width, self.frame_channels), dtype=np.uint8)
        
        # Simplified rendering of city layout
        grid_height = min(self.frame_height, self.city_grid.shape[0])
        grid_width = min(self.frame_width, self.city_grid.shape[1])
        
        for y in range(grid_height):
            for x in range(grid_width):
                if self.city_grid[y, x] == 0:  # Empty
                    frame[y, x] = [100, 200, 100]  # Green
                elif self.city_grid[y, x] == 1:  # Road
                    frame[y, x] = [100, 100, 100]  # Gray
                elif self.city_grid[y, x] == 2:  # Residential
                    frame[y, x] = [50, 150, 250]  # Blue
                elif self.city_grid[y, x] == 3:  # Commercial
                    frame[y, x] = [250, 200, 50]  # Yellow
                elif self.city_grid[y, x] == 4:  # Industrial
                    frame[y, x] = [250, 100, 50]  # Red
                else:  # Special buildings
                    frame[y, x] = [200, 50, 200]  # Purple
        
        # Add some noise to make frames unique
        noise = np.random.randint(0, 10, (self.frame_height, self.frame_width, self.frame_channels), dtype=np.uint8)
        frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
        
        # Add simple UI elements
        # Population counter at top
        text_y = 5
        text_x = 10
        text_width = 60
        text_height = 15
        frame[text_y:text_y+text_height, text_x:text_x+text_width] = [200, 200, 200]  # Gray background
        
        # Budget counter
        text_y = 25
        frame[text_y:text_y+text_height, text_x:text_x+text_width] = [200, 200, 200]
        
        # Convert to tensor and CHW format
        tensor_frame = torch.tensor(frame, dtype=self.dtype, device=self.device)
        tensor_frame = tensor_frame.permute(2, 0, 1)  # HWC to CHW
        
        return tensor_frame
    
    def _generate_menu_frame(self) -> torch.Tensor:
        """Generate a menu frame.
        
        Returns:
            torch.Tensor: Menu observation
        """
        # Create dark overlay base
        frame = np.ones((self.frame_height, self.frame_width, self.frame_channels), dtype=np.uint8) * 50
        
        # Draw menu box
        menu_y = self.frame_height // 4
        menu_height = self.frame_height // 2
        menu_x = self.frame_width // 4
        menu_width = self.frame_width // 2
        
        frame[menu_y:menu_y+menu_height, menu_x:menu_x+menu_width] = [150, 150, 150]
        
        # Draw menu items
        item_height = 20
        for i in range(4):
            item_y = menu_y + 20 + i * (item_height + 10)
            frame[item_y:item_y+item_height, menu_x+10:menu_x+menu_width-10] = [200, 200, 200]
        
        # Convert to tensor and CHW format
        tensor_frame = torch.tensor(frame, dtype=self.dtype, device=self.device)
        tensor_frame = tensor_frame.permute(2, 0, 1)  # HWC to CHW
        
        return tensor_frame
    
    def _update_city_state(self, action: int) -> float:
        """Update city state based on action.
        
        Args:
            action: Action taken
            
        Returns:
            float: Reward for the action
        """
        # Define action ranges for different city mechanics
        if action < 5:  # Road actions
            # Building roads increases traffic temporarily but enables growth
            self.traffic = min(1.0, self.traffic + 0.05)
            self.budget -= 200
            reward = 0.1
            
        elif action < 10:  # Residential zone actions
            # More residential zones increase population
            self.population += random.randint(50, 200)
            self.budget -= 150
            reward = 0.2
            
        elif action < 15:  # Commercial/industrial actions
            # Commercial zones improve happiness but may increase traffic
            self.happiness = min(1.0, self.happiness + 0.03)
            self.traffic = min(1.0, self.traffic + 0.02)
            self.pollution = min(1.0, self.pollution + 0.01)
            self.budget -= 300
            reward = 0.15
            
        else:  # Special buildings or services
            # Services cost money but improve happiness
            self.happiness = min(1.0, self.happiness + 0.05)
            self.pollution = max(0.0, self.pollution - 0.02)
            self.budget -= 500
            reward = 0.3
        
        # Natural changes
        self.population = max(1000, int(self.population * (0.99 + 0.02 * self.happiness - 0.01 * self.pollution)))
        self.budget += int(self.population * 0.1)  # Tax income
        self.budget -= int(self.population * 0.05)  # Maintenance costs
        
        # Constraints and natural decay
        self.happiness = max(0.0, min(1.0, self.happiness - 0.01))
        self.traffic = max(0.0, min(1.0, self.traffic - 0.02))
        self.pollution = max(0.0, min(1.0, self.pollution - 0.01))
        
        # Calculate reward based on city state
        state_reward = (
            0.1 * self.happiness +
            0.1 * (1.0 - self.pollution) +
            0.1 * (1.0 - self.traffic) +
            0.02 * (self.budget / 10000)
        )
        
        # Update some cells in the city grid based on action
        for _ in range(3):
            x = random.randint(0, 49)
            y = random.randint(0, 49)
            if self.city_grid[y, x] == 0:
                self.city_grid[y, x] = action % 5 + 1
        
        return reward + state_reward 