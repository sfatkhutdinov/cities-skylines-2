"""
Action space definition for Cities: Skylines 2 environment.

This module defines and manages the possible actions the agent can take
in the Cities: Skylines 2 environment.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class ActionSpace:
    """Defines and manages the action space for the Cities Skylines 2 agent."""
    
    def __init__(self, observation_manager=None):
        """Initialize action space.
        
        Args:
            observation_manager: Observation manager reference for screen dimensions
        """
        self.observation_manager = observation_manager
        self.actions = self._setup_actions()
        self.num_actions = len(self.actions)
        
        logger.info(f"Action space initialized with {self.num_actions} possible actions")
    
    def _setup_actions(self) -> Dict[int, Dict[str, Any]]:
        """Setup the action space for the agent using default game key bindings.
        
        Returns:
            Dict: Mapping from action index to action parameters
        """
        # Get screen dimensions if available
        screen_width, screen_height = self._get_screen_dimensions()
        
        base_actions = {
            # Speed control actions (0.0 to 1.0)
            0: {"type": "speed", "speed": 0.0},  # Slowest
            1: {"type": "speed", "speed": 0.25}, # Slow
            2: {"type": "speed", "speed": 0.5},  # Medium
            3: {"type": "speed", "speed": 0.75}, # Fast
            4: {"type": "speed", "speed": 1.0},  # Fastest
            
            # Basic camera movements (no semantic meaning, just key presses)
            5: {"type": "key", "key": "w", "duration": 0.1},
            6: {"type": "key", "key": "s", "duration": 0.1},
            7: {"type": "key", "key": "a", "duration": 0.1},
            8: {"type": "key", "key": "d", "duration": 0.1},
            9: {"type": "key", "key": "r", "duration": 0.1},
            10: {"type": "key", "key": "f", "duration": 0.1},
            11: {"type": "key", "key": "q", "duration": 0.1},
            12: {"type": "key", "key": "e", "duration": 0.1},
            13: {"type": "key", "key": "t", "duration": 0.1},
            14: {"type": "key", "key": "g", "duration": 0.1},
            
            # Basic UI interactions (no semantic meaning, just mouse actions)
            15: {"type": "mouse", "action": "click", "button": "left"},
            16: {"type": "mouse", "action": "click", "button": "right", "position": (0.5, 0.5)},
            17: {"type": "mouse", "action": "double_click", "button": "left"},
            18: {"type": "mouse", "action": "drag", "button": "left"},
            19: {"type": "mouse", "action": "scroll", "direction": 1},
            20: {"type": "mouse", "action": "scroll", "direction": -1},
            
            # Edge scrolling actions for map navigation
            21: {"type": "mouse", "action": "edge_scroll", "direction": "up", "duration": 0.5},
            22: {"type": "mouse", "action": "edge_scroll", "direction": "down", "duration": 0.5},
            23: {"type": "mouse", "action": "edge_scroll", "direction": "left", "duration": 0.5},
            24: {"type": "mouse", "action": "edge_scroll", "direction": "right", "duration": 0.5},
            
            # Basic game controls (no semantic meaning, just key presses)
            25: {"type": "key", "key": "space", "duration": 0.1},
            26: {"type": "key", "key": "1", "duration": 0.1},
            27: {"type": "key", "key": "2", "duration": 0.1},
            28: {"type": "key", "key": "3", "duration": 0.1},
            29: {"type": "key", "key": "b", "duration": 0.1},
            30: {"type": "key", "key": "escape", "duration": 0.1},
            
            # Basic info keys (no semantic meaning, just key presses)
            31: {"type": "key", "key": "p", "duration": 0.1},
            32: {"type": "key", "key": "z", "duration": 0.1},
            33: {"type": "key", "key": "c", "duration": 0.1},
            34: {"type": "key", "key": "v", "duration": 0.1},
            35: {"type": "key", "key": "x", "duration": 0.1},
            36: {"type": "key", "key": "m", "duration": 0.1},
        }
        
        # Create grid of points across screen (10x10 grid = 100 additional actions)
        grid_size = 10
        action_offset = 37  # Start after the base actions
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate normalized coordinates (0.0 to 1.0)
                x = (i + 0.5) / grid_size
                y = (j + 0.5) / grid_size
                
                # Create click action at this position
                action_idx = action_offset + i * grid_size + j
                base_actions[action_idx] = {
                    "type": "mouse", 
                    "action": "click", 
                    "button": "left",
                    "position": (x, y)
                }
        
        return base_actions
    
    def _get_screen_dimensions(self) -> Tuple[int, int]:
        """Get dimensions of game screen.
        
        Returns:
            Tuple[int, int]: Width and height of game screen
        """
        default_width, default_height = 1920, 1080
        
        # Try to get from observation manager
        if self.observation_manager and hasattr(self.observation_manager, 'screen_capture'):
            screen_capture = self.observation_manager.screen_capture
            
            # Check if client position is available
            if hasattr(screen_capture, 'client_position') and screen_capture.client_position:
                client_left, client_top, client_right, client_bottom = screen_capture.client_position
                return client_right - client_left, client_bottom - client_top
            
            # Check if get_resolution is available
            if hasattr(screen_capture, 'get_resolution'):
                try:
                    width, height = screen_capture.get_resolution()
                    if width > 0 and height > 0:
                        return width, height
                except Exception:
                    pass
                
        return default_width, default_height
    
    def get_action(self, action_idx: int) -> Dict[str, Any]:
        """Get action dictionary for the given action index.
        
        Args:
            action_idx: Index of the action to get
            
        Returns:
            Dict with action parameters
        """
        # Check if action index is valid
        if action_idx < 0 or action_idx >= self.num_actions:
            logger.warning(f"Invalid action index: {action_idx}, using random action")
            # Return a random action instead
            action_idx = self.sample()
            
        # Get action info
        action_info = self.actions[action_idx]
        logger.critical(f"ACTION SPACE: Returning action for index {action_idx}: {action_info}")
        return action_info
    
    def sample(self) -> int:
        """Sample a random action from the action space.
        
        Returns:
            int: Random action index
        """
        return np.random.randint(0, self.num_actions)
    
    def action_to_one_hot(self, action_idx: int) -> torch.Tensor:
        """Convert action index to one-hot encoded tensor.
        
        Args:
            action_idx: Action index
            
        Returns:
            torch.Tensor: One-hot encoded action
        """
        one_hot = torch.zeros(self.num_actions)
        if 0 <= action_idx < self.num_actions:
            one_hot[action_idx] = 1.0
        return one_hot
    
    def get_action_description(self, action_idx: int) -> str:
        """Get human-readable description of an action.
        
        Args:
            action_idx: Action index
            
        Returns:
            str: Description of the action
        """
        if action_idx < 0 or action_idx >= self.num_actions:
            return "Invalid action"
            
        action = self.actions[action_idx]
        action_type = action.get("type", "unknown")
        
        if action_type == "key":
            key = action.get("key", "unknown")
            duration = action.get("duration", 0.1)
            return f"Press key '{key}' for {duration:.1f}s"
            
        elif action_type == "mouse":
            mouse_action = action.get("action", "unknown")
            button = action.get("button", "left")
            position = action.get("position", None)
            if position:
                x, y = position
                return f"Mouse {mouse_action} with {button} button at position ({x:.2f}, {y:.2f})"
            else:
                return f"Mouse {mouse_action} with {button} button at current position"
                
        elif action_type == "speed":
            speed = action.get("speed", 0)
            speed_name = {0.0: "Slowest", 0.25: "Slow", 0.5: "Normal", 0.75: "Fast", 1.0: "Fastest"}
            return f"Set game speed to {speed_name.get(speed, f'{speed:.1f}')}"
            
        else:
            return f"Unknown action type: {action_type}"

    def get_num_actions(self) -> int:
        """Get the number of possible actions.
        
        Returns:
            int: Number of actions
        """
        return self.num_actions 