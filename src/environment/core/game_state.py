"""
Game state tracking for Cities: Skylines 2 environment.

This module tracks and manages the game state including menus,
performance, and other state variables.
"""

import torch
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)

class GameState:
    """Tracks and manages the state of the Cities: Skylines 2 game."""
    
    def __init__(self, history_length: int = 10):
        """Initialize game state tracking.
        
        Args:
            history_length: Length of history to maintain for observations and rewards
        """
        # Game state flags
        self.in_menu = False
        self.paused = False
        self.game_speed = 1  # 1=normal, 2=fast, 3=fastest
        self.game_crashed = False
        
        # History tracking
        self.history_length = history_length
        self.observation_history = deque(maxlen=history_length)
        self.reward_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)
        self.menu_history = deque(maxlen=history_length)
        
        # Menu tracking
        self.menu_stuck_counter = 0
        self.max_menu_stuck_steps = 5
        self.menu_entry_time = 0
        self.last_menu_exit_time = 0
        self.menu_entry_count = 0
        
        # Game window tracking
        self.game_window_missing_count = 0
        self.max_window_missing_threshold = 5
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
    def reset(self):
        """Reset the game state."""
        # Reset state flags
        self.in_menu = False
        self.paused = False
        self.game_speed = 1
        self.game_crashed = False
        
        # Clear histories
        self.observation_history.clear()
        self.reward_history.clear()
        self.action_history.clear()
        self.menu_history.clear()
        self.fps_history.clear()
        
        # Reset counters
        self.menu_stuck_counter = 0
        self.menu_entry_count = 0
        self.game_window_missing_count = 0
        self.last_frame_time = time.time()
        
    def update(self, 
              observation: torch.Tensor, 
              reward: float, 
              in_menu: bool,
              action_success: bool,
              action_info: Optional[Dict[str, Any]] = None):
        """Update the game state with new information.
        
        Args:
            observation: Current observation
            reward: Current reward
            in_menu: Whether the game is in a menu
            action_success: Whether the last action was successful
            action_info: Information about the last action
        """
        # Update histories
        self.observation_history.append(observation)
        self.reward_history.append(reward)
        if action_info:
            self.action_history.append(action_info)
        self.menu_history.append(in_menu)
        
        # Update menu state
        self.update_menu_state(in_menu)
        
        # Update FPS tracking
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        fps = 1.0 / frame_time if frame_time > 0 else 60.0
        self.fps_history.append(fps)
        self.last_frame_time = current_time
        
    def update_menu_state(self, in_menu: bool):
        """Update menu state tracking.
        
        Args:
            in_menu: Whether the game is currently in a menu
        """
        # If menu state changed
        if in_menu != self.in_menu:
            if in_menu:
                # Just entered menu
                self.menu_entry_time = time.time()
                self.menu_entry_count += 1
                self.menu_stuck_counter = 0
            else:
                # Just exited menu
                self.last_menu_exit_time = time.time()
                
            # Update current menu state
            self.in_menu = in_menu
        elif in_menu:
            # Still in menu, increment stuck counter
            self.menu_stuck_counter += 1
            
            # Log warning if stuck too long
            if self.menu_stuck_counter >= self.max_menu_stuck_steps:
                logger.warning(f"Potentially stuck in menu for {self.menu_stuck_counter} steps")
        else:
            # Not in menu, reset stuck counter
            self.menu_stuck_counter = 0
            
    def update_window_status(self, window_found: bool):
        """Update game window status.
        
        Args:
            window_found: Whether the game window was found
        """
        if window_found:
            self.game_window_missing_count = 0
            self.game_crashed = False
        else:
            self.game_window_missing_count += 1
            
            # Check if game crashed
            if self.game_window_missing_count >= self.max_window_missing_threshold:
                logger.warning("Game window not found for multiple checks, marking as crashed")
                self.game_crashed = True
    
    def get_average_reward(self) -> float:
        """Get the average reward over the history.
        
        Returns:
            float: Average reward
        """
        if not self.reward_history:
            return 0.0
        return sum(self.reward_history) / len(self.reward_history)
    
    def get_average_fps(self) -> float:
        """Get the average FPS over the history.
        
        Returns:
            float: Average FPS
        """
        if not self.fps_history:
            return 60.0
        return sum(self.fps_history) / len(self.fps_history)
    
    def is_menu_stuck(self) -> bool:
        """Check if the game appears to be stuck in a menu.
        
        Returns:
            bool: Whether the game appears stuck in a menu
        """
        return self.in_menu and self.menu_stuck_counter >= self.max_menu_stuck_steps
    
    def is_performance_stable(self) -> bool:
        """Check if the game performance is stable.
        
        Returns:
            bool: Whether the game performance is stable
        """
        # Need enough FPS samples
        if len(self.fps_history) < 5:
            return True
            
        # Calculate standard deviation of FPS
        fps_array = np.array(list(self.fps_history))
        fps_std = np.std(fps_array)
        fps_mean = np.mean(fps_array)
        
        # Check if FPS is stable and reasonable
        return fps_std / max(fps_mean, 1.0) < 0.3 and fps_mean > 10.0
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current game state.
        
        Returns:
            Dict: Summary of current game state
        """
        return {
            'in_menu': self.in_menu,
            'paused': self.paused,
            'game_speed': self.game_speed,
            'game_crashed': self.game_crashed,
            'avg_reward': self.get_average_reward(),
            'avg_fps': self.get_average_fps(),
            'menu_stuck': self.is_menu_stuck(),
            'menu_entries': self.menu_entry_count,
            'performance_stable': self.is_performance_stable()
        } 