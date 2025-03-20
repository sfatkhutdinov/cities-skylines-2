"""
Core environment interface for Cities: Skylines 2.

This module provides the base environment interface that handles
game interaction, state management, and reward computation.
"""

import torch
import time
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any

from src.config.hardware_config import HardwareConfig
from src.environment.core.game_state import GameState
from src.environment.core.observation import ObservationManager
from src.environment.core.action_space import ActionSpace
from src.environment.core.performance import PerformanceMonitor

# Import from other modules that we've already modularized
from src.environment.rewards.reward_system import AutonomousRewardSystem
from src.environment.menu.menu_handler import MenuHandler
from src.environment.input.actions import ActionExecutor

logger = logging.getLogger(__name__)

class Environment:
    """Core environment interface for Cities: Skylines 2."""
    
    def __init__(self, 
                 config: Optional[HardwareConfig] = None, 
                 mock_mode: bool = False,
                 menu_screenshot_path: Optional[str] = None,
                 disable_menu_detection: bool = False,
                 **kwargs):
        """Initialize the Cities: Skylines 2 environment.
        
        Args:
            config: Hardware configuration
            mock_mode: Whether to use mock mode (no actual game interaction)
            menu_screenshot_path: Path to a screenshot of a menu for detection
            disable_menu_detection: Whether to disable menu detection
            **kwargs: Additional arguments
        """
        # Basic setup
        self.config = config or HardwareConfig()
        self.mock_mode = mock_mode
        self.device = self.config.get_device()
        self.dtype = self.config.get_dtype()
        self.disable_menu_detection = disable_menu_detection
        
        if self.disable_menu_detection:
            logger.info("Menu detection is disabled - menu detection and recovery will be skipped")
        
        # Initialize the core components
        self.observation_manager = ObservationManager(config=self.config, mock_mode=mock_mode)
        self.action_space = ActionSpace(observation_manager=self.observation_manager)
        self.game_state = GameState()
        self.performance_monitor = PerformanceMonitor(config=self.config)
        
        # Get references to key components from observation manager
        self.screen_capture = self.observation_manager.screen_capture
        self.visual_metrics = self.observation_manager.visual_metrics
        
        # Initialize the input system
        self.action_executor = ActionExecutor(
            screen_capture=self.screen_capture,
            config=self.config
        )
        
        # Initialize menu handler if not disabled
        self.menu_handler = None
        if not disable_menu_detection:
            self.menu_handler = MenuHandler(
                screen_capture=self.screen_capture,
                input_simulator=self.action_executor.input_simulator,
                visual_metrics=self.visual_metrics,
                menu_screenshot_path=menu_screenshot_path
            )
            # Link menu handler to other components
            self.screen_capture.menu_handler = self.menu_handler
        
        # Initialize reward system
        self.reward_system = AutonomousRewardSystem(
            config=self.config,
            hardware_accelerated=True,
            checkpoints_dir="checkpoints/reward_system"
        )
        
        # Setup game state tracking variables
        self.current_frame = None
        self.steps_taken = 0
        self.max_steps = kwargs.get('max_steps', 1000)
        self.last_action_time = time.time()
        self.min_action_delay = 0.1  # Minimum delay between actions
        self.game_window_title = "Cities: Skylines II"
        
        logger.info("Environment initialized successfully.")
    
    def reset(self) -> torch.Tensor:
        """Reset the environment and return the initial observation.
        
        Returns:
            torch.Tensor: Initial observation
        """
        logger.info("Resetting environment")
        
        # Reset components
        self.observation_manager.reset()
        self.game_state.reset()
        self.performance_monitor.reset()
        
        # Reset counters
        self.steps_taken = 0
        self.last_action_time = time.time()
        
        # Verify game is running
        self._ensure_game_running()
        
        # Reset game speed
        self._set_game_speed(1)
        
        # Capture initial observation
        self.current_frame = self.get_observation()
        
        return self.current_frame
    
    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Execute an action in the environment.
        
        Args:
            action_idx: Index of the action to execute
            
        Returns:
            Tuple: (observation, reward, done, info)
        """
        # Get action info from action space
        action_info = self.action_space.get_action(action_idx)
        
        # Check if we're in a menu
        menu_detected = False
        if self.menu_handler and self.steps_taken % 30 == 0:  # Check periodically
            menu_detected = self.menu_handler.check_menu_state()
            self.game_state.in_menu = menu_detected
        
        # Execute appropriate action based on menu state
        if menu_detected and self.menu_handler:
            # In menu - try to recover
            success = self.menu_handler.handle_menu_recovery()
        else:
            # Normal gameplay - execute action
            success = self.action_executor.execute_action(action_info)
        
        # Small delay to let game process the action
        time.sleep(max(0, self.min_action_delay - (time.time() - self.last_action_time)))
        self.last_action_time = time.time()
        
        # Get next observation
        next_frame = self.get_observation()
        
        # Compute reward
        reward = self._compute_reward(action_info, action_idx, success, menu_detected)
        
        # Update game state
        self.game_state.update(next_frame, reward, menu_detected, success)
        
        # Update steps
        self.steps_taken += 1
        
        # Check if episode is done
        done = self.steps_taken >= self.max_steps
        
        # Update current frame
        self.current_frame = next_frame
        
        # Create info dict
        info = {
            'steps': self.steps_taken,
            'in_menu': menu_detected,
            'action_success': success,
            'action_info': action_info,
            'performance': self.performance_monitor.get_metrics()
        }
        
        # Periodically check and optimize performance
        if self.steps_taken % 50 == 0:
            self.performance_monitor.check_and_optimize(self)
        
        return next_frame, reward, done, info
    
    def get_observation(self) -> torch.Tensor:
        """Get the current observation.
        
        Returns:
            torch.Tensor: Current observation
        """
        return self.observation_manager.get_observation()
    
    def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing environment")
        self.observation_manager.close()
        self.reward_system.save_state()
        
    def _ensure_game_running(self) -> None:
        """Ensure the game is running and ready for interaction."""
        # Check if game is running
        running = self.action_executor.is_game_running(self.game_window_title)
        
        if not running:
            logger.warning("Game not running, waiting for it to start")
            self._wait_for_game_start()
    
    def _set_game_speed(self, speed_level: int) -> None:
        """Set the game speed.
        
        Args:
            speed_level: Speed level (1-3)
        """
        # Ensure valid speed level
        speed_level = max(1, min(3, speed_level))
        
        # Only update if different
        if self.game_state.game_speed != speed_level:
            # Execute speed change
            self.action_executor.set_game_speed(speed_level)
            self.game_state.game_speed = speed_level
    
    def _compute_reward(self, action_info: Dict, action_idx: int, success: bool, menu_detected: bool) -> float:
        """Compute reward for taking an action.
        
        Args:
            action_info: Information about the action
            action_idx: Index of the action
            success: Whether the action execution was successful
            menu_detected: Whether a menu was detected
            
        Returns:
            float: Reward value
        """
        # Penalty for being in a menu
        if menu_detected:
            return -0.1
        
        # Get frames for reward computation
        current_frame = self.observation_manager.get_previous_frame()
        next_frame = self.observation_manager.get_current_frame()
        
        if current_frame is None or next_frame is None:
            return 0.0
        
        # Convert action to one-hot
        action_tensor = torch.zeros(self.action_space.num_actions)
        action_tensor[action_idx] = 1.0
        
        # Compute reward using the autonomous reward system
        reward = self.reward_system.compute_reward(
            current_frame=current_frame.cpu().numpy(), 
            action=action_tensor.numpy(),
            next_frame=next_frame.cpu().numpy()
        )
        
        return reward
    
    def _wait_for_game_start(self) -> None:
        """Wait for the game to start."""
        logger.info("Waiting for game to start")
        
        # Try for up to 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            if self.action_executor.is_game_running(self.game_window_title):
                logger.info("Game started")
                # Give it a moment to initialize
                time.sleep(2)
                return
            time.sleep(1)
        
        logger.error("Game failed to start within timeout")
        raise RuntimeError("Game failed to start") 