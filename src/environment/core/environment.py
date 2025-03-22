"""
Core environment interface for Cities: Skylines 2.

This module provides the base environment interface that handles
game interaction, state management, and reward computation.
"""

import torch
import time
import numpy as np
import logging
import os
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from src.config.hardware_config import HardwareConfig
from src.environment.core.game_state import GameState
from src.environment.core.observation import ObservationManager
from src.environment.core.action_space import ActionSpace
from src.environment.core.performance import PerformanceMonitor
from src.environment.core.error_recovery import ErrorRecovery

# Import from other modules that we've already modularized
from src.environment.rewards.reward_system import AutonomousRewardSystem
from src.environment.menu.menu_handler import MenuHandler
from src.environment.input.actions import InputSimulator

logger = logging.getLogger(__name__)

class Environment:
    """Environment class for interacting with Cities: Skylines 2.
    
    The Environment class handles the interaction with the game environment.
    It provides methods for starting and stopping the environment, executing actions,
    and retrieving observations and rewards.
    
    The reward system has been enhanced to specifically deter menu access:
    - Severe penalty (-5.0) for entering a menu (via ESC key or gear icon)
    - Small reward (+0.5) for exiting a menu using the ESC key
    - Penalty (-2.0) for taking any non-ESC actions while in a menu
    - Small penalty (-1.0) for actions similar to ones that previously caused menu entry
    
    This design aims to train the agent to avoid menu interactions entirely, and if menus
    are accidentally accessed, to exit them immediately using the ESC key.
    """
    
    def __init__(self, 
                 config: Optional[HardwareConfig] = None, 
                 mock_mode: bool = False,
                 menu_screenshot_path: Optional[str] = None,
                 disable_menu_detection: bool = False,
                 game_path: Optional[str] = None,
                 process_name: str = "CitiesSkylines2",
                 window_title: Optional[str] = None,
                 skip_game_check: bool = False,
                 **kwargs):
        """Initialize the Cities: Skylines 2 environment.
        
        Args:
            config: Hardware configuration
            mock_mode: Whether to use mock mode (no actual game interaction)
            menu_screenshot_path: Path to a screenshot of a menu for detection
            disable_menu_detection: Whether to disable menu detection
            game_path: Path to the game executable
            process_name: Name of the game process
            window_title: Title of the game window
            skip_game_check: Skip game process verification (assume game is running)
            **kwargs: Additional arguments
        """
        # Basic setup
        self.config = config or HardwareConfig()
        self.mock_mode = mock_mode
        self.device = self.config.get_device()
        self.dtype = self.config.get_dtype()
        self.disable_menu_detection = disable_menu_detection
        self.game_path = game_path
        self.process_name = process_name
        self.window_title = window_title or "Cities: Skylines II"
        self.skip_game_check = skip_game_check
        
        if self.skip_game_check:
            logger.info("Game process verification is disabled - assuming game is already running")
        
        if self.disable_menu_detection:
            logger.info("Menu detection is disabled - menu detection and recovery will be skipped")
            
        # If config is a wrapped object with get method, update capture config with window title
        if hasattr(self.config, 'get') and callable(getattr(self.config, 'get')):
            capture_config = self.config.get('capture', {})
            capture_config['window_title'] = self.window_title
            
            # Create updated config
            if hasattr(self.config, 'sections'):
                self.config.sections['capture'] = capture_config
        
        # Initialize the core components
        self.observation_manager = ObservationManager(config=self.config, mock_mode=mock_mode)
        self._action_space_manager = ActionSpace(observation_manager=self.observation_manager)
        self.game_state = GameState(
            enabled=not self.mock_mode,
            mock_mode=self.mock_mode
        )
        self.performance_monitor = PerformanceMonitor(config=self.config)
        
        # Get references to key components from observation manager
        self.screen_capture = self.observation_manager.screen_capture
        self.visual_metrics = self.observation_manager.visual_metrics
        
        # Initialize the input system
        input_simulator = InputSimulator(config=self.config)
        
        # Store reference to input simulator and get action executor
        self.input_simulator = input_simulator
        self.action_executor = input_simulator.get_action_executor()
        
        # Find and focus game window
        if not self.mock_mode:
            logger.info("Finding and focusing game window")
            if not self.input_simulator.mouse_controller.find_game_window(self.window_title):
                logger.warning("Failed to find game window, input may not work correctly")
            else:
                logger.info("Successfully found and focused game window")
        
        # Initialize menu handler if not disabled
        self.menu_handler = None
        if not disable_menu_detection:
            # If menu_screenshot_path is None, use a default path
            templates_dir = menu_screenshot_path or "menu_templates"
            
            self.menu_handler = MenuHandler(
                observation_manager=self.observation_manager,
                input_simulator=input_simulator,
                performance_monitor=self.performance_monitor,
                templates_dir=templates_dir
            )
            # Link menu handler to other components
            self.screen_capture.menu_handler = self.menu_handler
            
        # Initialize error recovery system
        self.error_recovery = ErrorRecovery(
            process_name=self.process_name,
            game_path=self.game_path,
            max_restart_attempts=3,
            timeout_seconds=120,
            input_simulator=input_simulator,
            skip_game_check=self.skip_game_check
        )
        
        # Register callbacks for game restart events
        self.error_recovery.register_callbacks(
            pre_restart=self._pre_restart_callback,
            post_restart=self._post_restart_callback
        )
        
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
        self.game_window_title = kwargs.get('window_name', "Cities: Skylines II")
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5  # Maximum number of consecutive errors before giving up
        
        # Create OpenAI gym-compatible observation and action spaces
        from gymnasium.spaces import Box, Discrete
        self._observation_space = Box(
            low=0, high=255, 
            shape=self.observation_manager.get_observation_shape(), 
            dtype=np.uint8
        )
        
        # Use a different name for the action space property 
        # to avoid conflict with the ActionSpace instance
        self._gym_action_space = Discrete(self._action_space_manager.get_num_actions())
        
        # Initialize action tracking
        self.action_stats = {}
        self.action_success_rates = {}
        
        # Frame stacking for better temporal context
        self.frame_stack_size = self.config.get("environment", {}).get("frame_stack_size", 4)
        logger.critical(f"Initializing with frame stack size: {self.frame_stack_size}")
        self.frame_buffer = deque(maxlen=self.frame_stack_size)
        
        # Menu handling state tracking
        self.previous_menu_state = False
        self._last_menu_entry_cause = None  # Track what caused menu entry: 'escape_key', 'gear_icon', or 'unknown'
        self.last_menu_entry_time = 0
        self.last_menu_entry_action_idx = -1
        self.last_menu_causing_action = None
        
        logger.info("Environment initialized successfully.")
    
    @property
    def observation_space(self):
        """Get the observation space for use with RL algorithms.
        
        Returns:
            gym.spaces.Box: Observation space
        """
        return self._observation_space
    
    @property
    def action_space(self):
        """Get the action space for use with RL algorithms.
        
        Returns:
            gym.spaces.Discrete: Action space
        """
        return self._gym_action_space
    
    def reset(self) -> torch.Tensor:
        """Reset the environment and return the initial observation.
        
        Returns:
            torch.Tensor: Initial observation
        """
        logger.critical("Resetting environment")
        
        # Reset components
        logger.critical("Resetting observation manager")
        self.observation_manager.reset()
        logger.critical("Resetting game state")
        self.game_state.reset()
        logger.critical("Resetting performance monitor")
        self.performance_monitor.reset()
        logger.critical("Resetting error recovery state")
        self.error_recovery.reset_error_state()
        
        # Reset counters
        self.steps_taken = 0
        self.last_action_time = time.time()
        self.consecutive_errors = 0
        
        # Verify game is running
        logger.critical("Ensuring game is running")
        self._ensure_game_running()
        
        # Reset game speed
        logger.critical("Setting game speed")
        self._set_game_speed(1)
        
        # Reset frame buffer for stacking
        logger.critical("Resetting frame buffer")
        self.frame_buffer = deque(maxlen=self.frame_stack_size)
        
        # Get first observation and fill frame buffer
        observation = self.get_observation()
        for _ in range(self.frame_stack_size):
            self.frame_buffer.append(observation)
        
        # Return stacked observation
        stacked_observation = self._get_stacked_observation()
        
        return stacked_observation
    
    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Execute action in the environment and return next observation, reward, done, and info.
        
        Args:
            action_idx: Index of the action to execute
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        start_time = time.time()
        
        # Update step counter
        self.steps_taken += 1
        
        # Reset fail count if we're over the threshold
        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.warning(f"Exceeded max consecutive errors ({self.max_consecutive_errors}). Resetting environment.")
            self.reset()
            observation = self.get_observation()
            return observation, -1.0, True, {"action_success": False, "reset": True}
        
        # Ensure the game is running and the window is focused
        if not self._ensure_game_running():
            # Failed to ensure the game is running
            self.consecutive_errors += 1
            observation = self.get_observation()
            return observation, -1.0, False, {"action_success": False, "game_running": False}
        
        # Ensure window is focused
        self._ensure_window_focused()
        
        # Check if we're in a menu before executing the action
        in_menu_before_action = self.check_menu_state()
        menu_type_before = self.menu_handler.get_menu_type() if in_menu_before_action else None
        
        # Get the action to execute
        action_info = self._action_space_manager.get_action(action_idx)
        
        # Check if this action is ESC key or potentially a gear icon click (top-right click)
        is_escape_action = action_info.get('type') == 'key' and action_info.get('key') == 'escape'
        is_top_right_click = (action_info.get('type') == 'mouse' and action_info.get('action') == 'click' and
                            isinstance(action_info.get('position'), tuple) and 
                            action_info.get('position')[0] > 0.75)
        
        # Log the action information
        if is_escape_action:
            logger.debug(f"Executing ESCAPE KEY action: {action_info}")
        elif is_top_right_click:
            logger.debug(f"Executing POTENTIAL GEAR ICON (top-right) click: {action_info}")
        else:
            logger.debug(f"Executing action {action_idx}: {action_info}")
        
        # Execute the action
        success, action_results = self.action_executor.execute_action(action_info)
        
        # Update our internal tracking for consecutive failures
        self._track_action_result(action_idx, success)
        
        # Check menu state after action
        in_menu_after_action = self.check_menu_state()
        menu_type_after = self.menu_handler.get_menu_type() if in_menu_after_action else None
        
        # Detect menu entry/exit transitions
        menu_entered = not in_menu_before_action and in_menu_after_action
        menu_exited = in_menu_before_action and not in_menu_after_action
        
        # Update specific menu context based on transitions
        if menu_entered:
            if is_escape_action:
                logger.warning("ESC key press caused menu entry")
                self._last_menu_entry_cause = "escape_key"
            elif is_top_right_click:
                logger.warning("Top-right click (likely gear icon) caused menu entry")
                self._last_menu_entry_cause = "gear_icon"
            else:
                logger.warning(f"Menu entered after action {action_idx} (unknown cause)")
                self._last_menu_entry_cause = "unknown"
                
        if menu_exited and is_escape_action:
            logger.info("Successfully exited menu using ESC key")
        
        # Add a small delay to ensure that the action is completed
        time.sleep(self.min_action_delay)
        
        # Get observation after action
        observation = self.get_observation()
        
        # Compute reward
        reward = self._compute_reward(action_info, action_idx, success, in_menu_after_action)
        
        # Check if the episode is done
        done = self.steps_taken >= self.max_steps
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Prepare info dictionary
        info = {
            "action_success": success,
            "in_menu": in_menu_after_action,
            "menu_type": menu_type_after,
            "step_count": self.steps_taken,
            "response_time": response_time,
            "menu_entered": menu_entered,
            "menu_exited": menu_exited,
            "is_escape_action": is_escape_action,
            "is_potential_gear_click": is_top_right_click
        }
        
        # Include the action results if available
        if action_results:
            info["action_results"] = action_results
            
        return observation, reward, done, info
    
    def get_observation(self) -> torch.Tensor:
        """Get the current observation from the environment.
        
        Returns:
            torch.Tensor: Current observation
        """
        logger.critical("Getting observation from environment")
        try:
            logger.critical("Calling observation_manager.get_observation()")
            observation = self.observation_manager.get_observation()
            
            # Log basic information about the observation
            if observation is not None:
                logger.critical(f"Observation received successfully: shape={observation.shape}, device={observation.device}, dtype={observation.dtype}")
                if torch.isnan(observation).any():
                    logger.critical("WARNING: Observation contains NaN values!")
                if torch.isinf(observation).any():
                    logger.critical("WARNING: Observation contains Inf values!")
            else:
                logger.critical("Observation is None! This should not happen.")
                
            return observation
        except Exception as e:
            # Log detailed error information
            import traceback
            error_trace = traceback.format_exc()
            logger.critical(f"ERROR getting observation: {e}")
            logger.critical(f"Error traceback: {error_trace}")
            self.consecutive_errors += 1
            
            logger.critical(f"Consecutive errors: {self.consecutive_errors}")
            
            # Return the previous observation if available, otherwise a blank one
            if self.current_frame is not None:
                logger.critical("Returning previous frame as fallback")
                return self.current_frame
                
            # Create a compatible blank observation
            logger.critical("Returning blank observation as fallback")
            shape = self.observation_manager.get_observation_shape()
            return torch.zeros(shape, dtype=torch.float32, device=self.device)
    
    def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing environment")
        try:
            self.observation_manager.close()
            self.performance_monitor.close()
            if hasattr(self.reward_system, 'close'):
                self.reward_system.close()
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
    
    def check_game_state(self) -> bool:
        """Check if the game is running and responding.
        
        Returns:
            bool: True if game state is healthy, False otherwise
        """
        # Check if game is running
        if not self.error_recovery.check_game_running():
            logger.warning("Game not running, attempting to restart")
            if not self.error_recovery.restart_game():
                logger.error("Failed to restart game")
                return False
            # Wait for game to initialize
            time.sleep(5)
            return True
        
        return True
    
    def check_menu_state(self) -> bool:
        """Check if the game is currently showing a menu.
        
        Returns:
            bool: True if in menu, False otherwise
        """
        if self.menu_handler and not self.disable_menu_detection:
            return self.menu_handler.check_menu_state()
        return False
    
    def restart_game(self) -> bool:
        """Restart the game.
        
        Returns:
            bool: True if restart successful, False otherwise
        """
        return self.error_recovery.restart_game()
    
    def _pre_restart_callback(self):
        """Called before game restart."""
        logger.info("Preparing for game restart...")
        # Save any necessary state
        self.game_state.save_state()
    
    def _post_restart_callback(self):
        """Called after game restart."""
        logger.info("Game restarted, reinitializing components...")
        # Reset observation manager
        self.observation_manager.reset()
        # Wait for game to initialize
        self._wait_for_game_start()
    
    def _ensure_game_running(self) -> None:
        """Make sure the game is running, start it if not."""
        if self.skip_game_check:
            logger.info("Skipping game process verification as requested")
            # Still wait a bit to ensure window is properly focused
            time.sleep(1) 
            return
            
        if not self.check_game_state():
            logger.warning("Game not running at environment initialization")
            if not self.error_recovery.restart_game():
                logger.error("Failed to start game at initialization")
                raise RuntimeError("Could not start game - please check game path and installation")
            self._wait_for_game_start()
    
    def _set_game_speed(self, speed_level: int) -> bool:
        """Set game speed using keyboard shortcuts.
        
        Args:
            speed_level: Game speed level (1-3)
            
        Returns:
            bool: True if the speed change was successful, False otherwise
        """
        if self.mock_mode:
            return True
            
        try:
            logger.debug(f"Setting game speed to level {speed_level}")
            # First set to minimum speed
            self.input_simulator.press_key('1')
            time.sleep(0.1)
            
            # Then set to desired speed
            if speed_level == 2:
                self.input_simulator.press_key('2')
            elif speed_level == 3:
                self.input_simulator.press_key('3')
            
            # Speed actions should be considered successful even with float values
            # Map float values to int speeds (0.75 should still use speed 1)
            if isinstance(speed_level, float):
                logger.info(f"Float speed value {speed_level} mapped to standard game speed")
            
            return True
        except Exception as e:
            logger.error(f"Error setting game speed: {e}")
            return False
    
    def _compute_reward(self, action_info: Dict, action_idx: int, success: bool, menu_detected: bool) -> float:
        """Compute reward based on action and observation.
        
        Args:
            action_info: Information about the executed action
            action_idx: Index of the executed action
            success: Whether the action was executed successfully
            menu_detected: Whether a menu was detected
            
        Returns:
            float: Computed reward
        """
        try:
            # If game issues, give negative reward
            if self.consecutive_errors > 0:
                logger.debug(f"Assigning negative reward due to game issues ({self.consecutive_errors} consecutive errors)")
                return -0.5 * self.consecutive_errors
            
            # Track menu entry/exit state changes
            menu_state_changed = False
            previous_menu_state = getattr(self, 'previous_menu_state', False)
            entered_menu = not previous_menu_state and menu_detected 
            exited_menu = previous_menu_state and not menu_detected
            menu_state_changed = entered_menu or exited_menu
            
            # Store current menu state for next comparison
            self.previous_menu_state = menu_detected
            
            # Store the action that caused menu entry for future penalty
            if entered_menu:
                # Check if the action was ESC key or a potential gear icon click
                is_escape_action = action_info.get('type') == 'key' and action_info.get('key') == 'escape'
                is_top_right_click = (action_info.get('type') == 'mouse' and action_info.get('action') == 'click' and
                                      isinstance(action_info.get('position'), tuple) and 
                                      action_info.get('position')[0] > 0.75)  # Top-right quadrant of screen
                
                menu_causing_action = None
                if is_escape_action:
                    menu_causing_action = "escape_key"
                    logger.warning("Agent pressed ESC key and entered menu - severe penalty applied")
                elif is_top_right_click:
                    menu_causing_action = "gear_icon"
                    logger.warning("Agent likely clicked gear icon and entered menu - severe penalty applied")
                
                # Store the action that caused menu entry
                self.last_menu_causing_action = menu_causing_action
                self.last_menu_entry_action_idx = action_idx
                self.last_menu_entry_time = time.time()
                
                # Severe penalty for entering the menu: -5.0
                return -5.0
            
            # Small positive reward for properly exiting the menu via ESC key
            if exited_menu:
                is_escape_action = action_info.get('type') == 'key' and action_info.get('key') == 'escape'
                
                if is_escape_action:
                    logger.info("Agent successfully exited menu using ESC key - small reward applied")
                    # Small reward: +0.5 for correctly exiting
                    return 0.5
            
            # If in menu, punish any action that's not ESC key
            if menu_detected:
                is_escape_action = action_info.get('type') == 'key' and action_info.get('key') == 'escape'
                
                if not is_escape_action:
                    # Punish taking non-escape actions while in menu (-2.0)
                    logger.warning(f"Agent took action {action_idx} while in menu - applying penalty")
                    return -2.0
                else:
                    # Neutral reward for attempting to escape - don't punish this
                    return 0.0
            
            # Check if the current action is similar to one that previously caused menu entry
            # This helps build causality awareness in the agent
            if not menu_detected and hasattr(self, 'last_menu_causing_action') and self.last_menu_causing_action:
                time_since_last_menu = time.time() - getattr(self, 'last_menu_entry_time', 0)
                
                if time_since_last_menu < 300:  # Only consider recent menu entries (within 5 minutes)
                    is_escape_action = action_info.get('type') == 'key' and action_info.get('key') == 'escape'
                    is_top_right_click = (action_info.get('type') == 'mouse' and action_info.get('action') == 'click' and
                                        isinstance(action_info.get('position'), tuple) and 
                                        action_info.get('position')[0] > 0.75)
                    
                    if (self.last_menu_causing_action == "escape_key" and is_escape_action) or \
                       (self.last_menu_causing_action == "gear_icon" and is_top_right_click):
                        # Apply a smaller penalty for trying an action that previously caused menu entry
                        logger.warning(f"Agent tried an action that previously caused menu entry - applying small penalty")
                        return -1.0
            
            # If action failed, give negative reward
            if not success:
                logger.debug(f"Assigning negative reward for failed action {action_idx}")
                return -0.1
                
            # Get state representation
            if hasattr(self.game_state, 'get_feature_vector'):
                state = self.game_state.get_feature_vector()
                # For the reward system, we need raw frames, not feature vectors
                current_frame = self.observation_manager.get_current_frame() 
                next_frame = self.observation_manager.get_latest_frame()
            else:
                # Use the raw frames directly
                current_frame = self.observation_manager.get_current_frame()
                next_frame = self.observation_manager.get_latest_frame()
            
            # Compute reward based on transition
            try:
                # Check if frames are available for reward computation
                if current_frame is None or next_frame is None:
                    logger.warning("Missing frames for reward computation. Defaulting to 0.0 reward.")
                    reward = 0.0
                else:
                    # Ensure action_idx is valid
                    action_idx = action_idx if isinstance(action_idx, (int, np.integer)) else 0
                    
                    # Check if the frames are valid tensors
                    if not torch.is_tensor(current_frame) or not torch.is_tensor(next_frame):
                        logger.warning("Frames are not valid tensors. Converting to tensors.")
                        if not torch.is_tensor(current_frame):
                            current_frame = torch.zeros((3, 84, 84), device=self.device)
                        if not torch.is_tensor(next_frame):
                            next_frame = torch.zeros((3, 84, 84), device=self.device)
                    
                    # Check for NaN values in frames
                    if torch.isnan(current_frame).any() or torch.isinf(current_frame).any():
                        logger.warning("NaN/Inf values detected in current frame. Replacing with zeros.")
                        current_frame = torch.zeros_like(current_frame)
                    
                    if torch.isnan(next_frame).any() or torch.isinf(next_frame).any():
                        logger.warning("NaN/Inf values detected in next frame. Replacing with zeros.")
                        next_frame = torch.zeros_like(next_frame)
                
                    # Pass done flag and info to reward system
                    done = self.steps_taken >= self.max_steps
                    info = action_info or {}
                
                    # Compute reward with all required parameters
                    reward = self.reward_system.compute_reward(
                        current_frame=current_frame, 
                        next_frame=next_frame, 
                        action=action_idx, 
                        done=done, 
                        info=info
                    )
                    
                    # Validate the reward
                    if math.isnan(reward) or math.isinf(reward):
                        logger.warning(f"Invalid reward value detected: {reward}. Defaulting to 0.0")
                        reward = 0.0
            except Exception as e:
                logger.error(f"Error computing reward: {e}")
                reward = 0.0

            logger.critical(f"REWARD DEBUG: Computed reward: {reward}\n")
            return reward
        except Exception as e:
            logger.error(f"Error computing reward: {e}")
            return 0.0  # Default reward on error
    
    def _wait_for_game_start(self) -> None:
        """Wait for the game to become responsive after starting."""
        if self.mock_mode:
            return
            
        logger.info("Waiting for game to become responsive...")
        
        start_time = time.time()
        timeout = 60  # Timeout in seconds
        
        while time.time() - start_time < timeout:
            # Check if game is running
            if not self.error_recovery.check_game_running():
                logger.warning("Game process not detected during startup")
                time.sleep(2)
                continue
                
            # Try to get a frame, if successful, game is responsive
            try:
                observation = self.observation_manager.get_observation()
                logger.info("Game is now responsive")
                return
            except Exception as e:
                logger.debug(f"Game not yet responsive: {e}")
                time.sleep(2)
                
        logger.error(f"Game did not become responsive within {timeout} seconds")
        raise TimeoutError(f"Game startup timeout after {timeout} seconds")

    def _ensure_window_focused(self):
        """Ensure the game window is focused before taking actions."""
        if self.mock_mode:
            return True
            
        if hasattr(self.screen_capture, 'focus_game_window'):
            logger.debug("Attempting to focus game window")
            
            # Try multiple focus attempts if needed
            for attempt in range(3):
                result = self.screen_capture.focus_game_window()
                if result:
                    logger.info(f"Successfully focused game window on attempt {attempt+1}")
                    # Add a consistent wait time after focus to ensure window is fully responsive
                    # This is critical - the window needs time to properly receive input after focus
                    time.sleep(0.75)  # Increased from the original value
                    return True
                else:
                    logger.warning(f"Failed to focus game window (attempt {attempt+1}/3)")
                    time.sleep(0.5)
            
            logger.error("Failed to focus game window after multiple attempts")
            return False
            
        logger.warning("Screen capture doesn't support window focusing")
        return True 

    def _track_action_result(self, action_idx: int, success: bool) -> None:
        """Track action execution results for statistics.
        
        Args:
            action_idx: Index of the action
            success: Whether the action was successful
        """
        # Initialize tracking for this action if needed
        if action_idx not in self.action_stats:
            self.action_stats[action_idx] = {
                'total': 0,
                'success': 0,
                'failure': 0
            }
        
        # Update counters
        self.action_stats[action_idx]['total'] += 1
        
        if success:
            self.action_stats[action_idx]['success'] += 1
        else:
            self.action_stats[action_idx]['failure'] += 1
        
        # Calculate success rate
        total = self.action_stats[action_idx]['total']
        successes = self.action_stats[action_idx]['success']
        success_rate = (successes / total) if total > 0 else 0
        
        # Store success rate
        self.action_success_rates[action_idx] = success_rate
        
        # Log the result if it's particularly bad
        if total >= 5 and success_rate < 0.5:
            action_info = self._action_space_manager.get_action_description(action_idx)
            logger.warning(f"Low success rate for action {action_idx} ({action_info}): {success_rate:.2f} ({successes}/{total})")
    
    def get_action_stats(self) -> Dict[str, Any]:
        """Get statistics about action execution.
        
        Returns:
            Dict: Statistics about action execution
        """
        stats = {
            'action_stats': self.action_stats,
            'success_rates': self.action_success_rates,
        }
        
        # Identify problematic actions (success rate < 0.7 with at least 5 attempts)
        problematic = {}
        for action_idx, rate in self.action_success_rates.items():
            if rate < 0.7 and self.action_stats[action_idx]['total'] >= 5:
                action_info = self._action_space_manager.get_action_description(action_idx)
                problematic[action_idx] = {
                    'description': action_info,
                    'success_rate': rate,
                    'attempts': self.action_stats[action_idx]['total']
                }
        
        stats['problematic_actions'] = problematic
        return stats 

    def _get_stacked_observation(self):
        """Get stacked observations for frame stacking."""
        if len(self.frame_buffer) == 0:
            return None
            
        # Check if using image or vector observations
        sample_obs = self.frame_buffer[0]
        
        # Convert all tensors to CPU if they're on GPU
        cpu_frames = []
        for frame in self.frame_buffer:
            if isinstance(frame, torch.Tensor) and frame.is_cuda:
                cpu_frames.append(frame.cpu())
            else:
                cpu_frames.append(frame)
        
        # Handle image observations (stacked frames)
        if len(sample_obs.shape) == 3:  # Images: (C, H, W) or (H, W, C)
            if sample_obs.shape[0] == 3:  # If channels-first format
                return np.concatenate([f for f in cpu_frames], axis=0)
            else:  # If channels-last format
                stacked = np.concatenate([f for f in cpu_frames], axis=-1)
                # Convert to channels-first for PyTorch
                return np.transpose(stacked, (2, 0, 1))
        else:
            # Vector observation - concatenate
            return np.concatenate([f.flatten() for f in cpu_frames]) 