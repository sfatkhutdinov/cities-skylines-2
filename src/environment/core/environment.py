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
from typing import Dict, List, Tuple, Optional, Any

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
    """Core environment interface for Cities: Skylines 2."""
    
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
        self.game_state = GameState()
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
        
        # Capture initial observation
        logger.critical("Capturing initial observation")
        try:
            self.current_frame = self.get_observation()
            logger.critical(f"Observation captured successfully with shape {self.current_frame.shape if hasattr(self.current_frame, 'shape') else 'unknown'}")
        except Exception as e:
            logger.critical(f"Error capturing initial observation: {e}")
            import traceback
            logger.critical(f"Traceback: {traceback.format_exc()}")
            # Create an empty observation as fallback
            logger.critical("Creating fallback observation")
            self.current_frame = torch.zeros(self.observation_space.shape)
        
        return self.current_frame
    
    def step(self, action_idx: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Execute an action in the environment.
        
        Args:
            action_idx: Index of the action to execute
            
        Returns:
            Tuple: (observation, reward, done, info)
        """
        # Get action info from action space
        action_info = self._action_space_manager.get_action(action_idx)
        
        # Add explicit debug logging for the action being executed
        action_type = action_info.get("type", "unknown")
        if action_type == "key":
            key = action_info.get("key", "unknown")
            duration = action_info.get("duration", 0.1)
            logger.critical(f"ACTION DEBUG: Executing keyboard action - key={key}, duration={duration}")
        elif action_type == "mouse":
            mouse_action = action_info.get("action", "unknown")
            button = action_info.get("button", "left")
            position = action_info.get("position", None)
            direction = action_info.get("direction", None)
            logger.critical(f"ACTION DEBUG: Executing mouse action - action={mouse_action}, button={button}, position={position}, direction={direction}")
        else:
            logger.critical(f"ACTION DEBUG: Executing action of type: {action_type}")
            
        logger.critical(f"FULL ACTION INFO: {action_info}")
        
        # Update step counter
        self.steps_taken += 1
        
        # Check for game issues first
        if not self.check_game_state():
            # If game has issues, return a negative reward and observation that indicates an error
            observation = self.get_observation()
            reward = -1.0  # Penalty for game issues
            done = self.steps_taken >= self.max_steps or self.consecutive_errors >= self.max_consecutive_errors
            info = {
                'error': True,
                'error_type': 'game_issue',
                'steps': self.steps_taken,
                'action': action_idx,
                'success': False,
                'in_menu': False,
                'focus_success': False,
                'action_retries': 0,
                'fps': self.performance_monitor.get_fps(),
                'error_stats': self.error_recovery.get_error_stats() if self.consecutive_errors > 0 else {},
                'action_stats': self.get_action_stats()
            }
            return observation, reward, done, info
        
        # Find and focus the game window using the mouse controller directly first
        # This is more reliable than the error_recovery method
        logger.critical("FOCUS DEBUG: Attempting to find and focus game window directly...")
        window_found = self.input_simulator.mouse_controller.find_game_window(self.window_title)
        logger.critical(f"FOCUS DEBUG: Window found and focused: {window_found}")
        
        # As a fallback, try the error_recovery method
        focus_success = window_found
        if not window_found and hasattr(self.error_recovery, 'focus_game_window'):
            logger.critical("FOCUS DEBUG: Attempting to focus using error_recovery fallback...")
            focus_success = self.error_recovery.focus_game_window()
            logger.critical(f"FOCUS DEBUG: Focus using error_recovery: {focus_success}")
        
        # Detect if we're in a menu
        menu_detected = False
        if hasattr(self, 'check_menu_state'):
            logger.critical("MENU DEBUG: Checking if in menu...")
            try:
                menu_detected = self.check_menu_state()
                logger.critical(f"MENU DEBUG: In menu: {menu_detected}")
            except Exception as e:
                logger.error(f"Error checking menu state: {e}")
        
        # Try to handle menu if detected
        if menu_detected and hasattr(self.input_simulator, 'handle_menu_recovery'):
            logger.critical("MENU DEBUG: Attempting to recover from menu state...")
            try:
                menu_recovered = self.input_simulator.handle_menu_recovery(retries=1)
                logger.critical(f"MENU DEBUG: Menu recovery attempt: {'success' if menu_recovered else 'failed'}")
            except Exception as e:
                logger.error(f"Error in menu recovery: {e}")
        
        # Prepare for action execution
        success = False
        action_retries = 0
        max_action_retries = 3  # Maximum number of retry attempts for failed actions
        
        # Create delay between getting focus and executing action
        time.sleep(0.5)  # 500ms delay to ensure window has focus
        
        # Force refocusing before executing action (this is critical!)
        if not self.mock_mode:
            logger.critical("FOCUS DEBUG: Ensuring window focus before action execution...")
            window_found = self.input_simulator.mouse_controller.find_game_window(self.window_title)
            logger.critical(f"FOCUS DEBUG: Window find before execution: {'succeeded' if window_found else 'failed'}")
            time.sleep(0.2)  # Brief delay after focus
        
        while not success and action_retries < max_action_retries:
            try:
                # Ensure window is focused before each attempt - important for retries
                if action_retries > 0:
                    logger.critical(f"FOCUS DEBUG: Re-focusing window before retry {action_retries}")
                    self.input_simulator.mouse_controller.find_game_window(self.window_title)
                    time.sleep(0.5)  # Wait a bit more before retry
                
                logger.critical(f"ACTION EXEC DEBUG: Attempting to execute action (attempt {action_retries + 1})")
                
                # Execute the action
                if action_type == "key":
                    if "key" in action_info:
                        key = action_info["key"]
                        duration = action_info.get("duration", 0.1)
                        logger.critical(f"KEY DEBUG: Pressing key {key} for {duration}s")
                        success = self.action_executor.execute_action("key_press", key=key, duration=duration)
                    else:
                        logger.critical("KEY DEBUG: Missing 'key' in action_info")
                        success = False
                        
                elif action_type == "mouse":
                    if "action" in action_info:
                        mouse_action = action_info["action"]
                        
                        if mouse_action == "click":
                            if "position" in action_info:
                                x, y = action_info["position"]
                                # Convert normalized coordinates to screen coordinates
                                screen_width, screen_height = self.screen_capture.get_resolution()
                                x = int(x * screen_width)
                                y = int(y * screen_height)
                                logger.critical(f"MOUSE CLICK DEBUG: Clicking at x={x}, y={y}, button={action_info.get('button', 'left')}")
                                success = self.action_executor.execute_action("mouse_click", x=x, y=y, button=action_info.get("button", "left"))
                            else:
                                logger.critical("MOUSE CLICK DEBUG: No position specified, clicking at current position")
                                success = self.action_executor.execute_action("mouse_click", button=action_info.get("button", "left"))
                                
                        elif mouse_action == "scroll":
                            if "direction" in action_info:
                                direction = action_info["direction"]
                                logger.critical(f"MOUSE SCROLL DEBUG: Scrolling {direction}")
                                success = self.action_executor.execute_action("mouse_scroll", direction=direction)
                            else:
                                logger.critical("MOUSE SCROLL DEBUG: Missing 'direction' for scroll")
                                success = False
                                
                        elif mouse_action == "edge_scroll":
                            if "direction" in action_info:
                                direction = action_info["direction"]
                                duration = action_info.get("duration", 0.5)
                                logger.critical(f"EDGE SCROLL DEBUG: Edge scrolling {direction} for {duration}s")
                                success = self.action_executor.execute_action("edge_scroll", direction=direction, duration=duration)
                            else:
                                logger.critical("EDGE SCROLL DEBUG: Missing 'direction' for edge_scroll")
                                success = False
                        else:
                            logger.critical(f"MOUSE DEBUG: Unknown mouse action: {mouse_action}")
                            success = False
                    else:
                        logger.critical("MOUSE DEBUG: Missing 'action' in mouse action_info")
                        success = False
                        
                elif action_type == "speed":
                    if "speed" in action_info:
                        speed = action_info["speed"]
                        logger.critical(f"SPEED DEBUG: Setting game speed to {speed}")
                        success = self._set_game_speed(speed)
                    else:
                        logger.critical("SPEED DEBUG: Missing 'speed' in action_info")
                        success = False
                else:
                    logger.critical(f"ACTION DEBUG: Unknown action type: {action_type}")
                    success = False
                
                # Check if action was successful
                logger.critical(f"ACTION EXEC DEBUG: Action execution {'succeeded' if success else 'failed'}")
                if not success:
                    logger.warning(f"Action execution failed (attempt {action_retries + 1}/{max_action_retries})")
                    self.consecutive_errors += 1
                    
                    # If this wasn't the last retry, try again
                    if action_retries < max_action_retries - 1:
                        action_retries += 1
                        # Force refocus the window between retries
                        window_found = self.input_simulator.mouse_controller.find_game_window(self.window_title)
                        logger.critical(f"FOCUS DEBUG: Window refocus for retry {'succeeded' if window_found else 'failed'}")
                        time.sleep(1.0)  # Keep 1.0s delay before retry
                    else:
                        break
                        
            except Exception as e:
                logger.critical(f"Error executing action: {e}")
                import traceback
                logger.critical(f"Traceback: {traceback.format_exc()}")
                self.consecutive_errors += 1
                
                # If this wasn't the last retry, try again
                if action_retries < max_action_retries - 1:
                    action_retries += 1
                    # Force refocus the window between retries
                    window_found = self.input_simulator.mouse_controller.find_game_window(self.window_title)
                    logger.critical(f"FOCUS DEBUG: Window refocus for retry {'succeeded' if window_found else 'failed'}")
                    time.sleep(1.0)  # Keep 1.0s delay before retry
                else:
                    break
        
        # Brief delay after action execution before capturing the next observation
        time.sleep(0.3)
        
        # Get next observation
        logger.critical("OBSERVATION DEBUG: Getting next observation after action")
        next_observation = self.get_observation()
        
        # Check for frozen game
        if hasattr(self.error_recovery, 'check_game_frozen') and self.error_recovery.check_game_frozen(next_observation):
            logger.warning("Game appears to be frozen, attempting recovery")
            if self.error_recovery.restart_game():
                # Game was restarted, get fresh observation
                next_observation = self.get_observation()
            else:
                # Failed to restart, mark as error
                self.consecutive_errors += 1
        
        # Update state with new observation
        self.current_frame = next_observation
        
        # Compute reward
        reward = self._compute_reward(action_info, action_idx, success, menu_detected)
        logger.critical(f"REWARD DEBUG: Computed reward: {reward}")
        
        # Check if episode is done
        done = self.steps_taken >= self.max_steps
        
        # Reset consecutive errors counter on success
        if success:
            self.consecutive_errors = 0
        
        # Create info dictionary
        info = {
            'steps': self.steps_taken,
            'action': action_idx,
            'success': success,
            'in_menu': menu_detected,
            'focus_success': focus_success,
            'action_retries': action_retries,
            'fps': self.performance_monitor.get_fps(),
            'error_stats': self.error_recovery.get_error_stats() if self.consecutive_errors > 0 else {},
            'action_stats': self.get_action_stats()
        }
        
        logger.critical(f"STEP DEBUG: Step completed - success={success}, reward={reward}, done={done}")
        return next_observation, reward, done, info
    
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
    
    def _set_game_speed(self, speed_level: int) -> None:
        """Set game speed using keyboard shortcuts.
        
        Args:
            speed_level: Game speed level (1-3)
        """
        if self.mock_mode:
            return
            
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
        except Exception as e:
            logger.error(f"Error setting game speed: {e}")
    
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
                
            # If in menu, give negative reward
            if menu_detected:
                logger.debug("Assigning negative reward for being in menu")
                return -0.2
                
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
                
            # Compute reward using reward system
            if current_frame is not None and next_frame is not None:
                reward = self.reward_system.compute_reward(current_frame, action_idx, next_frame)
            else:
                logger.warning("Missing frames for reward computation, using default reward")
                reward = 0.0
            
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