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
        logger.info("Resetting environment")
        
        # Reset components
        self.observation_manager.reset()
        self.game_state.reset()
        self.performance_monitor.reset()
        self.error_recovery.reset_error_state()
        
        # Reset counters
        self.steps_taken = 0
        self.last_action_time = time.time()
        self.consecutive_errors = 0
        
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
        action_info = self._action_space_manager.get_action(action_idx)
        
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
                'action': action_idx
            }
            return observation, reward, done, info
        
        # Ensure game window is focused before interacting with it
        # Try up to 3 times with increasing force
        focus_success = False
        for focus_attempt in range(3):
            # More aggressive focus on later attempts
            force_focus = focus_attempt > 0
            if force_focus:
                logger.info(f"Retry focus with force=True (attempt {focus_attempt+1}/3)")
            
            if self._ensure_window_focused():
                focus_success = True
                # The _ensure_window_focused method now includes a delay after successful focus
                break
            elif focus_attempt < 2:  # Don't wait after the last attempt
                time.sleep(0.5)
        
        if not focus_success:
            logger.warning("Failed to focus game window after multiple attempts")
            # Return an error observation with a negative reward
            observation = self.get_observation()
            reward = -1.0  # Penalty for focus failure
            done = False
            info = {
                'error': True,
                'error_type': 'focus_failure',
                'steps': self.steps_taken,
                'action': action_idx
            }
            return observation, reward, done, info
        
        # Check if we're in a menu
        menu_detected = False
        if self.menu_handler and self.steps_taken % 30 == 0:  # Check periodically
            menu_detected = self.menu_handler.check_menu_state()
            self.game_state.in_menu = menu_detected
        
        # Execute appropriate action based on menu state
        success = False
        action_retries = 0
        max_action_retries = 3  # Increased from 2 to 3 for more reliability
        
        if menu_detected and self.menu_handler:
            # In menu - try to recover
            success = self.menu_handler.handle_menu_recovery()
            if not success:
                # If menu recovery failed, try error recovery
                success = self.error_recovery.handle_menu_detection()
        else:
            # In game - execute requested action with retries
            while not success and action_retries <= max_action_retries:
                try:
                    # Ensure minimum time between actions
                    elapsed = time.time() - self.last_action_time
                    if elapsed < self.min_action_delay:
                        time.sleep(self.min_action_delay - elapsed)
                    
                    # Check if focus was lost and try to regain it before retrying
                    if action_retries > 0:
                        logger.info(f"Retrying action execution (attempt {action_retries+1}/{max_action_retries+1})")
                        # Try harder to focus on retries
                        if not self._ensure_window_focused():
                            logger.warning("Failed to refocus window during action retry")
                            time.sleep(0.5)  # Wait a bit and try again anyway
                        
                    # Execute action
                    success = self.action_executor.execute_action(action_info)
                    self.last_action_time = time.time()
                    
                    # Track action success/failure for statistics
                    self._track_action_result(action_idx, success)
                    
                    # Break if successful
                    if success:
                        break
                    
                    # If failed but can retry, give a short delay
                    if not success and action_retries < max_action_retries:
                        logger.warning(f"Action execution failed, will retry after delay")
                        time.sleep(1.0)  # Increased wait time before retry
                    
                    action_retries += 1
                    
                except Exception as e:
                    logger.error(f"Error executing action {action_idx}: {e}")
                    success = False
                    self.consecutive_errors += 1
                    
                    # If this wasn't the last retry, try again
                    if action_retries < max_action_retries:
                        action_retries += 1
                        # Try to refocus the window in case the error was caused by lost focus
                        self._ensure_window_focused()
                        time.sleep(1.0)  # Increased wait time before retry
                    else:
                        break
        
        # Get next observation
        next_observation = self.get_observation()
        
        # Check for frozen game
        if self.error_recovery.check_game_frozen(next_observation):
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
        
        # Check if episode is done
        done = self.steps_taken >= self.max_steps
        
        # Reset consecutive errors counter if successful
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
        
        return next_observation, reward, done, info
    
    def get_observation(self) -> torch.Tensor:
        """Get the current observation from the environment.
        
        Returns:
            torch.Tensor: Current observation
        """
        try:
            observation = self.observation_manager.get_observation()
            return observation
        except Exception as e:
            # Log the specific error for debugging
            logger.error(f"Error getting observation: {e}")
            self.consecutive_errors += 1
            
            # Return the previous observation if available, otherwise a blank one
            if self.current_frame is not None:
                return self.current_frame
                
            # Create a compatible blank observation
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
            else:
                state = self.current_frame
                
            # Compute reward using reward system
            reward = self.reward_system.compute_reward(state, action_idx)
            
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
                    logger.debug(f"Successfully focused game window on attempt {attempt+1}")
                    # Add a consistent wait time after focus to ensure window is fully responsive
                    time.sleep(0.75)  # Increased from the 0.5s in the debug script that worked
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