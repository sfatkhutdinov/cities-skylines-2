"""
Game environment for Cities: Skylines 2.
Handles game interaction, observation, and reward computation.
"""

import torch
import time
import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
from .visual_metrics import VisualMetricsEstimator
from .autonomous_reward_system import AutonomousRewardSystem
from .optimized_capture import OptimizedScreenCapture
from src.utils.performance_safeguards import PerformanceSafeguards
from .input_simulator import InputSimulator
from src.config.hardware_config import HardwareConfig
import win32api
from .menu_handler import MenuHandler
from src.utils.image_utils import ImageUtils
import cv2


logger = logging.getLogger(__name__)

class CitiesEnvironment:
    """Environment for interacting with Cities: Skylines 2."""
    
    def __init__(self, config: Optional[HardwareConfig] = None, mock_mode: bool = False, menu_screenshot_path: Optional[str] = None, **kwargs):
        """Initialize the environment.
        
        Args:
            config (HardwareConfig, optional): Hardware configuration for optimization
            mock_mode (bool): Whether to use mock environment
            menu_screenshot_path (str, optional): Path to menu screenshot for reference
            **kwargs: Additional arguments
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Store config
        self.config = config if config else HardwareConfig()
        self.mock_mode = mock_mode
        self.menu_reference_path = menu_screenshot_path
        self.has_menu_reference = self.menu_reference_path is not None and os.path.exists(str(self.menu_reference_path))
        
        # Process kwargs for additional options
        self.disable_menu_detection = kwargs.get('disable_menu_detection', False)
        
        # Initialize components
        self.input_simulator = InputSimulator()
        
        # Create screen capture instance
        self.screen_capture = OptimizedScreenCapture(
            config=self.config
        )
        
        # Pass the screen capture reference to the input simulator
        self.input_simulator.screen_capture = self.screen_capture
        
        # Initialize visual metrics estimator
        self.visual_estimator = VisualMetricsEstimator(self.config)
        
        # Setup menu detection
        self.menu_detection_initialized = False
        self.detect_menu_every_n_steps = 10
        self.menu_stuck_counter = 0
        self.max_menu_stuck_steps = 5  # Initialize the missing attribute
        self.in_menu = False
        
        # Game crash detection
        self.game_crashed = False
        self.game_window_missing_count = 0
        self.max_window_missing_threshold = 5  # Number of consecutive checks before considering game crashed
        self.last_crash_check_time = time.time()
        self.crash_check_interval = 2.0  # Seconds between crash checks
        self.waiting_for_game_restart = False
        self.game_restart_check_interval = 5.0  # Seconds between checking if game has restarted
        self.last_game_restart_check = time.time()
        
        # Initialize image utilities for visual processing
        self.image_utils = ImageUtils(debug_mode=False)
        
        # Check if the menu reference exists in the specified path or root directory
        if menu_screenshot_path and os.path.exists(menu_screenshot_path):
            self.has_menu_reference = True
        elif menu_screenshot_path:
            # If not found in the specified path, try looking in the root directory
            root_path = os.path.join(os.getcwd(), os.path.basename(menu_screenshot_path))
            if os.path.exists(root_path):
                self.menu_reference_path = root_path
                self.has_menu_reference = True
                logger.info(f"Found menu reference image in root directory: {root_path}")
            else:
                self.has_menu_reference = False
                logger.warning(f"Menu reference image not found at {menu_screenshot_path} or {root_path}")
        else:
            # Check if there's a default menu_reference.png in the root directory
            default_path = os.path.join(os.getcwd(), "menu_reference.png")
            if os.path.exists(default_path):
                self.menu_reference_path = default_path
                self.has_menu_reference = True
                logger.info(f"Using default menu reference image found at {default_path}")
            else:
                self.has_menu_reference = False
        
        if self.has_menu_reference:
            try:
                self.visual_estimator.initialize_menu_detection(self.menu_reference_path)
                self.menu_detection_initialized = True
                logger.info(f"Initialized menu detection using reference image: {self.menu_reference_path}")
            except Exception as e:
                logger.error(f"Failed to initialize menu detection: {e}")
                self.has_menu_reference = False
                # Implement fallback menu detection
                logger.info("Setting up fallback menu detection based on color patterns")
                self.visual_estimator.setup_fallback_menu_detection()
        else:
            # No menu reference image, set up fallback menu detection
            logger.info("No menu reference image available, setting up fallback menu detection")
            self.visual_estimator.setup_fallback_menu_detection()
        
        # Use autonomous reward system
        self.reward_system = AutonomousRewardSystem(self.config)
        self.safeguards = PerformanceSafeguards(self.config)
        
        # Initialize menu handler for more advanced menu detection and recovery
        self.menu_handler = MenuHandler(
            screen_capture=self.screen_capture,
            input_simulator=self.input_simulator,
            visual_metrics=self.visual_estimator
        )
        
        # Define action space
        self.actions = self._setup_actions()
        self.num_actions = len(self.actions)
        
        # State tracking
        self.current_frame = None
        self.steps_taken = 0
        self.max_steps = kwargs.get('max_steps', 1000)
        self.last_action_time = time.time()
        self.min_action_delay = 0.1  # Minimum delay between actions
        
        # Game state tracking
        self.paused = False
        self.game_speed = 1
        
        # Performance tracking
        self.fps_history = []
        self.last_optimization_check = time.time()
        self.optimization_interval = 60  # Check optimization every 60 seconds
        
        # Add menu action suppression
        self.suppress_menu_actions = False
        self.menu_action_suppression_steps = 0
        self.max_menu_suppression_steps = 15  # How long to suppress menu actions after exiting a menu
        
        # Add adaptive frame skip
        self.adaptive_frame_skip = True
        self.min_frame_skip = 1
        self.max_frame_skip = 6
        self.current_frame_skip = getattr(self.config, 'frame_skip', 2)
        self.activity_level = 0.0  # 0.0 = passive, 1.0 = very active
        self.activity_history = []
        self.activity_history_max_len = 30
        
        # For mock mode, create a dummy frame
        if self.mock_mode:
            # Create a simple 3-channel frame (320x240 RGB)
            self.mock_frame = torch.ones((3, 240, 320), dtype=torch.float32)
            # Add some random elements to make it more realistic
            self.mock_frame = torch.rand_like(self.mock_frame)
            # Inform the screen capture to use mock mode
            self.screen_capture.use_mock = True
        
        # Menu handling
        self.detect_menu_every_n_steps = 30  # Check for menu every 30 steps
        
        # Game window
        self.game_window_title = "Cities: Skylines II"
        
        # New attributes
        self.in_menu = False
        
        # Game crash detection
        self.game_crashed = False
        self.game_window_missing_count = 0
        self.max_window_missing_threshold = 5  # Number of consecutive checks before considering game crashed
        self.last_crash_check_time = time.time()
        self.crash_check_interval = 2.0  # Seconds between crash checks
        self.waiting_for_game_restart = False
        self.game_restart_check_interval = 5.0  # Seconds between checking if game has restarted
        self.last_game_restart_check = time.time()
    
    def _setup_actions(self) -> Dict[int, Dict[str, Any]]:
        """Setup the action space for the agent using default game key bindings."""
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
            16: {"type": "mouse", "action": "click", "button": "right"},
            17: {"type": "mouse", "action": "double_click", "button": "left"},
            18: {"type": "mouse", "action": "drag", "button": "left"},
            19: {"type": "mouse", "action": "scroll", "direction": 1},
            20: {"type": "mouse", "action": "scroll", "direction": -1},
            
            # Basic game controls (no semantic meaning, just key presses)
            21: {"type": "key", "key": "space", "duration": 0.1},
            22: {"type": "key", "key": "1", "duration": 0.1},
            23: {"type": "key", "key": "2", "duration": 0.1},
            24: {"type": "key", "key": "3", "duration": 0.1},
            25: {"type": "key", "key": "b", "duration": 0.1},
            26: {"type": "key", "key": "escape", "duration": 0.1},
            
            # Basic info keys (no semantic meaning, just key presses)
            27: {"type": "key", "key": "p", "duration": 0.1},
            28: {"type": "key", "key": "z", "duration": 0.1},
            29: {"type": "key", "key": "c", "duration": 0.1},
            30: {"type": "key", "key": "v", "duration": 0.1},
            31: {"type": "key", "key": "x", "duration": 0.1},
            32: {"type": "key", "key": "m", "duration": 0.1},
        }
        
        # Create grid of points across screen (10x10 grid = 100 additional actions)
        grid_size = 10
        action_offset = 33  # Start after the base actions
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Get client area dimensions
                screen_width = 1920  # Default full HD
                screen_height = 1080
                
                if hasattr(self, 'screen_capture') and hasattr(self.screen_capture, 'client_position'):
                    client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
                    screen_width = client_right - client_left
                    screen_height = client_bottom - client_top
                
                x = int(screen_width * (i + 0.5) / grid_size)
                y = int(screen_height * (j + 0.5) / grid_size)
                action_idx = action_offset + i * grid_size + j
                base_actions[action_idx] = {
                    "type": "mouse", 
                    "action": "move",
                    "position": (x, y)
                }
                
                # Add click actions for each grid point
                action_idx = action_offset + 100 + i * grid_size + j
                base_actions[action_idx] = {
                    "type": "mouse", 
                    "action": "click",
                    "position": (x, y),
                    "button": "left"
                }
        
        # Add drag actions between random grid points with variable durations
        drag_count = 0
        max_drags = 50
        drag_offset = action_offset + 200
        
        for _ in range(max_drags):
            i1, j1 = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            i2, j2 = random.randint(0, grid_size-1), random.randint(0, grid_size-1)
            
            if i1 == i2 and j1 == j2:
                continue
                
            x1 = int(screen_width * (i1 + 0.5) / grid_size)
            y1 = int(screen_height * (j1 + 0.5) / grid_size)
            x2 = int(screen_width * (i2 + 0.5) / grid_size)
            y2 = int(screen_height * (j2 + 0.5) / grid_size)
            
            # Calculate duration based on distance
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            duration = min(2.0, max(0.5, distance / 500))  # Scale duration with distance
            
            action_idx = drag_offset + drag_count
            base_actions[action_idx] = {
                "type": "mouse", 
                "action": "drag",
                "start": (x1, y1),
                "end": (x2, y2),
                "button": "left",
                "duration": duration
            }
            drag_count += 1
            
            if drag_count >= max_drags:
                break
        
        return base_actions
    
    def capture_menu_reference(self, save_path: str = "menu_reference.png") -> bool:
        """Capture the current frame as a menu reference image.
        
        This should be called when the menu is visible on screen.
        
        Args:
            save_path: Path to save the reference image
            
        Returns:
            bool: True if successfully saved, False otherwise
        """
        if self.mock_mode:
            logger.warning("Cannot capture menu reference in mock mode")
            return False
            
        # Focus the game window
        if not self.input_simulator.ensure_game_window_focused():
            logger.error("Could not focus game window to capture menu reference")
            return False
            
        # Capture current frame
        frame = self.screen_capture.capture_frame()
        if frame is None:
            logger.error("Failed to capture frame for menu reference")
            return False
            
        # Save as reference
        success = self.visual_estimator.save_current_frame_as_menu_reference(frame, save_path)
        if success:
            self.menu_reference_path = save_path
            self.has_menu_reference = True
            self.menu_detection_initialized = True
            logger.info(f"Successfully captured menu reference image to {save_path}")
        else:
            logger.error("Failed to save menu reference image")
            
        return success
    
    def reset(self) -> torch.Tensor:
        """Reset the environment and return initial observation.
        
        Returns:
            torch.Tensor: Initial observation
        """
        # Reset internal state variables
        self.steps_taken = 0
        self.episode_step = 0
        self.cumulative_reward = 0.0
        self.done = False
        self.menu_stuck_counter = 0
        self.suppress_menu_actions = False
        self.menu_action_suppression_steps = 0
        
        # Reset game state detection
        self.game_crashed = False
        self.game_window_missing_count = 0
        self.waiting_for_game_restart = False
        
        # Check if game is running
        if not self._check_game_running():
            logger.warning("Game does not appear to be running during reset. Will wait for game to start.")
            self._wait_for_game_start()
        
        # Capture initial frame
        frame = self.screen_capture.capture_frame()
        
        # If in mock mode, just return mock frame
        if self.mock_mode:
            self.current_frame = self._get_mock_frame()
            logger.info("Reset complete in mock mode")
            return self.current_frame
            
        # If for some reason we can't capture frame, return mock frame
        if frame is None:
            logger.warning("Could not capture initial frame, providing mock frame")
            self.current_frame = self._get_mock_frame()
            return self.current_frame
            
        # Process frame and keep it as current state
        self.current_frame = self._process_frame(frame)
        
        # Check if this is the first run and we don't have a menu reference yet
        if not self.has_menu_reference and not self.menu_detection_initialized:
            # On first run, assume we might be at the menu and capture a reference
            logger.info("First run, capturing menu reference image")
            
            # Save the current frame as a reference (assuming we're starting at the menu)
            self.capture_menu_reference("menu_reference.png")
            
            # Try to exit the menu immediately after capturing reference
            self.input_simulator.key_press('escape')
            time.sleep(0.7)
            self.input_simulator.key_press('escape')  # Press twice to be safe
            time.sleep(0.7)
            
            # Also try clicking the Resume Game button
            width, height = self._get_screen_dimensions()
            resume_x = int(width * 0.15)  # About 15% in from left
            resume_y = int(height * 0.25)  # About 25% down from top
            self.input_simulator.mouse_click(resume_x, resume_y)
            time.sleep(0.7)
            
            # Update the current frame
            initial_frame = self.screen_capture.capture_frame()
        
        # Keep trying to exit menu if detected
        menu_attempts = 0
        max_menu_exit_attempts = 5
        
        while self.check_menu_state() and menu_attempts < max_menu_exit_attempts:
            logger.info(f"Detected menu during reset, attempting to exit (attempt {menu_attempts+1})")
            
            # Try to exit menu
            self.input_simulator.key_press('escape')
            time.sleep(0.7)
            
            # Try clicking resume button
            width, height = self._get_screen_dimensions()
            resume_x = int(width * 0.15)
            resume_y = int(height * 0.25)
            self.input_simulator.mouse_click(resume_x, resume_y)
            time.sleep(0.7)
            
            # Try again with specific resume coords for main menu
            resume_x = int(width * 0.5)  # Center X
            resume_y = int(height * 0.4)  # About 40% down from top
            self.input_simulator.mouse_click(resume_x, resume_y)
            time.sleep(0.7)
            
            # Increment counter
            menu_attempts += 1
            
            # Update frame
            initial_frame = self.screen_capture.capture_frame()
        
        if menu_attempts >= max_menu_exit_attempts:
            logger.warning("Failed to exit menu after multiple attempts - continuing anyway")
        
        # Ensure mouse can move freely after menu handling
        # Move mouse to center of screen
        width, height = self._get_screen_dimensions()
        center_x, center_y = width // 2, height // 2
        self.input_simulator.mouse_move(center_x, center_y)
        
        # Capture again after potential menu exit
        self.current_frame = self.screen_capture.capture_frame()
        
        # Reset game state (unpause if paused)
        self._ensure_game_running()
        
        # Set normal game speed
        self._set_game_speed(1)
        
        # Run mouse freedom test during first reset to diagnose any issues
        # Only run on the first episode to avoid disrupting training
        if self.steps_taken == 0:
            self.test_mouse_freedom()
        
        return self.current_frame
    
    def _ensure_game_running(self):
        """Make sure game is not paused. Press space if it is."""
        # In mock mode, no need to do anything
        if self.mock_mode:
            return
            
        # Skip occasionally to avoid pressing space too often
        if random.random() < 0.7:  # 70% chance to skip
            return
            
        # Check if game is already running
        if not self.paused:
            return
            
        try:
            # Press space to unpause game
            self.input_simulator.key_press('space')
            self.paused = False
            logger.info("Game unpaused")
            
            # Check if it worked after a delay
            time.sleep(0.5)
            frame = self.screen_capture.capture_frame()
            
            # Further checks could be added here if needed
            
        except Exception as e:
            logger.error(f"Error ensuring game is running: {e}")
            
        # Reset mouse position occasionally to prevent getting stuck
        if random.random() > 0.8:  # 20% chance
            # Move mouse to center of screen
            width, height = self._get_screen_dimensions()
            center_x, center_y = width // 2, height // 2
            self.input_simulator.mouse_move(center_x, center_y)
    
    def _set_game_speed(self, speed_level=1):
        """Set game speed to a specific level.
        
        Args:
            speed_level (int): 0=paused, 1=normal, 2=fast, 3=fastest
        """
        # First make sure we're at speed 1 (normal)
        # Press 1 for normal speed
        self.input_simulator.press_key('1')
        time.sleep(0.1)
        
        # Then adjust to desired speed
        if speed_level == 0:  # Paused
            self.input_simulator.press_key('space')
        elif speed_level == 2:  # Fast
            self.input_simulator.press_key('2')
        elif speed_level == 3:  # Fastest
            self.input_simulator.press_key('3')
            
        logger.info(f"Game speed set to level {speed_level}")
        time.sleep(0.2)  # Give time for speed change to take effect
    
    def get_observation(self) -> torch.Tensor:
        """Get the current observation (frame) from the environment.
        
        Returns:
            torch.Tensor: Current observation frame
        """
        # In mock mode, return the mock frame
        if self.mock_mode:
            return self.mock_frame
            
        # Capture the current frame from the game
        frame = self.screen_capture.capture_frame()
        
        # If capture failed, try to focus the window and try again
        if frame is None:
            logger.warning("Frame capture failed, trying to focus window")
            self.input_simulator.ensure_game_window_focused()
            time.sleep(0.1)
            frame = self.screen_capture.capture_frame()
            
            if frame is None:
                logger.error("Frame capture failed again, returning zeros")
                # Return a black frame with appropriate dimensions
                return torch.zeros((3, 180, 320), device=self.config.get_device())
        
        return frame
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take an action in the environment.
        
        Args:
            action_idx: Index of the action to take
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: 
                - Next state
                - Reward
                - Done flag
                - Additional info
        """
        info = {"menu_detected": False, "game_crashed": False}
        
        # Check if the game has crashed before taking any action
        self._check_game_running()
        
        # If game has crashed, return a special state
        if self.game_crashed or self.waiting_for_game_restart:
            logger.warning("Game crash detected during step, waiting for restart")
            
            # Check if game has restarted
            if self._check_for_game_restart():
                logger.info("Game has restarted, resuming training")
                self.game_crashed = False
                self.waiting_for_game_restart = False
                self.game_window_missing_count = 0
            else:
                # Return placeholder state and info
                mock_frame = self._get_mock_frame()
                info["game_crashed"] = True
                info["waiting_for_restart"] = True
                
                # Return minimal reward, don't count as episode end
                return mock_frame, -0.1, False, info
        
        # If action_idx is None or invalid, choose a random action
        if action_idx is None or action_idx < 0 or action_idx >= len(self.actions):
            action_idx = random.randint(0, len(self.actions) - 1)
            
        action_info = self.actions[action_idx]
        
        # Check if we are in a menu and need to avoid certain actions
        if self.steps_taken % self.detect_menu_every_n_steps == 0:
            menu_detected = self.check_menu_state()
            
            if menu_detected:
                logger.info("Menu detected during step()")
                self.menu_stuck_counter += 1
                
                # If stuck in menu too long, try recovery
                if self.menu_stuck_counter >= self.max_menu_stuck_steps:
                    logger.warning(f"Stuck in menu for {self.menu_stuck_counter} steps, attempting recovery")
                    self._handle_menu_recovery()
                    self.menu_stuck_counter = 0
            else:
                self.menu_stuck_counter = 0
        
        # Execute the action and get success status
        frame_before_action = self.screen_capture.capture_frame()
        
        # Special handling for menu-related actions
        if self.in_menu and action_info.get("type") in ["mouse_move", "click_left"]:
            # For menu navigation, use click_menu_button if possible
            success = self._handle_menu_action(action_info)
        else:
            # Normal action execution
            success = self._execute_action(action_info)
        
        # Wait briefly to let the game process the action and update the display
        time.sleep(0.1)
        
        # Get the current state (screenshot)
        frame_after_action = self.screen_capture.capture_frame()
        
        # Check if we're in a menu
        menu_detected = self.check_menu_state()
        
        # Compute reward based on action result and visual change
        reward = self._compute_reward(action_info, action_idx, success, menu_detected)
        
        # Get the current state and format for the agent
        self.state = self._process_frame(frame_after_action)
        
        # Check if episode is done
        done = False
        if self.steps_taken >= self.max_steps:
            done = True
            logger.info(f"Episode done after {self.steps_taken} steps")
        
        # Collect info
        info = {
            "action_type": action_info.get("type", "unknown"),
            "success": success,
            "menu_detected": menu_detected,
            "menu_stuck_counter": self.menu_stuck_counter,
            "frame_skip": self.current_frame_skip if hasattr(self, 'current_frame_skip') else 1
        }
        
        return self.state, reward, done, info
    
    def check_menu_state(self) -> bool:
        """Check if the game is currently in a menu state.
        
        Returns:
            bool: True if a menu is detected, False otherwise
        """
        # If menu detection is disabled, always return False
        if self.disable_menu_detection:
            return False
        
        # In mock mode, randomly simulate menu states (for testing)
        if self.mock_mode:
            # Very low probability to be in a menu (5%)
            return random.random() < 0.05
        
        # Get current frame
        current_frame = self.screen_capture.capture_frame()
        if current_frame is None:
            logger.warning("Failed to capture frame for menu detection")
            return False
        
        # Use menu handler if available
        if hasattr(self, 'menu_handler') and self.menu_handler is not None:
            menu_detected, _, _ = self.menu_handler.detect_menu(current_frame)
            return menu_detected
        
        # Fallback to visual estimator
        try:
            return self.visual_estimator.detect_main_menu(current_frame)
        except Exception as e:
            logger.error(f"Error in menu detection: {e}")
            return False
    
    def _handle_menu_recovery(self, retries: int = 2) -> bool:
        """Try to recover from being stuck in a menu.
        
        Args:
            retries: Number of recovery attempts
            
        Returns:
            bool: True if recovery successful, False otherwise
        """
        # Use menu handler if available
        if hasattr(self, 'menu_handler') and self.menu_handler:
            logger.info("Using menu handler for recovery")
            return self.menu_handler.recover_from_menu(max_attempts=retries)
            
        # Fallback to legacy recovery
        logger.info("Using fallback menu recovery")
        success = self.input_simulator.handle_menu_recovery(retries=retries)
        return success
    
    def _execute_action(self, action_info: Dict) -> bool:
        """Execute an action in the environment.
        
        This method takes a low-level action (keypress, mouse movement, click)
        and executes it without any game-specific knowledge. The agent must learn
        which actions achieve desired outcomes in the game.
        
        Args:
            action_info (Dict): Dictionary containing action details
            
        Returns:
            bool: Whether the action was successfully executed
        """
        # Get the current time to enforce minimum delay between actions
        current_time = time.time()
        time_since_last_action = current_time - self.last_action_time
        
        # If action is being executed too soon, wait
        if time_since_last_action < self.min_action_delay:
            time.sleep(self.min_action_delay - time_since_last_action)
        
        # Update last action time
        self.last_action_time = time.time()
        
        # In mock mode, just simulate success/failure with high success probability
        if self.mock_mode:
            # 95% chance of success in mock mode
            return random.random() < 0.95
        
        # Detect action type and execute accordingly
        try:
            action_type = action_info.get("type", "")
            
            if action_type == "key":
                # Execute keyboard input
                key = action_info.get("key", "")
                
                # Additional safety check for ESC key which can trigger menus
                if key.lower() == "escape":
                    # Before pressing ESC, check if it might open a menu
                    menu_check_count = getattr(self, "menu_check_count", 0)
                    if menu_check_count > 2:  # Don't trigger too many menu checks
                        setattr(self, "menu_check_count", 0)
                    else:
                        setattr(self, "menu_check_count", menu_check_count + 1)
                
                # Hold key if specified
                hold_duration = action_info.get("duration", 0.1)  # Default to short press
                return self.input_simulator.key_press(key, duration=hold_duration)
            
            elif action_type == "mouse":
                # Get action and execute appropriate mouse function
                mouse_action = action_info.get("action", "")
                
                if mouse_action == "move":
                    # Get position coordinates
                    position = action_info.get("position", (0.5, 0.5))
                    if isinstance(position, tuple) and len(position) == 2:
                        screen_x, screen_y = position
                    else:
                        # Fallback to normalized coordinates (0.0 to 1.0)
                        norm_x = action_info.get("x", 0.5)
                        norm_y = action_info.get("y", 0.5)
                        
                        # Convert to screen coordinates
                        width, height = self._get_screen_dimensions()
                        screen_x = int(norm_x * width)
                        screen_y = int(norm_y * height)
                    
                    # Log the move for debugging
                    logger.debug(f"Moving mouse to: ({screen_x}, {screen_y})")
                    
                    # Execute move
                    return self.input_simulator.mouse_move(screen_x, screen_y)
                
                elif mouse_action == "click":
                    # Get button and double-click setting
                    button = action_info.get("button", "left")
                    double = action_info.get("double", False)
                    
                    # Get position if specified, otherwise use the current mouse position
                    if "position" in action_info:
                        position = action_info.get("position")
                        if isinstance(position, tuple) and len(position) == 2:
                            screen_x, screen_y = position
                        else:
                            # Fallback to center of screen
                            width, height = self._get_screen_dimensions()
                            screen_x, screen_y = width // 2, height // 2
                    else:
                        # Use current mouse position
                        try:
                            mouse_pos = self.input_simulator.get_mouse_position()
                            if isinstance(mouse_pos, tuple) and len(mouse_pos) == 2:
                                screen_x, screen_y = mouse_pos
                            else:
                                # Fallback to center of screen
                                width, height = self._get_screen_dimensions()
                                screen_x, screen_y = width // 2, height // 2
                                logger.warning(f"Invalid mouse position format: {mouse_pos}, using center of screen")
                        except Exception as e:
                            # Fallback to center of screen
                            width, height = self._get_screen_dimensions()
                            screen_x, screen_y = width // 2, height // 2
                            logger.warning(f"Could not get mouse position: {e}, using center of screen")
                    
                    # Execute click
                    return self.input_simulator.mouse_click(screen_x, screen_y, button=button, double=double)
                
                elif mouse_action in ["down", "up"]:
                    # Get button
                    button = action_info.get("button", "left")
                    
                    # Execute press/release
                    if mouse_action == "down":
                        return self.input_simulator.mouse_down(button=button)
                    else:  # up
                        return self.input_simulator.mouse_up(button=button)
                
                elif mouse_action == "scroll":
                    # Get direction and amount
                    direction = action_info.get("direction", "down")
                    amount = action_info.get("amount", 5)
                    
                    # Convert direction to sign for clicks
                    clicks = amount if direction == "up" else -amount
                    
                    # Execute scroll
                    return self.input_simulator.mouse_scroll(clicks)
                
                elif mouse_action == "drag":
                    # Handle drag action with proper error checking
                    width, height = self._get_screen_dimensions()
                    
                    # Get target normalized coordinates
                    to_norm_x = action_info.get("to_x", 0.5)
                    to_norm_y = action_info.get("to_y", 0.5)
                    
                    # Convert to screen coordinates
                    to_screen_x = int(to_norm_x * width)
                    to_screen_y = int(to_norm_y * height)
                    
                    # Get starting position with proper error handling
                    try:
                        mouse_pos = self.input_simulator.get_mouse_position()
                        if isinstance(mouse_pos, tuple) and len(mouse_pos) == 2:
                            from_screen_x, from_screen_y = mouse_pos
                        else:
                            # Default to center if not a valid tuple
                            from_screen_x, from_screen_y = width // 2, height // 2
                            logger.warning(f"Invalid mouse position format: {mouse_pos}, using center of screen as start")
                    except Exception:
                        # If can't get position, use center of screen
                        from_screen_x, from_screen_y = width // 2, height // 2
                        logger.warning("Failed to get mouse position, using center of screen as start")
                    
                    # Also check if start point is explicitly specified
                    if "start" in action_info:
                        start_pos = action_info.get("start")
                        if isinstance(start_pos, tuple) and len(start_pos) == 2:
                            from_screen_x, from_screen_y = start_pos
                    
                    # Execute drag
                    return self.input_simulator.mouse_drag(
                        start=(from_screen_x, from_screen_y), 
                        end=(to_screen_x, to_screen_y)
                    )
            
            elif action_type == "combined":
                # Combined key + mouse actions
                key = action_info.get("key", "")
                mouse_action = action_info.get("mouse_action", "click")
                button = action_info.get("button", "left")
                
                # Press key
                self.input_simulator.key_down(key)
                time.sleep(0.05)  # Brief delay
                
                success = False
                # Perform mouse action
                if mouse_action == "click":
                    # Get current mouse position for click
                    try:
                        mouse_pos = self.input_simulator.get_mouse_position()
                        if isinstance(mouse_pos, tuple) and len(mouse_pos) == 2:
                            screen_x, screen_y = mouse_pos
                            success = self.input_simulator.mouse_click(screen_x, screen_y, button=button)
                        else:
                            # Use center of screen if invalid format
                            width, height = self._get_screen_dimensions()
                            screen_x, screen_y = width // 2, height // 2
                            success = self.input_simulator.mouse_click(screen_x, screen_y, button=button)
                    except Exception as e:
                        logger.error(f"Error getting mouse position for combined action: {e}")
                        success = False
                
                # Release key
                time.sleep(0.05)  # Brief delay
                self.input_simulator.key_up(key)
                
                return success
            
            elif action_type == "wait":
                # Simply wait for specified duration
                duration = action_info.get("duration", 0.5)
                time.sleep(duration)
                return True
            
            elif action_type == "speed":
                # Handle game speed action
                speed_value = action_info.get("speed", 1.0)
                
                # Convert to speed level (0-4)
                if speed_value <= 0.1:
                    speed_level = 0
                elif speed_value <= 0.3:
                    speed_level = 1
                elif speed_value <= 0.6:
                    speed_level = 2
                elif speed_value <= 0.8:
                    speed_level = 3
                else:
                    speed_level = 4
                
                # Use the existing method to set game speed
                if hasattr(self, '_set_game_speed'):
                    self._set_game_speed(speed_level)
                    return True
                else:
                    logger.warning("Speed control not available")
                    return False
            
            else:
                # Unknown action type
                logger.warning(f"Unknown action type: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            return False
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics."""
        current_time = time.time()
        if self.last_action_time:
            fps = 1 / (current_time - self.last_action_time)
            self.fps_history.append(fps)
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
    
    def _check_and_optimize(self):
        """Check and apply performance optimizations."""
        current_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Apply safeguards and optimizations
        self.safeguards.optimize_training_parameters(current_fps)
        memory_status = self.safeguards.optimize_memory_usage()
        
        self.last_optimization_check = time.time()
    
    def close(self):
        """Clean up resources when environment is closed."""
        try:
            logger.info("Closing environment and releasing resources")
            
            # Release any held keys
            if hasattr(self, 'input_simulator'):
                try:
                    for key in ['w', 'a', 's', 'd', 'shift', 'ctrl', 'alt']:
                        self.input_simulator.release_key(key)
                except:
                    pass
            
            # Release all mouse buttons
            try:
                self.input_simulator.mouse.release(self.input_simulator.mouse.Button.left)
                self.input_simulator.mouse.release(self.input_simulator.mouse.Button.right)
                self.input_simulator.mouse.release(self.input_simulator.mouse.Button.middle)
            except:
                pass
                
            # Reset mouse to center position
            try:
                # Move mouse to center of screen
                width, height = self._get_screen_dimensions()
                center_x, center_y = width // 2, height // 2
                self.input_simulator.mouse_move(center_x, center_y)
            except:
                pass
                
            # Close other resources
            self.screen_capture.close()
            self.input_simulator.close()
            
            logger.info("Environment resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during environment cleanup: {e}")
    
    def render(self):
        """Display the current frame (for debugging)."""
        if self.current_frame is not None:
            # Convert tensor to numpy for display
            frame_np = self.current_frame.cpu().numpy()
            frame_np = np.transpose(frame_np, (1, 2, 0))  # CHW to HWC
            
            # You could use cv2.imshow here if needed
            return frame_np
        return None

    def test_mouse_freedom(self):
        """Test mouse freedom by moving to different positions on the screen.
        This helps verify the mouse can reach all parts of the game window.
        """
        if not hasattr(self.screen_capture, 'client_position'):
            logger.warning("Client position not available, using system metrics")
            width = win32api.GetSystemMetrics(0)
            height = win32api.GetSystemMetrics(1)
        else:
            client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
            width = client_right - client_left
            height = client_bottom - client_top
        
        logger.info(f"Testing mouse freedom across game window ({width}x{height})")
        
        # Define a grid of test points
        test_points = [
            (0, 0),                    # Top-left
            (width // 2, 0),           # Top-center
            (width - 1, 0),            # Top-right
            (0, height // 2),          # Middle-left
            (width // 2, height // 2), # Center
            (width - 1, height // 2),  # Middle-right
            (0, height - 1),           # Bottom-left
            (width // 2, height - 1),  # Bottom-center
            (width - 1, height - 1),   # Bottom-right
        ]
        
        # First ensure we can focus the window
        success = self.input_simulator.ensure_game_window_focused()
        if not success:
            logger.error("Failed to focus game window for mouse freedom test")
            return False
            
        # Record current position to return to at the end
        current_x, current_y = win32api.GetCursorPos()
        
        try:
            # Move to each test point with verification
            for i, (x, y) in enumerate(test_points):
                # Log if coordinates are outside normal bounds but don't restrict them
                if x < 0 or x >= width or y < 0 or y >= height:
                    print(f"Notice: Test point ({x}, {y}) is outside normal screen bounds")
                
                print(f"Moving to point {i+1}/{len(test_points)}: ({x}, {y})")
                
                # First attempt
                current_pos = win32api.GetCursorPos()
                print(f"Moving mouse: {current_pos[0]},{current_pos[1]} -> {x},{y}")
                
                # Use Win32 direct positioning for reliability
                self.input_simulator.mouse_move(x, y, use_win32=True)
                time.sleep(0.2)  # Wait for movement to complete
                
                # Verify position
                new_pos = win32api.GetCursorPos()
                if abs(new_pos[0] - x) > 5 or abs(new_pos[1] - y) > 5:
                    logger.warning(f"Mouse position verification failed: Expected ({x},{y}), got ({new_pos[0]},{new_pos[1]})")
                    # Try again with direct positioning
                    win32api.SetCursorPos((x, y))
                    time.sleep(0.2)
            
            # Return to center position
            center_x, center_y = width // 2, height // 2
            print(f"Moving mouse: {win32api.GetCursorPos()[0]},{win32api.GetCursorPos()[1]} -> {center_x},{center_y}")
            self.input_simulator.mouse_move(center_x, center_y, use_win32=True)
            
            print("Mouse freedom test completed")
            return True
            
        except Exception as e:
            logger.error(f"Error during mouse freedom test: {str(e)}")
            # Attempt to restore cursor position
            try:
                win32api.SetCursorPos((current_x, current_y))
            except:
                pass
            return False

    def _get_screen_dimensions(self) -> Tuple[int, int]:
        """Get the screen dimensions for the main game window.
        
        Returns:
            Tuple[int, int]: Width and height of the main game window
        """
        # Try to use input simulator to get screen dimensions
        if hasattr(self.input_simulator, 'get_screen_dimensions'):
            dims = self.input_simulator.get_screen_dimensions()
            if dims and dims[0] > 0 and dims[1] > 0:
                return dims
        
        # Fallback to capturing a frame and getting its dimensions
        curr_frame = self.screen_capture.capture_frame()
        
        if curr_frame is not None:
            if isinstance(curr_frame, torch.Tensor):
                # PyTorch tensor in CHW format
                if len(curr_frame.shape) == 3 and curr_frame.shape[0] == 3:
                    return curr_frame.shape[2], curr_frame.shape[1]
                return curr_frame.shape[1], curr_frame.shape[0]
            else:
                # Numpy array in HWC format
                return curr_frame.shape[1], curr_frame.shape[0]
                
        # Last resort fallback
        return 1920, 1080

    def _compute_reward(self, action_info: Dict, action_idx: int, success: bool, menu_detected: bool) -> float:
        """Compute the reward for an action.
        
        Args:
            action_info: Action information
            action_idx: Action index
            success: Whether the action was successful
            menu_detected: Whether a menu is detected
            
        Returns:
            float: Reward value
        """
        # Base reward for successful actions
        reward = 0.0
        
        # Penalty for unsuccessful actions 
        if not success:
            reward -= 0.1
            
        # Severe penalty for being in a menu (we want to avoid menus)
        if menu_detected:
            reward -= 1.0
            
        # Get frames before and after action
        prev_frame = getattr(self, 'prev_frame', None)
        curr_frame = self.screen_capture.capture_frame()
        
        # Store current frame for next reward computation
        if curr_frame is not None:
            # Handle PyTorch tensors properly
            if isinstance(curr_frame, torch.Tensor):
                self.prev_frame = curr_frame.clone().detach()
            else:
                # For numpy arrays
                self.prev_frame = curr_frame.copy() if curr_frame is not None else None
        
        # If we have visual diversity reward system, use it
        if hasattr(self, 'reward_system') and self.reward_system:
            # Get additional reward from autonomous system
            try:
                # We need prev_frame, action_idx (int), and curr_frame (np.ndarray)
                if prev_frame is not None and curr_frame is not None:
                    # Convert action_type to action_idx if needed
                    action_type = action_info.get("type", "unknown")
                    if isinstance(action_type, str):
                        # Map string action types to indices
                        action_type_map = {
                            "key": 0, 
                            "mouse": 1, 
                            "combined": 2, 
                            "wait": 3,
                            "speed": 4
                        }
                        action_idx = action_type_map.get(action_type, 0)
                    else:
                        action_idx = action_type
                    
                    autonomous_reward = self.reward_system.compute_reward(
                        prev_frame,
                        action_idx,
                        curr_frame
                    )
                    reward += autonomous_reward
            except Exception as e:
                logger.error(f"Error computing reward: {str(e)}")
                # Don't add any reward if there was an error
        
        # Scale reward - this is a hyperparameter you can tune
        reward *= 0.1
            
        return reward

    def _process_frame(self, frame) -> torch.Tensor:
        """Process a frame for input to neural network.
        
        Args:
            frame: Raw frame
            
        Returns:
            torch.Tensor: Processed frame tensor
        """
        if frame is None:
            # Return zeros if frame capture failed
            return torch.zeros((3, 180, 320), device=self.config.get_device())
            
        # If frame is already a PyTorch tensor, handle it differently
        if isinstance(frame, torch.Tensor):
            # Move to CPU for processing if needed
            frame_cpu = frame.detach().cpu()
            
            # Convert from CHW to HWC format if needed
            if len(frame_cpu.shape) == 3 and frame_cpu.shape[0] == 3:
                frame_np = frame_cpu.permute(1, 2, 0).numpy()
            else:
                frame_np = frame_cpu.numpy()
                
            # Ensure proper scaling (0-1 range)
            if frame_np.max() > 1.0:
                frame_np = frame_np / 255.0
        else:
            # Handle numpy array
            frame_np = frame
            
            # Convert BGR to RGB if needed
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:  # If it has 3 channels, assume BGR (OpenCV default)
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
                
            # Resize frame (if not already at target resolution)
            if frame_np.shape[0] != 180 or frame_np.shape[1] != 320:
                frame_np = cv2.resize(frame_np, (320, 180))
                
            # Normalize pixel values to [0,1]
            frame_np = frame_np.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor
        frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)  # HWC to CHW
        
        # Move to device
        if frame_tensor.device != self.config.get_device():
            frame_tensor = frame_tensor.to(self.config.get_device())
            
        return frame_tensor

    def _check_game_running(self) -> bool:
        """Check if the game is still running.
        
        Returns:
            bool: True if game is running, False otherwise
        """
        # Don't check too frequently
        current_time = time.time()
        if current_time - self.last_crash_check_time < self.crash_check_interval:
            return not self.game_crashed
            
        self.last_crash_check_time = current_time
        
        # Skip check in mock mode
        if self.mock_mode:
            return True
            
        # Try to find the game window
        has_window = self.input_simulator.find_game_window()
        
        if not has_window:
            self.game_window_missing_count += 1
            logger.warning(f"Game window not found (count: {self.game_window_missing_count}/{self.max_window_missing_threshold})")
            
            # If too many consecutive failures, consider the game crashed
            if self.game_window_missing_count >= self.max_window_missing_threshold:
                if not self.game_crashed:
                    logger.error("GAME CRASH DETECTED - waiting for game to restart")
                    self.game_crashed = True
                    self.waiting_for_game_restart = True
                return False
        else:
            # Reset counter if window found
            self.game_window_missing_count = 0
            
            # If we were in crashed state but now window is found,
            # game might have restarted
            if self.game_crashed:
                logger.info("Game window found after crash, resuming normal operation")
                self.game_crashed = False
                self.waiting_for_game_restart = False
                
        return not self.game_crashed
        
    def _check_for_game_restart(self) -> bool:
        """Check if the game has restarted after a crash.
        
        Returns:
            bool: True if game has restarted, False otherwise
        """
        # Don't check too frequently
        current_time = time.time()
        if current_time - self.last_game_restart_check < self.game_restart_check_interval:
            return False
            
        self.last_game_restart_check = current_time
        
        # Try to find the game window
        has_window = self.input_simulator.find_game_window()
        
        if has_window:
            # Try to capture a frame to verify game is actually running
            frame = self.screen_capture.capture_frame()
            if frame is not None:
                logger.info("Game has successfully restarted, frame captured")
                return True
                
        return False
        
    def _wait_for_game_start(self) -> None:
        """Wait for the game to start after a crash."""
        logger.info("Waiting for game to start...")
        
        # Loop until game is running or timeout
        wait_start_time = time.time()
        max_wait_time = 300  # 5 minutes timeout
        
        while time.time() - wait_start_time < max_wait_time:
            # Check if game window found
            if self.input_simulator.find_game_window():
                # Try to capture a frame
                frame = self.screen_capture.capture_frame()
                if frame is not None:
                    logger.info("Game started successfully!")
                    return
                    
            # Wait before checking again
            time.sleep(5.0)
            logger.info("Still waiting for game to start...")
            
        logger.warning("Timed out waiting for game to start")
        
    def _get_mock_frame(self) -> torch.Tensor:
        """Generate a mock frame for when the game is not running.
        
        Returns:
            torch.Tensor: Mock frame
        """
        # Create a special frame for crashed state - red for crashed, yellow for waiting
        height, width = self.screen_capture.process_resolution
        
        # Create a numpy array for the frame
        frame_np = np.zeros((height, width, 3), dtype=np.uint8)
        
        if self.game_crashed:
            # Red background for crash
            frame_np[:, :, 2] = 120  # Red channel (BGR in OpenCV)
            
            # Add text message about crash
            try:
                import cv2
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame_np, "GAME CRASHED", (width//6, height//3), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_np, "Please restart the game", (width//6, height//2), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_np, "Agent will wait automatically", (width//6, 2*height//3), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            except ImportError:
                logger.warning("OpenCV not available for adding text to mock frame")
            
        elif self.waiting_for_game_restart:
            # Yellow background for waiting
            frame_np[:, :, 0] = 100  # Blue channel 
            frame_np[:, :, 1] = 200  # Green channel
            frame_np[:, :, 2] = 200  # Red channel
            
            # Add text message about waiting
            try:
                import cv2
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame_np, "WAITING FOR GAME", (width//6, height//3), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_np, "Agent will resume automatically", (width//6, height//2), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_np, "when game is detected", (width//6, 2*height//3), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            except ImportError:
                logger.warning("OpenCV not available for adding text to mock frame")
        
        # Convert to tensor with shape [C, H, W]
        frame_tensor = torch.from_numpy(frame_np).float().permute(2, 0, 1) / 255.0
        
        # Move to the correct device
        if frame_tensor.device != self.config.get_device():
            frame_tensor = frame_tensor.to(self.config.get_device())
        
        return frame_tensor

    def _handle_menu_action(self, action_info: Dict) -> bool:
        """Handle actions specifically when in menu mode.
        
        This is a specialized action handler for menu navigation.
        
        Args:
            action_info (Dict): Dictionary containing action details
            
        Returns:
            bool: Whether the action was successfully executed
        """
        try:
            action_type = action_info.get("type", "")
            
            # Use menu handler if available
            if hasattr(self, 'menu_handler') and self.menu_handler is not None:
                # For mouse movement actions in menu
                if action_type == "mouse" and action_info.get("action") == "move":
                    position = action_info.get("position", (0.5, 0.5))
                    if isinstance(position, tuple) and len(position) == 2:
                        x, y = position
                        return self.menu_handler.navigate_to_menu_position(x, y)
                
                # For click actions in menu
                elif action_type == "mouse" and action_info.get("action") == "click":
                    # Get position
                    if "position" in action_info:
                        position = action_info.get("position")
                        if isinstance(position, tuple) and len(position) == 2:
                            x, y = position
                            return self.menu_handler.click_menu_position(x, y)
                
                # Handle key presses in menu
                elif action_type == "key":
                    key = action_info.get("key", "")
                    if key.lower() == "escape":
                        return self.menu_handler.exit_current_menu()
                    elif key.lower() in ["enter", "return"]:
                        return self.menu_handler.confirm_menu_selection()
            
            # Fallback to normal action execution
            return self._execute_action(action_info)
            
        except Exception as e:
            logger.error(f"Error handling menu action: {e}")
            return False