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
from collections import deque
from .visual_metrics import VisualMetricsEstimator
from .autonomous_reward_system import AutonomousRewardSystem
from .optimized_capture import OptimizedScreenCapture
from src.utils.performance_safeguards import PerformanceSafeguards
from .input_simulator import InputSimulator
from src.config.hardware_config import HardwareConfig
import win32api
from src.environment.screen_capture import ScreenCapture
from src.environment.visual_metrics import VisualMetrics
from src.environment.menu_handler import MenuHandler
from src.utils.image_utils import ImageUtils
from src.config.action_space import get_action_space

logger = logging.getLogger(__name__)

class CitiesEnvironment:
    """Environment for interacting with Cities: Skylines 2."""
    
    def __init__(self, config: Optional[HardwareConfig] = None, mock_mode: bool = False, menu_screenshot_path: Optional[str] = None):
        """Initialize the environment.
        
        Args:
            config: Optional hardware configuration
            mock_mode: If True, use a mock environment for testing/training without the actual game
            menu_screenshot_path: Path to a reference screenshot of the menu screen
        """
        self.config = config or HardwareConfig()
        self.mock_mode = mock_mode
        
        # Initialize components
        self.screen_capture = OptimizedScreenCapture(self.config)
        self.input_simulator = InputSimulator()
        # Connect input simulator to screen capture for coordinate translation
        self.input_simulator.screen_capture = self.screen_capture
        self.visual_estimator = VisualMetricsEstimator(self.config)
        
        # Initialize image utilities for visual processing
        self.image_utils = ImageUtils(debug_mode=False)
        
        # Initialize menu detection with reference image if provided
        self.menu_reference_path = menu_screenshot_path
        
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
        
        self.menu_detection_initialized = False
        
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
        self.max_steps = 1000
        self.last_action_time = time.time()
        self.min_action_delay = 0.1  # Minimum delay between actions
        
        # Game state tracking
        self.paused = False
        self.game_speed = 1
        
        # Performance tracking
        self.fps_history = []
        self.last_optimization_check = time.time()
        self.optimization_interval = 60  # Check optimization every 60 seconds
        
        # Counter for how long the agent has been stuck in a menu
        self.menu_stuck_counter = 0
        self.max_menu_penalty = -1000.0  # Set as per user instruction to -1000.0
        
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
        
        # Handle mock mode
        if self.mock_mode:
            logger.info("Using mock environment")
            # Create mock observation with correct channel/dimension order [C, H, W]
            self.current_frame = torch.rand(3, 180, 320, device=self.config.get_device())
            return self.current_frame
            
        # For real game mode, try to focus the game window
        logger.info("Focusing Cities: Skylines II window...")
        if not self.input_simulator.find_game_window():
            logger.warning("Cities: Skylines II window not found. Make sure the game is running.")
            logger.warning("Falling back to mock mode for environment testing...")
            self.mock_mode = True
            self.current_frame = torch.rand((3, 224, 224), device=self.config.get_device())
            return self.current_frame
        
        if not self.input_simulator.ensure_game_window_focused():
            logger.warning("Could not focus Cities: Skylines II window. Continuing anyway.")
            logger.warning("You may need to manually focus the game window for proper interaction.")
        
        # Capture initial frame
        initial_frame = self.screen_capture.capture_frame()
        
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
        
        while self.visual_estimator.detect_main_menu(initial_frame) and menu_attempts < max_menu_exit_attempts:
            menu_attempts += 1
            logger.info(f"Main menu detected during reset (attempt {menu_attempts}/{max_menu_exit_attempts}) - trying to exit")
            
            # For Cities Skylines II menu, use the exact coordinates for RESUME GAME button
            logger.info("Clicking RESUME GAME button with exact coordinates (720, 513)")
            
            # Get screen dimensions for scaling
            width, height = self._get_screen_dimensions()
            
            # Exact coordinates from user: (720, 513) for 1920x1080 resolution
            resume_x, resume_y = (720, 513)
            
            # Scale for different resolutions
            if width != 1920 or height != 1080:
                x_scale = width / 1920
                y_scale = height / 1080
                resume_x = int(resume_x * x_scale)
                resume_y = int(resume_y * y_scale)
            
            # Click the button with precision
            logger.info(f"Clicking at coordinates: ({resume_x}, {resume_y})")
            self.input_simulator.mouse_click(resume_x, resume_y)
            time.sleep(1.0)  # Longer delay for reliable click registration
            
            # Update frame to check if we're still in the menu
            initial_frame = self.screen_capture.capture_frame()
            
            # If still in menu, try alternative positions
            if self.visual_estimator.detect_main_menu(initial_frame) and menu_attempts < max_menu_exit_attempts:
                # Try the input simulator's safe menu handling method which has additional positions
                logger.info("Still in menu, using safe menu handling with multiple positions")
                self.input_simulator.safe_menu_handling()
                time.sleep(1.0)
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
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take a step in the environment.
        
        Args:
            action_idx (int): Index of the action to take
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: Next state, reward, done, info
        """
        # Increment step counter
        self.steps_taken += 1
        
        # Track for menu detection later
        self.last_action_idx = action_idx
        
        # Get action details from action space
        if action_idx < 0 or action_idx >= len(self.actions):
            logger.warning(f"Invalid action index: {action_idx}, using random action")
            action_idx = random.randint(0, len(self.actions) - 1)
            
        action_info = self.actions[action_idx]
        
        # Check if we are in a menu and need to avoid certain actions
        if not self.mock_mode and self.steps_taken % self.detect_menu_every_n_steps == 0:
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
        frame_before_action = self.screen_capture.get_latest_frame()
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
        """Check if game is currently in a menu state.
        
        Returns:
            bool: True if in menu, False otherwise
        """
        # Use menu handler if available
        if hasattr(self, 'menu_handler') and self.menu_handler:
            current_frame = self.screen_capture.get_latest_frame()
            if current_frame is not None:
                in_menu, menu_type, confidence = self.menu_handler.detect_menu(current_frame)
                if in_menu:
                    logger.debug(f"Menu detected: {menu_type} (confidence: {confidence:.2f})")
                return in_menu
                
        # Fallback to visual metrics menu detection
        return self.visual_estimator.detect_main_menu(self.screen_capture.get_latest_frame())
    
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
                    # Get normalized coordinates (0.0 to 1.0)
                    norm_x = action_info.get("x", 0.5)
                    norm_y = action_info.get("y", 0.5)
                    
                    # Convert to screen coordinates
                    width, height = self._get_screen_dimensions()
                    screen_x = int(norm_x * width)
                    screen_y = int(norm_y * height)
                    
                    # Execute move
                    return self.input_simulator.mouse_move(screen_x, screen_y)
                
                elif mouse_action == "click":
                    # Get button and double-click setting
                    button = action_info.get("button", "left")
                    double = action_info.get("double", False)
                    
                    # Use current mouse position for click
                    # This requires the agent to first move the mouse to the desired position
                    return self.input_simulator.mouse_click(button=button, double=double)
                
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
                    
                    # Execute scroll
                    return self.input_simulator.mouse_scroll(direction=direction, amount=amount)
                
                elif mouse_action == "drag":
                    # Get target normalized coordinates (0.0 to 1.0)
                    to_norm_x = action_info.get("to_x", 0.5)
                    to_norm_y = action_info.get("to_y", 0.5)
                    
                    # Convert to screen coordinates
                    width, height = self._get_screen_dimensions()
                    to_screen_x = int(to_norm_x * width)
                    to_screen_y = int(to_norm_y * height)
                    
                    # Get current mouse position as start position
                    try:
                        from_screen_x, from_screen_y = self.input_simulator.get_mouse_position()
                    except Exception:
                        # If can't get position, use center of screen
                        from_screen_x, from_screen_y = width // 2, height // 2
                    
                    # Execute drag
                    return self.input_simulator.mouse_drag(
                        from_screen_x, from_screen_y, 
                        to_screen_x, to_screen_y
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
                    success = self.input_simulator.mouse_click(button=button)
                
                # Release key
                time.sleep(0.05)  # Brief delay
                self.input_simulator.key_up(key)
                
                return success
            
            elif action_type == "wait":
                # Simply wait for specified duration
                duration = action_info.get("duration", 0.5)
                time.sleep(duration)
                return True
            
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
        """Get the dimensions of the game screen.
        
        Returns:
            Tuple[int, int]: Width and height of the screen
        """
        # Use screen capture client position if available
        if hasattr(self, 'screen_capture') and self.screen_capture:
            if hasattr(self.screen_capture, 'client_position') and self.screen_capture.client_position:
                client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
                return (client_right - client_left, client_bottom - client_top)
        
        # Try to use screen capture resolution
        if hasattr(self, 'screen_capture') and self.screen_capture:
            if hasattr(self.screen_capture, 'capture_resolution'):
                return self.screen_capture.capture_resolution
        
        # Fallback to system metrics
        try:
            return (win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1))
        except Exception:
            # Final fallback
            return (1920, 1080)

    def _compute_reward(self, action_info: Dict, action_idx: int, success: bool, menu_detected: bool) -> float:
        """Compute reward based purely on visual changes without domain knowledge.
        
        Args:
            action_info: Action details
            action_idx: Action index
            success: Whether action execution was successful
            menu_detected: Whether a menu was detected
            
        Returns:
            float: Computed reward
        """
        # Get frames before and after action
        prev_frame = getattr(self, 'prev_frame', None)
        curr_frame = self.screen_capture.get_latest_frame()
        
        # Store current frame for next reward computation
        self.prev_frame = curr_frame.copy() if curr_frame is not None else None
        
        # Use autonomous reward system for computation
        if prev_frame is not None and curr_frame is not None:
            reward = self.reward_system.compute_reward(prev_frame, action_idx, curr_frame)
        else:
            # If we don't have frames, return small positive reward for successful actions
            reward = 0.01 if success else -0.01
            
        return reward