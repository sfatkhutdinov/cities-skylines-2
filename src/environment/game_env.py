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
    
    def __init__(self, config: Optional[HardwareConfig] = None, mock_mode: bool = False, menu_screenshot_path: Optional[str] = None, disable_menu_detection: bool = False, **kwargs):
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
        self.menu_stuck_counter = 0
        self.disable_menu_detection = disable_menu_detection
        
        if self.disable_menu_detection:
            logger.info("Menu detection is disabled - menu detection and recovery will be skipped")
        
        # Set up components for interacting with the game
        if not mock_mode:
            # Initialize screen capture
            self.screen_capture = OptimizedScreenCapture(
                config=self.config
            )
            
            # Initialize input simulator
            self.input_simulator = InputSimulator()
            
            # Properly set screen_capture in input_simulator for action tracking
            self.input_simulator.screen_capture = self.screen_capture
            
            # Initialize visual metrics estimator
            self.visual_metrics = VisualMetricsEstimator(self.config)
            
            # Performance monitoring hooks into visual estimator
            if hasattr(self.visual_metrics, "visual_change_analyzer"):
                self.visual_change_analyzer = self.visual_metrics.visual_change_analyzer
            else:
                self.visual_change_analyzer = VisualChangeAnalyzer(self.config)
                self.visual_metrics.visual_change_analyzer = self.visual_change_analyzer
            
            # Load menu reference if provided
            self.has_menu_reference = False
            self.menu_reference_path = menu_screenshot_path
            self.menu_detection_initialized = False
            
            if menu_screenshot_path and os.path.exists(menu_screenshot_path):
                logger.info(f"Using menu reference image: {menu_screenshot_path}")
                self.has_menu_reference = True
                self.visual_metrics.initialize_menu_detection(menu_screenshot_path)
                self.menu_detection_initialized = True
            elif os.path.exists("menu_reference.png"):
                # Try to use a default reference if available
                self.menu_reference_path = os.path.abspath("menu_reference.png")
                logger.info(f"Using default menu reference image found at {self.menu_reference_path}")
                self.has_menu_reference = True
                self.visual_metrics.initialize_menu_detection(self.menu_reference_path)
                self.menu_detection_initialized = True
            else:
                # No menu reference image, set up fallback menu detection
                logger.info("No menu reference image available, setting up fallback menu detection")
                self.visual_metrics.setup_fallback_menu_detection()
            
            # Use autonomous reward system
            self.reward_system = AutonomousRewardSystem(self.config)
            self.safeguards = PerformanceSafeguards(self.config)
            
            # Initialize menu handler for more advanced menu detection and recovery
            self.menu_handler = MenuHandler(
                screen_capture=self.screen_capture,
                input_simulator=self.input_simulator,
                visual_metrics=self.visual_metrics
            )
            
            # Set menu_handler reference in screen_capture for action tracking
            self.screen_capture.menu_handler = self.menu_handler
            
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
            self.max_menu_stuck_steps = 5
            self.likely_in_menu_from_mouse_test = False
            self.mouse_freedom_visual_change = 1.0  # Initialize with high value (indicates not in menu)
            self.last_mouse_freedom_test_time = 0
            self.previous_menu_state = False
            self.menu_entry_time = 0
            self.last_menu_exit_time = 0
            self.menu_entry_count = 0
            
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
        success = self.visual_metrics.save_current_frame_as_menu_reference(frame, save_path)
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
        last_menu_type = None
        consecutive_same_menu = 0
        
        while self.check_menu_state() and menu_attempts < max_menu_exit_attempts:
            # Get current menu info
            menu_detected, menu_type, _ = self.menu_handler.detect_menu(self.screen_capture.capture_frame())
            
            # Track if we're seeing the same menu repeatedly
            if menu_type == last_menu_type:
                consecutive_same_menu += 1
            else:
                consecutive_same_menu = 0
            last_menu_type = menu_type
                
            logger.info(f"Detected menu during reset, attempting to exit (attempt {menu_attempts+1})")
            
            # If we've tried the same menu 3 times, try a different approach
            if consecutive_same_menu >= 2:
                logger.warning(f"Same menu type {menu_type} detected multiple times, trying alternative exit method")
                # Click in different locations based on menu type
                width, height = self._get_screen_dimensions()
                
                if menu_type == "main_menu":
                    # For main menu, try clicking "Resume Game" near top
                    resume_x = int(width * 0.5)
                    resume_y = int(height * 0.35)
                    self.input_simulator.mouse_click(resume_x, resume_y)
                    time.sleep(0.5)
                    
                    # Also try "Continue" button
                    resume_x = int(width * 0.5)
                    resume_y = int(height * 0.45)
                    self.input_simulator.mouse_click(resume_x, resume_y)
                elif menu_type == "settings_menu":
                    # For settings menu, try clicking back/close button (top-right)
                    close_x = int(width * 0.95)
                    close_y = int(height * 0.05)
                    self.input_simulator.mouse_click(close_x, close_y)
                    time.sleep(0.5)
                    
                    # Also try back button (bottom)
                    back_x = int(width * 0.15)
                    back_y = int(height * 0.9)
                    self.input_simulator.mouse_click(back_x, back_y)
                else:
                    # Generic approach - try pressing ESC and clicking center-bottom
                    self.input_simulator.key_press('escape')
                    time.sleep(0.5)
                    self.input_simulator.mouse_click(width // 2, int(height * 0.8))
            else:
                # Standard approach - try to exit menu with ESC key
                self.input_simulator.key_press('escape')
                time.sleep(0.5)
                self.input_simulator.key_press('enter')  # Sometimes helps confirm dialogs
                time.sleep(0.2)
                
                # Try clicking resume button (vary location slightly each time)
                width, height = self._get_screen_dimensions()
                variation = menu_attempts * 0.05  # Vary by 5% each attempt
                resume_x = int(width * (0.15 + variation))
                resume_y = int(height * (0.25 + variation))
                self.input_simulator.mouse_click(resume_x, resume_y)
                time.sleep(0.5)
                
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
        """Execute an action in the environment.
        
        Args:
            action_idx: Action index to execute
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: state, reward, done, info
        """
        # First check for menus before taking action
        self._check_for_menus()
        
        # Get action type
        action_type = self.actions[action_idx].get("type", "")
        
        # Initialize return values
        reward = 0.0
        done = False
        info = {}
        
        # Track action in menu handler (helps with menu detection)
        # Certain actions like ESC can trigger menus
        if action_type == "key" and self.actions[action_idx].get("key") == "escape":
            self.menu_handler.track_action("esc_key")
        elif action_type == "mouse" and self.actions[action_idx].get("action") == "click":
            # Check if click is near settings/gear icon
            mouse_x, mouse_y = self.input_simulator.get_mouse_position()
            gear_x, gear_y = self.menu_handler.GEAR_ICON_POSITION
            screen_w, screen_h = self._get_screen_dimensions()
            gear_x_px, gear_y_px = int(gear_x * screen_w), int(gear_y * screen_h)
            
            # If click is within 50px of gear icon
            if ((mouse_x - gear_x_px)**2 + (mouse_y - gear_y_px)**2) < 2500:  # 50px radius
                self.menu_handler.track_action("gear_click")
        
        # Execute action
        if action_type == "mouse":
            target_x, target_y = self.actions[action_idx].get("position", (0.5, 0.5))
            if isinstance(target_x, tuple) and isinstance(target_y, tuple):
                x, y = target_x
            else:
                # Fallback to normalized coordinates (0.0 to 1.0)
                x = target_x if isinstance(target_x, (int, float)) else 0.5
                y = target_y if isinstance(target_y, (int, float)) else 0.5
            
            logger.debug(f"Moving mouse to: ({x}, {y})")
            self.input_simulator.mouse_move(x, y)
            
        elif action_type == "mouse_click":
            button = self.actions[action_idx].get("button", "left")
            double = self.actions[action_idx].get("double", False)
            self.input_simulator.mouse_click(x, y, button=button, double=double)
            
        elif action_type == "key":
            key = self.actions[action_idx].get("key", "")
            if key:
                self.input_simulator.key_press(key)
                
        elif action_type == "speed":
            speed_value = self.actions[action_idx].get("speed", 1.0)
            speed_level = 0 if speed_value <= 0.1 else 1 if speed_value <= 0.3 else 2 if speed_value <= 0.6 else 3 if speed_value <= 0.8 else 4
            self._set_game_speed(speed_level)
            
        # Sleep briefly to allow action to take effect
        time.sleep(0.05)
        
        # Get the current state
        state = self._process_frame(self.screen_capture.capture_frame())
        
        # Calculate reward using autonomous reward system
        reward = self._compute_reward(self.actions[action_idx], action_idx, True, self.check_menu_state())
        
        # Add penalty if in a menu and not attempting to exit
        # Important: Always check for menus after executing an action
        is_in_menu = self.check_menu_state()
        if is_in_menu:
            # We're in a menu - apply menu penalty and try to recover 
            # unless the action was an attempt to exit (like ESC or clicking exit buttons)
            is_exit_attempt = (action_type == "key" and self.actions[action_idx].get("key") == "escape") or \
                              (action_type == "mouse" and self.actions[action_idx].get("action") == "click")  # Any click could be trying to exit
                              
            if not is_exit_attempt:
                # Apply a penalty for not trying to exit the menu
                reward -= 0.5
                info["menu_penalty"] = True
                
            # Try to recover from menu
            self._handle_menu_recovery()
            
            # Get fresh state after recovery attempt
            state = self._process_frame(self.screen_capture.capture_frame())
        
        # Update the step counter
        self.steps_taken += 1
        
        # Check if the episode is done
        done = self.steps_taken >= self.max_steps
        
        # Add diagnostic information
        info["step"] = self.steps_taken
        info["action_type"] = action_type
        info["in_menu"] = is_in_menu
        
        return state, reward, done, info
    
    def check_menu_state(self) -> bool:
        """Check if the game is currently in a menu state.
        
        Returns:
            bool: True if a menu is detected, False otherwise
        """
        # If menu detection is disabled, always return False
        if hasattr(self, 'disable_menu_detection') and self.disable_menu_detection:
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
            
        # First check for version string (most reliable indicator)
        if hasattr(self, 'visual_metrics') and hasattr(self.visual_metrics, 'detect_version_string'):
            try:
                # Convert frame to numpy
                frame_np = current_frame.cpu().numpy() if hasattr(current_frame, 'cpu') else current_frame
                if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:  # CHW format
                    frame_np = frame_np.transpose(1, 2, 0)  # Convert to HWC
                
                # Ensure frame is in uint8 format for OpenCV operations
                if frame_np.dtype != np.uint8:
                    if frame_np.max() <= 1.0:
                        frame_np = (frame_np * 255).astype(np.uint8)
                    else:
                        frame_np = frame_np.astype(np.uint8)
                    
                version_string_detected = self.visual_metrics.detect_version_string(frame_np)
                if version_string_detected:
                    logger.info("Menu detected via version string - high confidence")
                    return True
            except Exception as e:
                logger.error(f"Error in version string detection: {e}")
        
        # If we've been having trouble with menu detection, run a mouse freedom test
        # This is only done rarely to avoid disrupting gameplay
        menu_uncertain = (hasattr(self, 'menu_stuck_counter') and 
                         self.menu_stuck_counter > 0 and 
                         self.menu_stuck_counter % 3 == 0)  # Every 3rd uncertain detection
                         
        if menu_uncertain and not hasattr(self, 'last_mouse_freedom_test_time'):
            self.last_mouse_freedom_test_time = 0
            
        current_time = time.time()
        if (menu_uncertain and 
            hasattr(self, 'last_mouse_freedom_test_time') and 
            current_time - self.last_mouse_freedom_test_time > 30):  # At most once every 30 seconds
            logger.info("Running mouse freedom test to help with menu detection")
            self.last_mouse_freedom_test_time = current_time
            self.test_mouse_freedom()
            # The test_mouse_freedom method will update self.likely_in_menu_from_mouse_test
        
        # Use menu handler if available
        if hasattr(self, 'menu_handler') and self.menu_handler is not None:
            menu_detected, menu_type, confidence = self.menu_handler.detect_menu(current_frame)
            if menu_detected:
                logger.info(f"Menu detected: {menu_type} with confidence {confidence:.2f}")
            return menu_detected
            
        # Fallback to visual metrics
        try:
            return self.visual_metrics.detect_main_menu(current_frame)
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
    
    def _check_for_menus(self) -> bool:
        """Force check for menus without waiting for cooldown.
        
        Returns:
            bool: True if menu was detected, False otherwise
        """
        # Skip menu detection if disabled
        if hasattr(self, 'disable_menu_detection') and self.disable_menu_detection:
            return False
            
        # Force menu detection to run (bypassing cooldown)
        # This is needed to detect manually opened menus more quickly
        try:
            current_frame = self.screen_capture.capture_frame()
            if current_frame is None:
                return False
            
            # First try direct version string detection as it's most reliable
            if hasattr(self, 'visual_metrics') and hasattr(self.visual_metrics, 'detect_version_string'):
                try:
                    # Convert frame to numpy for version string detection
                    frame_np = current_frame.cpu().numpy() if hasattr(current_frame, 'cpu') else current_frame
                    if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:  # CHW format
                        frame_np = frame_np.transpose(1, 2, 0)  # Convert to HWC
                    
                    # Ensure frame is in uint8 format for OpenCV operations
                    if frame_np.dtype != np.uint8:
                        if frame_np.max() <= 1.0:
                            frame_np = (frame_np * 255).astype(np.uint8)
                        else:
                            frame_np = frame_np.astype(np.uint8)
                    
                    version_string_detected = self.visual_metrics.detect_version_string(frame_np)
                    if version_string_detected:
                        logger.info("Menu detected via version string - high confidence indicator")
                        return True
                except Exception as e:
                    logger.error(f"Error in version string detection: {e}")
            
            # Access the menu handler's detect_menu method directly
            # To bypass cooldown, we need to temporarily save and restore the check time
            if self.menu_handler:
                last_check_time = self.menu_handler.last_menu_check_time
                # Set to a time far in the past to force check
                self.menu_handler.last_menu_check_time = 0
                
                in_menu, menu_type, confidence = self.menu_handler.detect_menu(current_frame)
                
                # Restore the original check time if no menu was found
                if not in_menu:
                    self.menu_handler.last_menu_check_time = last_check_time
                    
                if in_menu:
                    logger.info(f"Detected {menu_type} menu (confidence: {confidence:.2f})")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking for menus: {e}")
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
        Also measures visual change to detect if in a menu.
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
        
        # Capture frame before moving mouse
        initial_frame = self.screen_capture.capture_frame()
        visual_changes = []
        
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
                
                # Capture frame after moving mouse to measure visual change
                new_frame = self.screen_capture.capture_frame()
                if initial_frame is not None and new_frame is not None:
                    # Calculate visual change
                    try:
                        visual_change = self.visual_metrics.calculate_frame_difference(initial_frame, new_frame)
                        visual_changes.append(visual_change)
                        logger.debug(f"Visual change after mouse move to ({x},{y}): {visual_change:.6f}")
                    except Exception as e:
                        logger.error(f"Error calculating visual change: {e}")
                
                # Verify position
                new_pos = win32api.GetCursorPos()
                if abs(new_pos[0] - x) > 5 or abs(new_pos[1] - y) > 5:
                    logger.warning(f"Mouse position verification failed: Expected ({x},{y}), got ({new_pos[0]},{new_pos[1]})")
                    # Try again with direct positioning
                    win32api.SetCursorPos((x, y))
                    time.sleep(0.2)
            
            # Calculate average visual change from mouse movement
            if visual_changes:
                avg_visual_change = sum(visual_changes) / len(visual_changes)
                max_visual_change = max(visual_changes)
                logger.info(f"Mouse movement visual change - Avg: {avg_visual_change:.6f}, Max: {max_visual_change:.6f}")
                
                # If visual change is very low, likely in a menu
                if max_visual_change < 0.01:
                    logger.warning("Very low visual change detected during mouse movement, likely in a menu")
                    # Store this information for the menu detection system
                    self.mouse_freedom_visual_change = max_visual_change
                    self.likely_in_menu_from_mouse_test = True
                else:
                    logger.info("Normal visual change detected during mouse movement, likely not in a menu")
                    self.mouse_freedom_visual_change = max_visual_change
                    self.likely_in_menu_from_mouse_test = False
            
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
            
            # Store the timestamp when we first detected the menu
            if not hasattr(self, 'menu_entry_time') or not hasattr(self, 'previous_menu_state') or not self.previous_menu_state:
                self.menu_entry_time = time.time()
        
        # Add a reward for successfully exiting a menu (but prevent farming)
        if hasattr(self, 'previous_menu_state') and self.previous_menu_state and not menu_detected:
            # Successfully exited a menu
            # Calculate how long we were in the menu
            current_time = time.time()
            menu_duration = current_time - getattr(self, 'menu_entry_time', current_time)
            
            # Only give exit reward if we were in the menu for a reasonable time
            # This prevents the agent from rapidly toggling the menu for rewards
            if menu_duration > 1.0:  # Must be in menu for at least 1 second
                # Base reward for exiting menu
                exit_reward = 0.5
                
                # Scale reward based on how quickly the agent exited (encourage faster exits)
                # Cap at reasonable maximum time to not overly punish long menu stays
                time_factor = min(menu_duration, 10.0) / 10.0  # Normalize to 0-1
                # Inverted - faster exits (lower time_factor) get higher rewards
                time_bonus = (1.0 - time_factor) * 0.5
                
                # Add the exit reward and time bonus
                reward += exit_reward + time_bonus
                
                logger.debug(f"Menu exit reward: +{exit_reward + time_bonus:.2f} after {menu_duration:.1f}s in menu")
                
                # Reset menu cooldown timer - this prevents immediate re-entry farming
                self.last_menu_exit_time = current_time
                self.menu_entry_count = getattr(self, 'menu_entry_count', 0) + 1
        
        # Extra penalty for repeatedly entering/exiting menus (anti-farming)
        if menu_detected and not getattr(self, 'previous_menu_state', False):
            # Just entered a menu, check if this is too soon after exiting one
            current_time = time.time()
            time_since_last_exit = current_time - getattr(self, 'last_menu_exit_time', 0)
            menu_entry_count = getattr(self, 'menu_entry_count', 0)
            
            # Penalty for entering menu too soon after exiting
            if time_since_last_exit < 5.0 and menu_entry_count > 0:
                # Calculate penalty that increases with each rapid re-entry
                farming_penalty = min(0.5 * menu_entry_count, 3.0)
                reward -= farming_penalty
                logger.debug(f"Menu farming penalty: -{farming_penalty:.2f} (re-entered after {time_since_last_exit:.1f}s)")
        
        # Update menu state tracking
        self.previous_menu_state = menu_detected
        
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