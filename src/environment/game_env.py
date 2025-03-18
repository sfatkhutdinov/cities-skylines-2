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
        self._reset_mouse_position()
        
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
    
    def _reset_mouse_position(self):
        """Reset mouse to a neutral position to ensure free movement."""
        # Move mouse to center of screen to ensure free movement
        width, height = self._get_screen_dimensions()
        center_x, center_y = width // 2, height // 2
        
        logger.info(f"Resetting mouse to center position ({center_x}, {center_y})")
        self.input_simulator.mouse_move(center_x, center_y)
        time.sleep(0.5)  # Short delay to ensure mouse movement completes
    
    def _ensure_game_running(self):
        """Make sure game is not paused. Press space if it is."""
        # Use space to unpause the game without using ESC
        # Space toggles pause/unpause
        self.input_simulator.press_key('space')
        time.sleep(0.2)
        # Press space again to ensure we're in the correct state if needed
        # This helps clear some menus without using ESC
        self.input_simulator.press_key('space')
        time.sleep(0.2)
        logger.info("Game state reset - ensured game is running")
        
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
    
    def step(self, action_index: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """Take an action in the environment.
        
        Args:
            action_index: Index of the action to take
            
        Returns:
            observation: The next observation (frame)
            reward: The reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        # Store previous state for activity calculation
        if hasattr(self, 'last_observation') and self.last_observation is not None:
            prev_state = self.last_observation
            
        # Check if we need to delay before taking another action
        elapsed = time.time() - self.last_action_time
        if elapsed < self.min_action_delay:
            time.sleep(self.min_action_delay - elapsed)
            
        # In mock mode, just increment counter and return mock
        if self.mock_mode:
            # Simulate a short delay
            time.sleep(0.01)
            
            # Update counters
            self.steps_taken += 1
            self.episode_step += 1
            self.last_action_time = time.time()
            
            # Return mock observation and reward
            reward = np.random.normal(0.0, 0.1)  # Random small reward
            done = self.steps_taken >= self.max_steps
            self.current_frame = torch.randn((3, 240, 320), device=self.config.get_device())
            return self.current_frame, reward, done, {"mock": True}
            
        # Focus game window if needed
        self.input_simulator.ensure_game_window_focused()
        
        # Determine frame skip based on activity level
        if self.adaptive_frame_skip:
            # Calculate appropriate frame skip - lower when active, higher when passive
            target_skip = int(self.min_frame_skip + (self.max_frame_skip - self.min_frame_skip) * (1.0 - self.activity_level))
            self.current_frame_skip = max(self.min_frame_skip, min(target_skip, self.max_frame_skip))
        else:
            self.current_frame_skip = getattr(self.config, 'frame_skip', 2)
            
        # Log frame skip if it changed
        if hasattr(self, 'last_frame_skip') and self.current_frame_skip != self.last_frame_skip:
            logger.debug(f"Frame skip adjusted to {self.current_frame_skip} (activity: {self.activity_level:.2f})")
        self.last_frame_skip = self.current_frame_skip
        
        # First check if we're in a menu before executing action
        pre_action_frame = self.screen_capture.capture_frame()
        if self.visual_estimator.detect_main_menu(pre_action_frame):
            logger.info("Menu detected before action - attempting to exit menu first")
            
            # Try pressing escape once to exit the menu
            logger.info("Trying single Escape key press to exit menu")
            self.input_simulator.block_escape = False  # Temporarily allow Escape
            self.input_simulator.key_press('escape')
            self.input_simulator.block_escape = True   # Re-block Escape
            time.sleep(0.7)
            
            # Check if we're still in a menu
            if self.visual_estimator.detect_main_menu(self.screen_capture.capture_frame()):
                logger.info("Still in menu after Escape key, trying to click 'RESUME GAME' button")
                
                # Now try clicking Resume Game button
                width, height = self._get_screen_dimensions()
                
                # Try several potential positions for the resume game button, starting with exact coordinates from screenshot
                resume_positions = [
                    (int(width * 0.45), int(height * 0.387)),   # Exact position from screenshot
                    (int(width * 0.452), int(height * 0.39)),   # Slightly adjusted
                    (450, 387),                                 # Absolute coordinates from screenshot
                    (int(width * 0.288), int(height * 0.27)),   # Previous position from logs
                    (int(width * 0.5), int(height * 0.5))       # Center of screen
                ]
                
                # Try each position
                for resume_x, resume_y in resume_positions:
                    self.input_simulator.mouse_click(resume_x, resume_y)
                    time.sleep(0.7)
                    
                    # Check if we're out of the menu
                    if not self.visual_estimator.detect_main_menu(self.screen_capture.capture_frame()):
                        break
        
        # Get action details
        if action_index < 0 or action_index >= len(self.actions):
            logger.warning(f"Invalid action index: {action_index}")
            action_index = 0  # Default to first action
        
        action_info = self.actions[action_index]
        
        # Check if we should suppress menu-opening actions
        if self.suppress_menu_actions and action_info.get("type") == "menu" and action_info.get("action") == "open_pause_menu":
            logger.info("Suppressing menu-opening action due to recent menu exit")
            # Replace with a harmless action like waiting
            harmless_actions = [i for i, a in enumerate(self.actions) if a.get("type") == "time" and a.get("action") == "wait_short"]
            if harmless_actions:
                action_index = harmless_actions[0]
                action_info = self.actions[action_index]
                logger.info(f"Replaced with action: {action_info}")
        
        # Execute the action
        self._execute_action(action_info)
        self.last_action_time = time.time()
        
        # Apply frame skip based on the adaptive calculation
        for _ in range(self.current_frame_skip - 1):
            time.sleep(0.01)  # Small sleep to let game process any changes
        
        # Get new state
        next_state = self.get_observation()
        self.last_observation = next_state
        
        # Increment step counter
        self.steps_taken += 1
        
        # Wait for a short period to let the game update
        # This delay is calibrated based on the game's response time
        time.sleep(0.2)
        
        # Capture new frame after action
        self.current_frame = self.screen_capture.capture_frame()
        
        # Check if we ended up in a menu after the action (in case the action triggered a menu)
        if self.visual_estimator.detect_main_menu(self.current_frame):
            logger.info("Menu detected after action - trying to exit menu")
            
            # Try pressing escape once to exit the menu
            logger.info("Trying single Escape key press to exit menu")
            self.input_simulator.block_escape = False  # Temporarily allow Escape
            self.input_simulator.key_press('escape')
            self.input_simulator.block_escape = True   # Re-block Escape
            time.sleep(0.7)
            
            # Update the current frame to see if we're still in the menu
            self.current_frame = self.screen_capture.capture_frame()
            
            # If still in menu, try clicking buttons
            if self.visual_estimator.detect_main_menu(self.current_frame):
                logger.info("Still in menu after Escape key, trying to click 'RESUME GAME' button")
                menu_exit_attempts = 0
                max_exit_attempts = 3
                
                while self.visual_estimator.detect_main_menu(self.current_frame) and menu_exit_attempts < max_exit_attempts:
                    menu_exit_attempts += 1
                    logger.info(f"Attempting to click menu buttons (attempt {menu_exit_attempts}/{max_exit_attempts})")
                    
                    # Try clicking Resume Game button at different positions
                    width, height = self._get_screen_dimensions()
                    
                    # Try several potential positions for the resume game button
                    resume_positions = [
                        (int(width * 0.45), int(height * 0.387)),   # Exact position from screenshot
                        (int(width * 0.452), int(height * 0.39)),   # Slightly adjusted
                        (450, 387),                                 # Absolute coordinates from screenshot
                        (int(width * 0.288), int(height * 0.27)),   # Previous position from logs
                        (int(width * 0.5), int(height * 0.5))       # Center of screen
                    ]
                    
                    # Try the position corresponding to the current attempt
                    position_index = min(menu_exit_attempts - 1, len(resume_positions) - 1)
                    resume_x, resume_y = resume_positions[position_index]
                    self.input_simulator.mouse_click(resume_x, resume_y)
                    time.sleep(0.7)
                    
                    # Update the current frame to see if we're still in the menu
                    self.current_frame = self.screen_capture.capture_frame()
        
        # Calculate reward using autonomous reward system
        reward = self.reward_system.compute_reward(self.current_frame, action_index)
        
        # Apply menu penalty if a menu is detected
        menu_detected = self.visual_estimator.detect_main_menu(self.current_frame)
        menu_penalty = 0.0
        
        if menu_detected:
            # Increment the counter for consecutive menu steps
            self.menu_stuck_counter += 1
            
            # Calculate escalating penalty based on how long the agent has been stuck
            menu_penalty = self.reward_system.calculate_menu_penalty(menu_detected, self.menu_stuck_counter)
            
            # Apply penalty but don't let it exceed the maximum defined penalty
            menu_penalty = max(menu_penalty, self.max_menu_penalty)
            
            reward += menu_penalty
            logger.info(f"Applied menu penalty: {menu_penalty} (consecutive menu steps: {self.menu_stuck_counter}, total reward: {reward})")
            
            # Try to exit the menu immediately to prevent getting stuck
            if self.menu_stuck_counter > 1:
                logger.info("Stuck in menu for multiple steps - attempting to exit")
                
                # Click directly on the RESUME GAME button with exact coordinates
                width, height = self._get_screen_dimensions()
                
                # Exact coordinates from user: (720, 513) for 1920x1080 resolution
                resume_x, resume_y = (720, 513)
                
                # Scale for different resolutions
                if width != 1920 or height != 1080:
                    x_scale = width / 1920
                    y_scale = height / 1080
                    resume_x = int(resume_x * x_scale)
                    resume_y = int(resume_y * y_scale)
                    
                logger.info(f"Clicking RESUME GAME at coordinates: ({resume_x}, {resume_y})")
                self.input_simulator.mouse_click(resume_x, resume_y)
                time.sleep(1.0)
                
                # If still in menu, try the safe handling method
                if self.visual_estimator.detect_main_menu(self.screen_capture.capture_frame()):
                    self.input_simulator.safe_menu_handling()
                
                # Reset mouse position after trying to exit menu
                self._reset_mouse_position()
        else:
            # If we just exited a menu, handle transition properly
            if self.menu_stuck_counter > 0:
                logger.info("Successfully exited menu - restoring normal gameplay")
                # Reset mouse to allow free movement
                self._reset_mouse_position()
                
                # Enable menu action suppression for a while
                self.suppress_menu_actions = True
                self.menu_action_suppression_steps = self.max_menu_suppression_steps
                logger.info(f"Enabling menu action suppression for {self.max_menu_suppression_steps} steps")
            
            # Reset the menu counter when not in menu
            self.menu_stuck_counter = 0
            
            # Decrement suppression counter if active
            if self.suppress_menu_actions:
                self.menu_action_suppression_steps -= 1
                if self.menu_action_suppression_steps <= 0:
                    self.suppress_menu_actions = False
                    logger.info("Menu action suppression ended")
        
        # Check if episode should terminate (using step limit for now)
        done = self.steps_taken >= self.max_steps
        
        # Calculate activity level based on pixel differences
        if hasattr(self, 'last_observation') and self.last_observation is not None and 'prev_state' in locals():
            with torch.no_grad():
                prev_frame = prev_state
                curr_frame = self.current_frame
                
                # Calculate frame difference and convert to activity measure
                if prev_frame.device != torch.device('cpu'):
                    prev_frame = prev_frame.cpu()
                if curr_frame.device != torch.device('cpu'):
                    curr_frame = curr_frame.cpu()
                    
                # Calculate difference and normalize
                diff = torch.abs(curr_frame - prev_frame).mean().item()
                
                # Higher threshold for meaningful activity
                activity = min(1.0, diff * 20.0)  # Scale for detection sensitivity
                
                # Update activity history
                self.activity_history.append(activity)
                if len(self.activity_history) > self.activity_history_max_len:
                    self.activity_history.pop(0)
                    
                # Calculate smoothed activity level
                self.activity_level = sum(self.activity_history) / len(self.activity_history)
                
        # Prepare info dict
        info = {
            'steps': self.steps_taken,
            'action_type': action_info.get("type", "unknown"),
            'action': action_info.get("action", "unknown"),
            'menu_detected': self.visual_estimator.detect_main_menu(self.current_frame),
            'menu_penalty': menu_penalty if menu_detected else 0.0,
            'menu_stuck_counter': self.menu_stuck_counter,
            'activity_level': self.activity_level
        }
        
        return self.current_frame, reward, done, info
    
    def _execute_action(self, action_info: Dict):
        """Execute a selected action based on its type.
        
        Args:
            action_info (Dict): Action information dictionary
        """
        action_type = action_info.get("type", "")
        
        # Handle speed control actions
        if action_type == "speed":
            speed = action_info["speed"]
            self.input_simulator.set_movement_speed(speed)
            logger.info(f"Set movement speed to {speed}")
            return
        
        # Handle key press actions    
        if action_type == "key":
            key = action_info.get("key", "")
            duration = action_info.get("duration", 0.1)
            if key:
                self.input_simulator.key_press(key, duration)
                logger.debug(f"Pressed key: {key} for {duration}s")
            return
            
        # Handle mouse actions
        if action_type == "mouse":
            action = action_info.get("action", "")
            button = action_info.get("button", "left")
            
            # Get the client area center coordinates for the mouse action
            width, height = self._get_screen_dimensions()
            center_x, center_y = width // 2, height // 2
            
            # Get position from action_info - try both "position" tuple and separate "x"/"y" coordinates
            position = action_info.get("position", None)
            if position:
                x, y = position
            else:
                # If no position tuple, try to get separate x and y coordinates
                x = action_info.get("x", None)
                y = action_info.get("y", None)
                
                # If no x/y coordinates provided, generate random positions rather than using center
                if x is None or y is None:
                    # Use random positions within the main game area (avoiding edges)
                    margin = 100  # pixels from edge
                    x = random.randint(margin, width - margin)
                    y = random.randint(margin, height - margin)
                    logger.debug(f"Generated random position: ({x}, {y})")
                    
            if action == "move":
                self.input_simulator.mouse_move(x, y)
                logger.debug(f"Mouse moved to ({x}, {y})")
            elif action == "click":
                self.input_simulator.mouse_click(x, y, button=button)
                logger.debug(f"Mouse {button} click at ({x}, {y})")
            elif action == "double_click":
                self.input_simulator.mouse_click(x, y, button=button, double=True)
                logger.debug(f"Mouse {button} double click at ({x}, {y})")
            elif action == "scroll":
                direction = action_info.get("direction", 0)
                # Move mouse to position first, then scroll
                self.input_simulator.mouse_move(x, y)
                self.input_simulator.mouse_scroll(direction)
                logger.debug(f"Mouse scroll: {direction} at ({x}, {y})")
            elif action == "drag":
                # For drag, we need start and end positions
                end_x = action_info.get("end_x", x + random.randint(-100, 100))
                end_y = action_info.get("end_y", y + random.randint(-100, 100))
                self.input_simulator.mouse_drag(x, y, end_x, end_y)
                logger.debug(f"Mouse drag from ({x}, {y}) to ({end_x}, {end_y})")
            return
            
        # For other action types, extract the action
        action = action_info.get("action", "")
        if not action:
            logger.warning(f"No action specified for action type: {action_type}")
            return
        
        if action_type == "camera":
            self._handle_camera_action(action)
        elif action_type == "build":
            self._handle_build_action(action)
        elif action_type == "service":
            self._handle_service_action(action)
        elif action_type == "tool":
            self._handle_tool_action(action)
        elif action_type == "ui":
            self._handle_ui_action(action)
        elif action_type == "ui_position":
            if action == "drag" and "start" in action_info and "end" in action_info:
                # Handle drag between two positions
                start = action_info["start"]
                end = action_info["end"]
                duration = action_info.get("duration", None)  # Use provided duration or default (None = auto-calculate)
                self.input_simulator.mouse_drag(start[0], start[1], end[0], end[1], duration)
            else:
                # Handle single position actions (click, right-click, etc.)
                position = action_info.get("position", (0, 0))
                self._handle_ui_position_action(action, position)
    
    def _handle_camera_action(self, action: str):
        """Execute camera movement actions using default game key bindings."""
        # Get screen dimensions
        width, height = self._get_screen_dimensions()
        # Calculate center and margins
        center_x, center_y = width // 2, height // 2
        margin = 100  # Pixels from edge to avoid
        
        # Random offsets for more varied camera movements
        rand_x = random.randint(-100, 100)
        rand_y = random.randint(-100, 100)
        
        # Make center slightly randomized to prevent getting stuck in patterns
        center_x += rand_x
        center_y += rand_y
        
        # Ensure center stays within safe bounds
        center_x = max(margin, min(width - margin, center_x))
        center_y = max(margin, min(height - margin, center_y))
        
        # Handle basic movement with keys according to default keybindings
        if action == "move_up":
            self.input_simulator.key_press('w', duration=0.1)
        elif action == "move_down":
            self.input_simulator.key_press('s', duration=0.1)
        elif action == "move_left":
            self.input_simulator.key_press('a', duration=0.1)
        elif action == "move_right":
            self.input_simulator.key_press('d', duration=0.1)
        # Enhanced diagonal movements
        elif action == "move_up_left":
            self.input_simulator.press_key('w')
            self.input_simulator.press_key('a')
            time.sleep(0.1)
            self.input_simulator.release_key('w')
            self.input_simulator.release_key('a')
        elif action == "move_up_right":
            self.input_simulator.press_key('w')
            self.input_simulator.press_key('d')
            time.sleep(0.1)
            self.input_simulator.release_key('w')
            self.input_simulator.release_key('d')
        elif action == "move_down_left":
            self.input_simulator.press_key('s')
            self.input_simulator.press_key('a')
            time.sleep(0.1)
            self.input_simulator.release_key('s')
            self.input_simulator.release_key('a')
        elif action == "move_down_right":
            self.input_simulator.press_key('s')
            self.input_simulator.press_key('d')
            time.sleep(0.1)
            self.input_simulator.release_key('s')
            self.input_simulator.release_key('d')
        # Fast movements - hold keys longer
        elif action == "move_up_fast":
            self.input_simulator.key_press('w', duration=0.3)
        elif action == "move_down_fast":
            self.input_simulator.key_press('s', duration=0.3)
        # Zoom with R/F per default bindings
        elif action == "zoom_in":
            self.input_simulator.key_press('r', duration=0.1)
        elif action == "zoom_out":
            self.input_simulator.key_press('f', duration=0.1)
        # Rotation with Q/E per default bindings
        elif action == "rotate_left":
            # Mix key press with mouse rotation for more varied movement
            self.input_simulator.key_press('q', duration=0.1)
            # Also try mouse-based rotation for variety
            if random.random() > 0.5:  # 50% chance to use mouse rotation
                self.input_simulator.rotate_camera(
                    center_x, center_y, 
                    center_x - 200, center_y
                )
        elif action == "rotate_right":
            self.input_simulator.key_press('e', duration=0.1)
            # Also try mouse-based rotation for variety
            if random.random() > 0.5:  # 50% chance to use mouse rotation
                self.input_simulator.rotate_camera(
                    center_x, center_y, 
                    center_x + 200, center_y
                )
        # Orbital camera movements (combined rotation and movement)
        elif action == "orbit_left":
            # Press rotate left and move left
            self.input_simulator.press_key('q')
            time.sleep(0.05)
            self.input_simulator.press_key('a')
            time.sleep(0.1)
            self.input_simulator.release_key('q')
            self.input_simulator.release_key('a')
        elif action == "orbit_right":
            # Press rotate right and move right
            self.input_simulator.press_key('e')
            time.sleep(0.05)
            self.input_simulator.press_key('d')
            time.sleep(0.1)
            self.input_simulator.release_key('e')
            self.input_simulator.release_key('d')
        # Tilt with T/G per default bindings
        elif action == "tilt_up":
            self.input_simulator.key_press('t', duration=0.1)
        elif action == "tilt_down":
            self.input_simulator.key_press('g', duration=0.1)
        # Pan camera with a combo of rotation and movement
        elif action == "pan_left":
            self.input_simulator.key_press('a', duration=0.2)
        elif action == "pan_right":
            self.input_simulator.key_press('d', duration=0.2)
        
        # Reset mouse position occasionally to prevent getting stuck
        if random.random() > 0.8:  # 20% chance
            self._reset_mouse_position()
    
    def _handle_build_action(self, action: str):
        """Execute building actions."""
        # Get the client area of the game window for mouse actions
        if hasattr(self.screen_capture, 'client_position'):
            client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
            # Calculate window dimensions
            width = client_right - client_left
            height = client_bottom - client_top
            # Calculate center (relative to client area)
            center_x = width // 2
            center_y = height // 2
        else:
            # Fall back to system metrics if client position is not available
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            center_x, center_y = screen_width // 2, screen_height // 2
        
        # Use Escape to cancel any current actions before starting a new one (default keybinding)
        self.input_simulator.key_press('escape')
        time.sleep(0.2)
        
        if action == "residential_zone":
            # Select residential zoning
            self.input_simulator.key_press('1')
            time.sleep(0.2)
            self.input_simulator.key_press('r')
            time.sleep(0.2)
            
            # Draw a small residential zone with mouse
            start_x, start_y = center_x - 100, center_y - 100
            end_x, end_y = center_x, center_y
            self._drag_zone(start_x, start_y, end_x, end_y)
            
        elif action == "commercial_zone":
            # Select commercial zoning
            self.input_simulator.key_press('1')
            time.sleep(0.2)
            self.input_simulator.key_press('c')
            time.sleep(0.2)
            
            # Draw a small commercial zone with mouse
            start_x, start_y = center_x, center_y
            end_x, end_y = center_x + 100, center_y + 100
            self._drag_zone(start_x, start_y, end_x, end_y)
            
        elif action == "industrial_zone":
            # Select industrial zoning
            self.input_simulator.key_press('1')
            time.sleep(0.2)
            self.input_simulator.key_press('i')
            time.sleep(0.2)
            
            # Draw a small industrial zone with mouse
            start_x, start_y = center_x - 100, center_y + 50
            end_x, end_y = center_x + 100, center_y + 150
            self._drag_zone(start_x, start_y, end_x, end_y)
            
        elif action == "road":
            # Select road building
            self.input_simulator.key_press('2')
            time.sleep(0.2)
            self.input_simulator.key_press('r')
            time.sleep(0.2)
            
            # Draw a road with the mouse
            start_x, start_y = center_x - 150, center_y
            end_x, end_y = center_x + 150, center_y
            self._drag_zone(start_x, start_y, end_x, end_y)
            
        elif action == "highway":
            # Select highway building
            self.input_simulator.key_press('2')
            time.sleep(0.2)
            self.input_simulator.key_press('h')
            time.sleep(0.2)
            
            # Draw a highway with the mouse
            start_x, start_y = center_x - 200, center_y - 50
            end_x, end_y = center_x + 200, center_y - 50
            self._drag_zone(start_x, start_y, end_x, end_y)
            
        elif action == "avenue":
            # Select avenue building
            self.input_simulator.key_press('2')
            time.sleep(0.2)
            self.input_simulator.key_press('a')
            time.sleep(0.2)
            
            # Draw an avenue with the mouse
            start_x, start_y = center_x - 150, center_y + 50
            end_x, end_y = center_x + 150, center_y + 50
            self._drag_zone(start_x, start_y, end_x, end_y)
            
        elif action == "curve_road":
            # Select road building
            self.input_simulator.key_press('2')
            time.sleep(0.2)
            self.input_simulator.key_press('r')
            time.sleep(0.2)
            
            # Create a curved road with multiple segment clicks
            start_x, start_y = center_x - 150, center_y
            mid_x, mid_y = center_x, center_y - 100
            end_x, end_y = center_x + 150, center_y
            
            # Click to start the road
            self.input_simulator.mouse_click(start_x, start_y)
            time.sleep(0.3)
            
            # Click for the curve midpoint
            self.input_simulator.mouse_click(mid_x, mid_y)
            time.sleep(0.3)
            
            # Click for the endpoint
            self.input_simulator.mouse_click(end_x, end_y)
            time.sleep(0.3)
            
            # Right-click to finish
            self.input_simulator.mouse_click(end_x, end_y, button='right')
            
        elif action == "power_line":
            # Select power line
            self.input_simulator.key_press('3')
            time.sleep(0.2)
            self.input_simulator.key_press('p')
            time.sleep(0.2)
            
            # Draw a power line with the mouse
            start_x, start_y = center_x - 100, center_y - 50
            end_x, end_y = center_x + 100, center_y - 50
            self._drag_zone(start_x, start_y, end_x, end_y)
            
        elif action == "water_pipe":
            # Select water pipe
            self.input_simulator.key_press('3')
            time.sleep(0.2)
            self.input_simulator.key_press('w')
            time.sleep(0.2)
            
            # Draw a water pipe with the mouse
            start_x, start_y = center_x - 100, center_y + 50
            end_x, end_y = center_x + 100, center_y + 50
            self._drag_zone(start_x, start_y, end_x, end_y)
            
        elif action == "park":
            # Select park building
            self.input_simulator.key_press('5')
            time.sleep(0.2)
            self.input_simulator.key_press('p')
            time.sleep(0.2)
            
            # Place a park
            self._place_building(center_x - 75, center_y - 75)
            
        elif action == "plaza":
            # Select plaza building
            self.input_simulator.key_press('5')
            time.sleep(0.2)
            self.input_simulator.key_press('l')  # 'l' for plaza
            time.sleep(0.2)
            
            # Place a plaza
            self._place_building(center_x + 75, center_y - 75)
    
    def _handle_service_action(self, action: str):
        """Handle service building placement actions.
        
        Args:
            action (str): Service action name
        """
        try:
            screen_width, screen_height = self._get_screen_dimensions()
            center_x, center_y = screen_width // 2, screen_height // 2
        except Exception:
            # Fall back to system metrics
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            center_x, center_y = screen_width // 2, screen_height // 2
        
        # IMPORTANT: Don't directly use ESC which triggers menus
        # Instead of: self.input_simulator.key_press('escape')
        # First check if we're in a menu before doing anything
        frame = self.screen_capture.capture_frame()
        menu_detected = self.visual_estimator.detect_main_menu(frame)
        
        if menu_detected:
            logger.info("Menu detected, attempting to exit menu using safe method")
            self.input_simulator.safe_menu_handling()
            time.sleep(0.5)
            return
            
        # For service actions, we'll use the corresponding keyboard shortcuts
        # instead of navigating through menus
        time.sleep(0.2)
        
        if action == "police":
            # Select police station
            self.input_simulator.key_press('4')
            time.sleep(0.2)
            self.input_simulator.key_press('p')
            time.sleep(0.2)
            
            # Place police station
            self._place_building(center_x - 50, center_y - 50)
            
        elif action == "fire":
            # Select fire station
            self.input_simulator.key_press('4')
            time.sleep(0.2)
            self.input_simulator.key_press('f')
            time.sleep(0.2)
            
            # Place fire station
            self._place_building(center_x + 50, center_y - 50)
            
        elif action == "healthcare":
            # Select healthcare
            self.input_simulator.key_press('4')
            time.sleep(0.2)
            self.input_simulator.key_press('h')
            time.sleep(0.2)
            
            # Place healthcare building
            self._place_building(center_x, center_y + 50)
            
        elif action == "elementary_school":
            # Select education menu
            self.input_simulator.key_press('4')
            time.sleep(0.2)
            self.input_simulator.key_press('e')  # Education
            time.sleep(0.2)
            
            # Select elementary school
            self.input_simulator.key_press('1')  # Assuming 1 is for elementary
            time.sleep(0.2)
            
            # Place elementary school
            self._place_building(center_x - 100, center_y - 100)
            
        elif action == "high_school":
            # Select education menu
            self.input_simulator.key_press('4')
            time.sleep(0.2)
            self.input_simulator.key_press('e')  # Education
            time.sleep(0.2)
            
            # Select high school
            self.input_simulator.key_press('2')  # Assuming 2 is for high school
            time.sleep(0.2)
            
            # Place high school
            self._place_building(center_x, center_y - 100)
            
        elif action == "university":
            # Select education menu
            self.input_simulator.key_press('4')
            time.sleep(0.2)
            self.input_simulator.key_press('e')  # Education
            time.sleep(0.2)
            
            # Select university
            self.input_simulator.key_press('3')  # Assuming 3 is for university
            time.sleep(0.2)
            
            # Place university
            self._place_building(center_x + 100, center_y - 100)
            
        elif action == "bus_stop":
            # Select transportation menu
            self.input_simulator.key_press('6')  # Transportation
            time.sleep(0.2)
            self.input_simulator.key_press('b')  # Bus
            time.sleep(0.2)
            
            # Place bus stop
            self._place_building(center_x - 50, center_y + 100)
            
        elif action == "train_station":
            # Select transportation menu
            self.input_simulator.key_press('6')  # Transportation
            time.sleep(0.2)
            self.input_simulator.key_press('t')  # Train
            time.sleep(0.2)
            
            # Place train station
            self._place_building(center_x + 50, center_y + 100)
    
    def _handle_tool_action(self, action: str):
        """Execute tool actions using default game key bindings."""
        # Get screen resolution for mouse positioning
        if hasattr(self.screen_capture, 'client_position'):
            client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
            width = client_right - client_left
            height = client_bottom - client_top
            center_x = width // 2
            center_y = height // 2
        else:
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            center_x, center_y = screen_width // 2, screen_height // 2
        
        # Cancel any current action using Escape (default keybinding)
        self.input_simulator.key_press('escape')
        time.sleep(0.2)
        
        if action == "bulldoze":
            # Select bulldoze tool with B (default keybinding)
            self.input_simulator.key_press('b')
            time.sleep(0.2)
            
            # Bulldoze at center of screen
            self.input_simulator.mouse_click(center_x, center_y)
            
        elif action == "raise_terrain":
            # Use Page Up for increasing elevation (default keybinding)
            # First move mouse to target location
            target_x, target_y = center_x - 100, center_y - 100
            self.input_simulator.mouse_move(target_x, target_y)
            time.sleep(0.1)
            
            # Press Page Up to raise terrain
            self.input_simulator.key_press('page_up')
            time.sleep(0.1)
            
            # Click to apply
            self.input_simulator.mouse_click(target_x, target_y)
            
        elif action == "lower_terrain":
            # Use Page Down for decreasing elevation (default keybinding)
            # First move mouse to target location
            target_x, target_y = center_x + 100, center_y - 100
            self.input_simulator.mouse_move(target_x, target_y)
            time.sleep(0.1)
            
            # Press Page Down to lower terrain
            self.input_simulator.key_press('page_down')
            time.sleep(0.1)
            
            # Click to apply
            self.input_simulator.mouse_click(target_x, target_y)
            
        elif action == "level_terrain":
            # For level terrain, we'll use a combination of default controls
            # First ensure we're in terrain mode with Page Up/Down
            self.input_simulator.key_press('page_up')
            time.sleep(0.1)
            
            # Move to target location
            target_x, target_y = center_x, center_y - 100
            self.input_simulator.mouse_move(target_x, target_y)
            time.sleep(0.1)
            
            # Click and drag to level an area
            start_x, start_y = target_x - 50, target_y
            end_x, end_y = target_x + 50, target_y
            self.input_simulator.mouse_drag((start_x, start_y), (end_x, end_y))
        
        elif action == "focus_selection":
            # Focus on current selection with U (default keybinding)
            self.input_simulator.key_press('u')
            time.sleep(0.1)
            
        elif action == "hide_ui":
            # Hide UI with backtick (`) (default keybinding)
            self.input_simulator.key_press('`')
            time.sleep(0.1)
            
        elif action == "quicksave":
            # Quicksave with F5 (default keybinding)
            self.input_simulator.key_press('f5')
            time.sleep(0.1)
            
        elif action == "quickload":
            # Quickload with F9 (default keybinding)
            self.input_simulator.key_press('f9')
            time.sleep(0.1)
    
    def _handle_time_action(self, action: str):
        """Execute time control actions using default game key bindings."""
        if action == "pause":
            # Press space to pause/unpause (default keybinding)
            self.input_simulator.key_press('space')
            self.paused = not self.paused
            logger.info(f"Game {'paused' if self.paused else 'unpaused'}")
            
        elif action == "play_normal":
            # First ensure game is unpaused if needed
            if self.paused:
                self.input_simulator.key_press('space')
                self.paused = False
                
            # Set to normal speed (1) (default keybinding)
            self.input_simulator.key_press('1')
            self.game_speed = 1
            logger.info("Game speed set to normal")
            
        elif action == "play_fast":
            # First ensure game is unpaused if needed
            if self.paused:
                self.input_simulator.key_press('space')
                self.paused = False
                
            # Set to fast speed (2) (default keybinding)
            self.input_simulator.key_press('2')
            self.game_speed = 2
            logger.info("Game speed set to fast")
            
        elif action == "play_fastest":
            # First ensure game is unpaused if needed
            if self.paused:
                self.input_simulator.key_press('space')
                self.paused = False
                
            # Set to fastest speed (3) (default keybinding)
            self.input_simulator.key_press('3')
            self.game_speed = 3
            logger.info("Game speed set to fastest")
            
        elif action == "wait_short":
            # Wait for a short time (1 second)
            time.sleep(1.0)
            logger.info("Waited for 1 second")
            
        elif action == "wait_medium":
            # Wait for a medium time (3 seconds)
            time.sleep(3.0)
            logger.info("Waited for 3 seconds")
            
        elif action == "open_pause_menu":
            # Open pause menu with Escape (default keybinding)
            self.input_simulator.key_press('escape')
            time.sleep(0.5)
            logger.info("Opened pause menu")
    
    def _handle_info_view_action(self, action: str):
        """Execute info view actions using default game key bindings."""
        if action == "progression":
            # Open progression view with P (default keybinding)
            self.input_simulator.key_press('p')
            time.sleep(0.3)
            logger.info("Opened progression view")
            
        elif action == "city_economy":
            # Open city economy view with Z (default keybinding)
            self.input_simulator.key_press('z')
            time.sleep(0.3)
            logger.info("Opened city economy view")
            
        elif action == "city_information":
            # Open city information view with C (default keybinding)
            self.input_simulator.key_press('c')
            time.sleep(0.3)
            logger.info("Opened city information view")
            
        elif action == "city_statistics":
            # Open city statistics view with V (default keybinding)
            self.input_simulator.key_press('v')
            time.sleep(0.3)
            logger.info("Opened city statistics view")
            
        elif action == "transportation_overview":
            # Open transportation overview with X (default keybinding)
            self.input_simulator.key_press('x')
            time.sleep(0.3)
            logger.info("Opened transportation overview")
            
        elif action == "map_tiles":
            # Open map tiles view with M (default keybinding)
            self.input_simulator.key_press('m')
            time.sleep(0.3)
            logger.info("Opened map tiles view")
            
        elif action == "photo_mode":
            # Enter photo mode with period/dot (.) (default keybinding)
            self.input_simulator.key_press('.')
            time.sleep(0.3)
            logger.info("Entered photo mode")
            
        elif action == "take_photo":
            # Take photo with Enter (default keybinding)
            self.input_simulator.key_press('enter')
            time.sleep(0.3)
            logger.info("Took photo")
    
    def _drag_zone(self, start_x: int, start_y: int, end_x: int, end_y: int):
        """Simulate dragging action for zoning or road placement.
        
        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
        """
        try:
            # Position mouse at starting point
            self.input_simulator.mouse_move(start_x, start_y)
            time.sleep(0.3)
            
            # Calculate distance for duration
            distance = ((end_x - start_x) ** 2 + (end_y - start_y) ** 2) ** 0.5
            
            # Scale duration with distance
            duration = min(2.0, max(0.5, distance / 500))
            
            # Drag to ending point with calculated duration (None = auto-calculate based on distance and speed)
            self.input_simulator.mouse_drag(start_x, start_y, end_x, end_y, None)
            time.sleep(0.3)
            
            # Click to confirm (some actions require this)
            self.input_simulator.mouse_click(end_x, end_y, button='left', double=False)
            time.sleep(0.3)
            
        except Exception as e:
            logger.error(f"Error during drag operation: {e}")
    
    def _place_building(self, x: int, y: int):
        """Place a building at the given coordinates."""
        # Add random variation to prevent getting stuck
        rand_x = random.randint(-20, 20)
        rand_y = random.randint(-20, 20)
        
        # Add variation to coordinates
        x += rand_x
        y += rand_y
        
        # Get screen dimensions to ensure we stay in bounds
        width, height = self._get_screen_dimensions()
        margin = 50  # Pixels from edge to avoid
        
        # Ensure coordinates stay within screen bounds
        x = max(margin, min(width - margin, x))
        y = max(margin, min(height - margin, y))
        
        # Move to position
        self.input_simulator.mouse_move(x, y)
        time.sleep(0.2)
        
        # Click to place
        self.input_simulator.mouse_click(x, y)
        time.sleep(0.2)
        
        # Reset mouse position to center after placement
        self._reset_mouse_position()
    
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
        """Clean up resources."""
        try:
            # Ensure all keys and mouse buttons are released
            logger.info("Releasing all input devices...")
            
            # Release any pressed keys
            # Commonly used keys in the game
            common_keys = ['w', 'a', 's', 'd', 'q', 'e', 'r', 'f', 't', 'g', 
                          'escape', 'space', '1', '2', '3', '4', '5', '6']
            
            for key in common_keys:
                try:
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
                self._reset_mouse_position()
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

    def _handle_ui_action(self, action: str):
        """Execute UI exploration actions."""
        # Get client area dimensions for random coordinates
        if hasattr(self.screen_capture, 'client_position'):
            client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
            width = client_right - client_left
            height = client_bottom - client_top
        else:
            # Fallback to system metrics
            width = win32api.GetSystemMetrics(0)
            height = win32api.GetSystemMetrics(1)
            
        # Handle random UI interactions
        if action == "click_random":
            # Click at a random position
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            logger.info(f"Random UI click at ({x}, {y})")
            self.input_simulator.mouse_click(x, y)
            
        elif action == "right_click_random":
            # Right-click at a random position
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            logger.info(f"Random UI right-click at ({x}, {y})")
            self.input_simulator.mouse_click(x, y, button='right')
            
        elif action == "drag_random":
            # Drag from one random position to another
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)
            end_x = random.randint(0, width - 1)
            end_y = random.randint(0, height - 1)
            logger.info(f"Random UI drag from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            self.input_simulator.mouse_drag((start_x, start_y), (end_x, end_y))
            
        elif action == "click_top_menu":
            # Click in the top menu area (first 50 pixels from top)
            x = random.randint(0, width - 1)
            y = random.randint(0, 50)
            logger.info(f"UI click in top menu at ({x}, {y})")
            self.input_simulator.mouse_click(x, y)
            
        elif action == "hover_random":
            # Hover the mouse at a random position without clicking
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            logger.info(f"UI hover at ({x}, {y})")
            self.input_simulator.mouse_move(x, y)
            # Wait a moment to allow tooltips to appear
            time.sleep(0.5)
            
        elif action == "double_click_random":
            # Double click at a random position
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            logger.info(f"Random UI double-click at ({x}, {y})")
            self.input_simulator.mouse_click(x, y, double=True)
            
        elif action == "scroll_up":
            # Scroll up at current mouse position
            logger.info("UI scroll up")
            self.input_simulator.mouse_scroll(2)
            
        elif action == "scroll_down":
            # Scroll down at current mouse position
            logger.info("UI scroll down")
            self.input_simulator.mouse_scroll(-2)
    
    def _handle_ui_position_action(self, action: str, position: Tuple[int, int]):
        """Execute UI action at a specific position."""
        x, y = position
        
        if action == "click":
            logger.info(f"UI click at position ({x}, {y})")
            self.input_simulator.mouse_click(x, y)
        elif action == "right_click":
            logger.info(f"UI right-click at position ({x}, {y})")
            self.input_simulator.mouse_click(x, y, button='right')
        elif action == "double_click":
            logger.info(f"UI double-click at position ({x}, {y})")
            self.input_simulator.mouse_click(x, y, double=True)
        elif action == "drag":
            start_x, start_y = position.get("start", (0, 0))
            end_x, end_y = position.get("end", (0, 0))
            logger.info(f"UI drag from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            self.input_simulator.mouse_drag((start_x, start_y), (end_x, end_y))
    
    def _handle_sequence_action(self, action: str):
        """Execute multi-step action sequences."""
        # Get screen resolution
        if hasattr(self.screen_capture, 'client_position'):
            client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
            width = client_right - client_left
            height = client_bottom - client_top
            center_x = width // 2
            center_y = height // 2
        else:
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            center_x, center_y = screen_width // 2, screen_height // 2
        
        if action == "open_menu_click":
            # Sequence: Click on menu, wait, then click on submenu item
            # First, click in top menu area
            top_menu_x = random.randint(0, width - 1)
            top_menu_y = random.randint(0, 30)
            logger.info(f"Sequence: Click top menu at ({top_menu_x}, {top_menu_y})")
            self.input_simulator.mouse_click(top_menu_x, top_menu_y)
            
            # Wait for menu to appear
            time.sleep(0.5)
            
            # Then click on a random position below (assuming submenu item)
            submenu_x = top_menu_x + random.randint(-50, 50)
            submenu_y = top_menu_y + random.randint(30, 100)
            
            # Log if coordinates are outside normal bounds but don't restrict them
            if submenu_x < 0 or submenu_x >= width or submenu_y < 0 or submenu_y >= height:
                logger.info(f"Notice: Submenu position ({submenu_x}, {submenu_y}) is outside normal screen bounds")
            
            logger.info(f"Sequence: Click submenu at ({submenu_x}, {submenu_y})")
            self.input_simulator.mouse_click(submenu_x, submenu_y)
            
        elif action == "build_and_connect":
            # Sequence: Place a building then connect it with a road
            
            # 1. First place a residential zone
            self.input_simulator.key_press('1')  # Zoning
            time.sleep(0.2)
            self.input_simulator.key_press('r')  # Residential
            time.sleep(0.2)
            
            # Draw a small residential zone
            zone_start_x, zone_start_y = center_x - 100, center_y - 100
            zone_end_x, zone_end_y = center_x - 50, center_y - 50
            self._drag_zone(zone_start_x, zone_start_y, zone_end_x, zone_end_y)
            time.sleep(0.3)
            
            # 2. Then build a road connecting to it
            self.input_simulator.key_press('2')  # Roads
            time.sleep(0.2)
            self.input_simulator.key_press('r')  # Basic road
            time.sleep(0.2)
            
            # Draw a road from center to the zone
            road_start_x, road_start_y = center_x, center_y
            road_end_x, road_end_y = zone_end_x, zone_end_y
            self._drag_zone(road_start_x, road_start_y, road_end_x, road_end_y)
            time.sleep(0.3)
            
            # 3. Finally add a power line
            self.input_simulator.key_press('3')  # Utilities
            time.sleep(0.2)
            self.input_simulator.key_press('p')  # Power
            time.sleep(0.2)
            
            # Draw a power line to connect
            power_start_x, power_start_y = center_x + 50, center_y
            power_end_x, power_end_y = zone_start_x, zone_start_y
            self._drag_zone(power_start_x, power_start_y, power_end_x, power_end_y)
    
    def _get_screen_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the game window or screen."""
        if hasattr(self.screen_capture, 'client_position'):
            client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
            width = client_right - client_left
            height = client_bottom - client_top
        else:
            width = win32api.GetSystemMetrics(0)
            height = win32api.GetSystemMetrics(1)
            
        return width, height 