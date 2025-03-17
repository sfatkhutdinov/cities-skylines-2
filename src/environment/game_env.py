"""
Game environment for Cities: Skylines 2.
Handles game interaction, observation, and reward computation.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional
import time
from environment.visual_metrics import VisualMetricsEstimator
from environment.reward_system import RewardSystem
from environment.optimized_capture import OptimizedScreenCapture
from config.hardware_config import HardwareConfig
from utils.performance_safeguards import PerformanceSafeguards
from environment.input_simulator import InputSimulator
import logging
import win32api
import random

logger = logging.getLogger(__name__)

class CitiesEnvironment:
    """Environment for interacting with Cities: Skylines 2."""
    
    def __init__(self, config: Optional[HardwareConfig] = None, mock_mode: bool = False):
        """Initialize the environment.
        
        Args:
            config: Optional hardware configuration
            mock_mode: If True, use a mock environment for testing/training without the actual game
        """
        self.config = config or HardwareConfig()
        self.mock_mode = mock_mode
        
        # Initialize components
        self.screen_capture = OptimizedScreenCapture(self.config)
        self.input_simulator = InputSimulator()
        # Connect input simulator to screen capture for coordinate translation
        self.input_simulator.screen_capture = self.screen_capture
        self.visual_estimator = VisualMetricsEstimator(self.config)
        self.reward_system = RewardSystem(self.config)
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
        
        # For mock mode, create a dummy frame
        if self.mock_mode:
            # Create a simple 3-channel frame (320x240 RGB)
            self.mock_frame = torch.ones((3, 240, 320), dtype=torch.float32)
            # Add some random elements to make it more realistic
            self.mock_frame = torch.rand_like(self.mock_frame)
            # Inform the screen capture to use mock mode
            self.screen_capture.use_mock = True
    
    def _setup_actions(self) -> Dict[int, Dict[str, Any]]:
        """Setup the action space for the agent."""
        base_actions = {
            0: {"type": "camera", "action": "move_up"},
            1: {"type": "camera", "action": "move_down"},
            2: {"type": "camera", "action": "move_left"},
            3: {"type": "camera", "action": "move_right"},
            4: {"type": "camera", "action": "zoom_in"},
            5: {"type": "camera", "action": "zoom_out"},
            6: {"type": "camera", "action": "rotate_left"},
            7: {"type": "camera", "action": "rotate_right"},
            8: {"type": "camera", "action": "tilt_up"},
            9: {"type": "camera", "action": "tilt_down"},
            10: {"type": "camera", "action": "pan_left"},
            11: {"type": "camera", "action": "pan_right"},
            # New UI exploration actions
            12: {"type": "ui", "action": "click_random"},
            13: {"type": "ui", "action": "right_click_random"},
            14: {"type": "ui", "action": "drag_random"},
            15: {"type": "ui", "action": "click_top_menu"},
            # Building actions
            16: {"type": "build", "action": "residential_zone"},
            17: {"type": "build", "action": "commercial_zone"},
            18: {"type": "build", "action": "industrial_zone"},
            19: {"type": "build", "action": "road"},
            20: {"type": "build", "action": "power_line"},
            21: {"type": "build", "action": "water_pipe"},
            22: {"type": "service", "action": "police"},
            23: {"type": "service", "action": "fire"},
            24: {"type": "service", "action": "healthcare"},
            25: {"type": "tool", "action": "bulldoze"}
        }
        
        # Add free exploration coordinates
        # Allow the agent to click anywhere on screen by selecting discrete coordinates
        screen_width = 1920  # Default full HD
        screen_height = 1080
        
        if hasattr(self, 'screen_capture') and hasattr(self.screen_capture, 'client_position'):
            client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
            screen_width = client_right - client_left
            screen_height = client_bottom - client_top
        
        # Create grid of points across screen (5x5 grid = 25 additional actions)
        grid_size = 5
        for i in range(grid_size):
            for j in range(grid_size):
                x = int(screen_width * (i + 0.5) / grid_size)
                y = int(screen_height * (j + 0.5) / grid_size)
                action_idx = 26 + i * grid_size + j
                base_actions[action_idx] = {
                    "type": "ui_position", 
                    "action": "click",
                    "position": (x, y)
                }
        
        return base_actions
    
    def reset(self) -> torch.Tensor:
        """Reset the environment to initial state."""
        # In mock mode, just return the mock frame
        if self.mock_mode:
            self.current_frame = self.screen_capture.capture_frame()
            self.steps_taken = 0
            return self.current_frame
            
        # For real game mode, try to focus the game window
        logger.info("Focusing Cities: Skylines II window...")
        if not self.input_simulator.find_game_window():
            raise RuntimeError("Cities: Skylines II window not found. Make sure the game is running.")
                
        if not self.input_simulator.ensure_game_window_focused():
            raise RuntimeError("Could not focus Cities: Skylines II window. Make sure the game window is visible.")
            
        # Reset game state (unpause if paused)
        self._ensure_game_running()
        
        # Set normal game speed
        self._set_game_speed(1)
        
        # Run mouse freedom test during first reset to diagnose any issues
        # Only run on the first episode to avoid disrupting training
        if self.steps_taken == 0:
            self.test_mouse_freedom()
        
        # Capture initial frame
        self.current_frame = self.screen_capture.capture_frame()
        self.steps_taken = 0
        
        return self.current_frame
        
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
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action (int): Action to take
            
        Returns:
            Tuple[torch.Tensor, float, bool, Dict[str, Any]]: (next_state, reward, done, info)
        """
        # Check if we need to delay before taking another action
        elapsed = time.time() - self.last_action_time
        if elapsed < self.min_action_delay:
            time.sleep(self.min_action_delay - elapsed)
            
        # Get action details
        if action < 0 or action >= len(self.actions):
            logger.warning(f"Invalid action index: {action}")
            action = 0  # Default to first action
            
        action_info = self.actions[action]
        
        # Execute action
        self._execute_action(action_info)
        self.last_action_time = time.time()
        self.steps_taken += 1
        
        # Observe new state
        next_frame = self.screen_capture.capture_frame()
        self.current_frame = next_frame
        
        # Calculate reward
        if hasattr(self, "reward_system") and hasattr(self.reward_system, "compute_reward"):
            # Use the new reward system
            reward = self.reward_system.compute_reward(next_frame, action, action_info)
        else:
            # Legacy reward calculation
            reward, _ = self.reward_system.compute_population_reward(next_frame)
            
        # Check if episode is done
        done = self.steps_taken >= self.max_steps
        
        # Compile info
        info = {
            "steps": self.steps_taken,
            "action_type": action_info["type"],
            "action": action_info["action"]
        }
        
        return next_frame, reward, done, info
    
    def _execute_action(self, action_info: Dict[str, Any]):
        """Execute the specified action in the game."""
        action_type = action_info["type"]
        action = action_info["action"]
        
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
            position = action_info.get("position", (0, 0))
            self._handle_ui_position_action(action, position)
    
    def _handle_camera_action(self, action: str):
        """Execute camera movement actions."""
        # Get the center of the game window for rotation and panning
        if hasattr(self.screen_capture, 'client_position'):
            client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
            width = client_right - client_left
            height = client_bottom - client_top
            center_x = width // 2
            center_y = height // 2
        else:
            # Fallback to system metrics
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            center_x, center_y = screen_width // 2, screen_height // 2
            
        # Handle basic movement with keys
        if action == "move_up":
            self.input_simulator.key_press('w', duration=0.1)
        elif action == "move_down":
            self.input_simulator.key_press('s', duration=0.1)
        elif action == "move_left":
            self.input_simulator.key_press('a', duration=0.1)
        elif action == "move_right":
            self.input_simulator.key_press('d', duration=0.1)
        elif action == "zoom_in":
            self.input_simulator.mouse_scroll(1)
        elif action == "zoom_out":
            self.input_simulator.mouse_scroll(-1)
        # Handle rotation with middle-click drag
        elif action == "rotate_left":
            # Rotate left with middle mouse drag
            start_x, start_y = center_x, center_y
            end_x, end_y = center_x - 100, center_y
            self.input_simulator.rotate_camera(start_x, start_y, end_x, end_y)
        elif action == "rotate_right":
            # Rotate right with middle mouse drag
            start_x, start_y = center_x, center_y
            end_x, end_y = center_x + 100, center_y
            self.input_simulator.rotate_camera(start_x, start_y, end_x, end_y)
        elif action == "tilt_up":
            # Tilt up with middle mouse drag
            start_x, start_y = center_x, center_y
            end_x, end_y = center_x, center_y - 100
            self.input_simulator.rotate_camera(start_x, start_y, end_x, end_y)
        elif action == "tilt_down":
            # Tilt down with middle mouse drag
            start_x, start_y = center_x, center_y
            end_x, end_y = center_x, center_y + 100
            self.input_simulator.rotate_camera(start_x, start_y, end_x, end_y)
        # Handle panning with right-click drag
        elif action == "pan_left":
            # Pan left with right mouse drag
            start_x, start_y = center_x, center_y
            end_x, end_y = center_x + 100, center_y
            self.input_simulator.pan_camera(start_x, start_y, end_x, end_y)
        elif action == "pan_right":
            # Pan right with right mouse drag
            start_x, start_y = center_x, center_y
            end_x, end_y = center_x - 100, center_y
            self.input_simulator.pan_camera(start_x, start_y, end_x, end_y)
        # Handle corner movements to test mouse freedom
        elif action == "move_top_left":
            # Move to top-left corner of game window
            self.input_simulator.mouse_move(0, 0)
            time.sleep(0.2)
        elif action == "move_top_right":
            # Move to top-right corner of game window
            self.input_simulator.mouse_move(width-1, 0)
            time.sleep(0.2)
        elif action == "move_bottom_left":
            # Move to bottom-left corner of game window
            self.input_simulator.mouse_move(0, height-1)
            time.sleep(0.2)
        elif action == "move_bottom_right":
            # Move to bottom-right corner of game window
            self.input_simulator.mouse_move(width-1, height-1)
            time.sleep(0.2)
    
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
        
        # Use Tab key instead of ESC to cancel any current actions before starting a new one
        # This is safer than using ESC which might bring up the menu
        self.input_simulator.press_key('tab')
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
            
    def _handle_service_action(self, action: str):
        """Execute service building actions."""
        # Get screen resolution
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)
        
        # Calculate center and placement positions
        center_x, center_y = screen_width // 2, screen_height // 2
        
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
            
    def _handle_tool_action(self, action: str):
        """Execute tool actions."""
        # Get screen resolution
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)
        
        # Calculate center and various positions
        center_x, center_y = screen_width // 2, screen_height // 2
        
        if action == "bulldoze":
            self.input_simulator.key_press('b')
            time.sleep(0.2)
            
            # Use the bulldoze tool
            self._use_tool(center_x - 25, center_y - 25, center_x + 25, center_y + 25)
    
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
            
            # Press and hold mouse button
            self.input_simulator.mouse_click(start_x, start_y, button='left', double=False)
            time.sleep(0.3)
            
            # Drag to ending point
            self.input_simulator.mouse_drag((start_x, start_y), (end_x, end_y), button='left', duration=0.5)
            time.sleep(0.3)
            
            # Click to confirm (some actions require this)
            self.input_simulator.mouse_click(end_x, end_y, button='left', double=False)
            time.sleep(0.3)
            
        except Exception as e:
            logger.error(f"Error during drag operation: {e}")
    
    def _place_building(self, x: int, y: int):
        """Place a building at the specified coordinates.
        
        Args:
            x, y: Coordinates for building placement
        """
        try:
            # Move to position
            self.input_simulator.mouse_move(x, y)
            time.sleep(0.3)
            
            # Click to place
            self.input_simulator.mouse_click(x, y, button='left')
            time.sleep(0.3)
            
        except Exception as e:
            logger.error(f"Error during building placement: {e}")
    
    def _use_tool(self, start_x: int, start_y: int, end_x: int, end_y: int):
        """Use a tool like bulldoze by dragging over an area.
        
        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
        """
        try:
            # Position at start
            self.input_simulator.mouse_move(start_x, start_y)
            time.sleep(0.3)
            
            # Drag to use the tool
            self.input_simulator.mouse_drag((start_x, start_y), (end_x, end_y), button='left', duration=0.5)
            time.sleep(0.3)
            
        except Exception as e:
            logger.error(f"Error using tool: {e}")
    
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
        self.screen_capture.close()
        self.input_simulator.close()
        
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
        """Test mouse movement across the entire game window.
        Useful for debugging mouse constraints."""
        if hasattr(self.screen_capture, 'client_position'):
            client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
            width = client_right - client_left
            height = client_bottom - client_top
            
            print(f"\nTesting mouse freedom across game window ({width}x{height})")
            
            # Get 9 points distributed across the window
            points = [
                (0, 0),                    # top-left
                (width//2, 0),             # top-center
                (width-1, 0),              # top-right
                (0, height//2),            # mid-left
                (width//2, height//2),     # center
                (width-1, height//2),      # mid-right
                (0, height-1),             # bottom-left
                (width//2, height-1),      # bottom-center
                (width-1, height-1)        # bottom-right
            ]
            
            # Move to each point with a delay
            for i, (x, y) in enumerate(points):
                print(f"Moving to point {i+1}/9: ({x}, {y})")
                self.input_simulator.mouse_move(x, y)
                time.sleep(0.5)
                
            # Return to center
            self.input_simulator.mouse_move(width//2, height//2)
            print("Mouse freedom test completed\n")

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