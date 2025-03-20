"""
High-level input actions for Cities: Skylines 2.

This module provides game-specific high-level actions built on the 
basic keyboard and mouse input functionality.
"""

import time
import logging
from typing import Tuple, Optional, Dict, List, Any

from .keyboard import KeyboardInput
from .mouse import MouseInput

logger = logging.getLogger(__name__)

class InputActions:
    """High-level input actions for the game environment."""
    
    def __init__(
        self,
        keyboard_input: KeyboardInput,
        mouse_input: MouseInput
    ):
        """Initialize actions with keyboard and mouse controllers.
        
        Args:
            keyboard_input: Keyboard input controller
            mouse_input: Mouse input controller
        """
        self.keyboard = keyboard_input
        self.mouse = mouse_input
        logger.info("Input actions initialized")
    
    # Camera control actions
    
    def rotate_camera(self, direction: str, duration: float = 0.5) -> bool:
        """Rotate the camera in the specified direction.
        
        Args:
            direction: Direction to rotate ('left', 'right', 'up', 'down')
            duration: How long to hold the rotation key
            
        Returns:
            True if successful, False otherwise
        """
        direction_keys = {
            'left': 'q',
            'right': 'e',
            'up': 'pageup',
            'down': 'pagedown'
        }
        
        key = direction_keys.get(direction.lower())
        if not key:
            logger.warning(f"Invalid camera rotation direction: {direction}")
            return False
            
        logger.info(f"Rotating camera {direction} for {duration}s")
        return self.keyboard.key_press(key, duration)
    
    def pan_camera(
        self, 
        direction: str = None, 
        start_pos: Optional[Tuple[int, int]] = None,
        end_pos: Optional[Tuple[int, int]] = None,
        duration: float = 0.5
    ) -> bool:
        """Pan the camera in a specified direction or between positions.
        
        Args:
            direction: Direction to pan ('left', 'right', 'up', 'down', 'upleft', etc.)
            start_pos: Start position for custom panning (screen coordinates)
            end_pos: End position for custom panning (screen coordinates)
            duration: Duration of panning
            
        Returns:
            True if successful, False otherwise
        """
        # Use predefined directions if specified
        if direction and not (start_pos and end_pos):
            # Get screen dimensions
            width, height = self.mouse.screen_width, self.mouse.screen_height
            center_x, center_y = width // 2, height // 2
            
            # Calculate pan distance (30% of screen)
            pan_distance_x = int(width * 0.3)
            pan_distance_y = int(height * 0.3)
            
            # Calculate start and end positions based on direction
            if direction.lower() == 'left':
                start_pos = (center_x, center_y)
                end_pos = (center_x + pan_distance_x, center_y)
            elif direction.lower() == 'right':
                start_pos = (center_x, center_y)
                end_pos = (center_x - pan_distance_x, center_y)
            elif direction.lower() == 'up':
                start_pos = (center_x, center_y)
                end_pos = (center_x, center_y + pan_distance_y)
            elif direction.lower() == 'down':
                start_pos = (center_x, center_y)
                end_pos = (center_x, center_y - pan_distance_y)
            elif direction.lower() == 'upleft':
                start_pos = (center_x, center_y)
                end_pos = (center_x + pan_distance_x, center_y + pan_distance_y)
            elif direction.lower() == 'upright':
                start_pos = (center_x, center_y)
                end_pos = (center_x - pan_distance_x, center_y + pan_distance_y)
            elif direction.lower() == 'downleft':
                start_pos = (center_x, center_y)
                end_pos = (center_x + pan_distance_x, center_y - pan_distance_y)
            elif direction.lower() == 'downright':
                start_pos = (center_x, center_y)
                end_pos = (center_x - pan_distance_x, center_y - pan_distance_y)
            else:
                logger.warning(f"Invalid camera pan direction: {direction}")
                return False
        
        # Make sure we have both positions
        if not (start_pos and end_pos):
            logger.warning("Both start and end positions must be provided for camera panning")
            return False
            
        # Perform the pan
        logger.info(f"Panning camera from {start_pos} to {end_pos}")
        return self.mouse.pan_camera(start_pos[0], start_pos[1], end_pos[0], end_pos[1], duration)
    
    def zoom_camera(self, direction: str, clicks: int = 5) -> bool:
        """Zoom the camera in or out.
        
        Args:
            direction: 'in' or 'out'
            clicks: Number of scroll clicks to perform
            
        Returns:
            True if successful, False otherwise
        """
        if direction.lower() not in ['in', 'out']:
            logger.warning(f"Invalid zoom direction: {direction}")
            return False
            
        # Determine scroll direction (positive = zoom in, negative = zoom out)
        scroll_amount = clicks if direction.lower() == 'in' else -clicks
        
        # Scroll at screen center
        center_x, center_y = self.mouse.screen_width // 2, self.mouse.screen_height // 2
        
        logger.info(f"Zooming camera {direction} with {clicks} clicks")
        return self.mouse.mouse_scroll(scroll_amount, center_x, center_y)
    
    # Game control actions
    
    def toggle_pause(self) -> bool:
        """Toggle game pause state.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Toggling game pause state")
        return self.keyboard.key_press('space')
    
    def toggle_ui(self) -> bool:
        """Toggle UI visibility.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Toggling UI visibility")
        return self.keyboard.key_press('u')
    
    def open_menu(self) -> bool:
        """Open game menu (ESC).
        
        Returns:
            True if successful, False otherwise
        """
        # Temporarily allow escape key
        old_block_state = self.keyboard.block_escape
        self.keyboard.block_escape = False
        
        try:
            logger.info("Opening menu (ESC)")
            result = self.keyboard.key_press('escape')
        finally:
            # Restore escape key blocking state
            self.keyboard.block_escape = old_block_state
            
        return result
    
    def take_screenshot(self) -> bool:
        """Take an in-game screenshot.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Taking screenshot")
        return self.keyboard.key_press('f11')
    
    # Tool selection actions
    
    def select_tool(self, tool_number: int) -> bool:
        """Select a tool by number (1-9).
        
        Args:
            tool_number: Tool number to select (1-9)
            
        Returns:
            True if successful, False otherwise
        """
        if not (1 <= tool_number <= 9):
            logger.warning(f"Invalid tool number: {tool_number}, must be 1-9")
            return False
            
        logger.info(f"Selecting tool {tool_number}")
        return self.keyboard.key_press(str(tool_number))
    
    def cancel_tool(self) -> bool:
        """Cancel current tool selection.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Canceling tool selection")
        return self.keyboard.key_press('escape', force_direct=True)
    
    # Building and zoning actions
    
    def place_object(self, x: int, y: int) -> bool:
        """Place an object at the specified location.
        
        Args:
            x: X screen coordinate
            y: Y screen coordinate
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Placing object at ({x}, {y})")
        return self.mouse.mouse_click(x, y)
    
    def drag_zone(self, start: Tuple[int, int], end: Tuple[int, int], duration: float = 0.5) -> bool:
        """Drag to create a zone.
        
        Args:
            start: Start coordinates (x, y)
            end: End coordinates (x, y)
            duration: Duration of drag operation
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Creating zone from {start} to {end}")
        return self.mouse.mouse_drag(start, end, 'left', duration)
    
    def place_road(self, points: List[Tuple[int, int]], duration_per_segment: float = 0.5) -> bool:
        """Place a road by connecting multiple points.
        
        Args:
            points: List of (x, y) coordinates to connect
            duration_per_segment: Duration per road segment
            
        Returns:
            True if successful, False otherwise
        """
        if len(points) < 2:
            logger.warning("At least two points required to place a road")
            return False
            
        logger.info(f"Placing road with {len(points)} points")
        
        # First click
        if not self.mouse.mouse_click(points[0][0], points[0][1]):
            logger.warning(f"Failed to click first road point at {points[0]}")
            return False
            
        # Connect subsequent points
        for i in range(1, len(points)):
            time.sleep(0.2)  # Short delay between segments
            if not self.mouse.mouse_click(points[i][0], points[i][1]):
                logger.warning(f"Failed to click road point at {points[i]}")
                return False
                
        # Double-click last point to complete
        time.sleep(0.2)
        return self.mouse.mouse_click(points[-1][0], points[-1][1], double=True)
    
    def demolish(self, position: Tuple[int, int]) -> bool:
        """Demolish at specified position.
        
        Args:
            position: (x, y) coordinates to demolish
            
        Returns:
            True if successful, False otherwise
        """
        # First select demolish tool
        if not self.keyboard.key_press('delete'):
            logger.warning("Failed to select demolish tool")
            return False
            
        # Wait for tool to activate
        time.sleep(0.3)
        
        # Click to demolish
        logger.info(f"Demolishing at {position}")
        return self.mouse.mouse_click(position[0], position[1])
    
    # Combined actions
    
    def close(self) -> None:
        """Release all input resources."""
        # Nothing specific needed here, as keyboard and mouse are
        # passed in and managed externally
        logger.info("Input actions resources released")


class ActionExecutor:
    """Executes high-level actions in the game environment."""
    
    def __init__(
        self,
        keyboard_controller = None,
        mouse_controller = None,
        config: Dict[str, Any] = None
    ):
        """Initialize action executor.
        
        Args:
            keyboard_controller: Keyboard controller
            mouse_controller: Mouse controller
            config: Configuration dictionary
        """
        self.keyboard = keyboard_controller
        self.mouse = mouse_controller
        self.config = config or {}
        
        # Initialize input actions if controllers provided
        self.actions = None
        if self.keyboard and self.mouse:
            self.actions = InputActions(self.keyboard, self.mouse)
            
        # Action history for tracking
        self.action_history = []
        self.max_history_size = 100
        
        logger.info("Action executor initialized")
        
    def set_controllers(
        self,
        keyboard_controller,
        mouse_controller
    ):
        """Set input controllers.
        
        Args:
            keyboard_controller: Keyboard controller
            mouse_controller: Mouse controller
        """
        self.keyboard = keyboard_controller
        self.mouse = mouse_controller
        self.actions = InputActions(self.keyboard, self.mouse)
        logger.info("Input controllers set for action executor")
        
    def execute_action(self, action_name: str, **kwargs) -> bool:
        """Execute a named action with parameters.
        
        Args:
            action_name: Name of the action to execute
            **kwargs: Parameters for the action
            
        Returns:
            bool: Success flag
        """
        if not self.actions:
            logger.error("Cannot execute action, no input actions available")
            return False
            
        # Check if the action exists
        if not hasattr(self.actions, action_name):
            logger.error(f"Unknown action: {action_name}")
            return False
            
        try:
            # Get the action method
            action_method = getattr(self.actions, action_name)
            
            # Execute action
            logger.info(f"Executing action: {action_name} with params: {kwargs}")
            result = action_method(**kwargs)
            
            # Record action
            self._record_action(action_name, kwargs, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing action {action_name}: {e}")
            return False
            
    def _record_action(self, action_name: str, params: dict, success: bool):
        """Record action execution to history.
        
        Args:
            action_name: Name of the action
            params: Parameters used
            success: Whether action was successful
        """
        timestamp = time.time()
        record = {
            'timestamp': timestamp,
            'action': action_name,
            'params': params,
            'success': success
        }
        
        self.action_history.append(record)
        
        # Limit history size
        if len(self.action_history) > self.max_history_size:
            self.action_history.pop(0)
            
    def get_action_history(self) -> List[Dict]:
        """Get action execution history.
        
        Returns:
            List of action records
        """
        return self.action_history.copy()


class InputSimulator:
    """Simulates all user input for the game environment."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize input simulator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        input_config = self.config.get('input', {})
        
        # Initialize components
        self.keyboard_controller = None
        self.mouse_controller = None
        self.action_executor = None
        
        # Configure components based on config
        self._configure_input(input_config)
        
        logger.info("Input simulator initialized")
        
    def _configure_input(self, input_config: Dict[str, Any]):
        """Configure input components.
        
        Args:
            input_config: Input configuration dictionary
        """
        # Create keyboard controller
        from .keyboard import KeyboardController
        self.keyboard_controller = KeyboardController(
            config=input_config.get('keyboard', {})
        )
        
        # Create mouse controller
        from .mouse import MouseController
        self.mouse_controller = MouseController(
            config=input_config.get('mouse', {})
        )
        
        # Create action executor
        self.action_executor = ActionExecutor(
            keyboard_controller=self.keyboard_controller,
            mouse_controller=self.mouse_controller,
            config=input_config
        )
        
    def get_action_executor(self) -> ActionExecutor:
        """Get the action executor.
        
        Returns:
            ActionExecutor instance
        """
        return self.action_executor
        
    def cleanup(self):
        """Cleanup resources used by the input simulator."""
        if self.keyboard_controller:
            self.keyboard_controller.cleanup()
        if self.mouse_controller:
            self.mouse_controller.cleanup()
        logger.info("Input simulator cleaned up") 