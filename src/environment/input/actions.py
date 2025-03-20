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
    
    def click(self, x: int, y: int, button: str = 'left', double: bool = False) -> bool:
        """Click at a specific location.
        
        Args:
            x: X screen coordinate
            y: Y screen coordinate
            button: Mouse button ('left', 'right', or 'middle')
            double: Whether to perform a double click
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Clicking at ({x}, {y}) with {button} button, double={double}")
        if double:
            return self.mouse.mouse_double_click(x, y, button)
        else:
            return self.mouse.mouse_click(x, y, button)
    
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5, button: str = 'left') -> bool:
        """Drag from one position to another.
        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration: Duration of the drag operation
            button: Mouse button to use for dragging
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Dragging from ({start_x}, {start_y}) to ({end_x}, {end_y}), duration={duration}s")
        return self.mouse.mouse_drag(start_x, start_y, end_x, end_y, duration, button)
    
    def edge_scroll(self, direction: str, duration: float = 0.5) -> bool:
        """Scroll the view by moving the mouse to the screen edge.
        
        Args:
            direction: Direction to scroll ('left', 'right', 'up', 'down')
            duration: How long to hold at the edge
            
        Returns:
            True if successful, False otherwise
        """
        # Calculate edge position based on direction
        screen_width, screen_height = self.mouse.screen_width, self.mouse.screen_height
        
        # Calculate coordinates (stay 2 pixels from edge to avoid triggering OS features)
        if direction.lower() == 'left':
            x, y = 2, screen_height // 2
        elif direction.lower() == 'right':
            x, y = screen_width - 2, screen_height // 2
        elif direction.lower() == 'up':
            x, y = screen_width // 2, 2
        elif direction.lower() == 'down':
            x, y = screen_width // 2, screen_height - 2
        else:
            logger.warning(f"Invalid edge scroll direction: {direction}")
            return False
        
        logger.info(f"Edge scrolling {direction} for {duration}s")
        
        # Move mouse to edge and wait
        self.mouse.mouse_move(x, y)
        time.sleep(duration)
        
        # Move back to center
        self.mouse.mouse_move(screen_width // 2, screen_height // 2)
        
        return True


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
            
        # Handle case where action is an integer (action ID)
        if isinstance(action_name, int):
            logger.info(f"Converting action ID {action_name} to string")
            action_name = str(action_name)
            
        # Handle case where action is a dict
        if isinstance(action_name, dict):
            # Handle "type" + "key" format - convert to key_press action
            if 'type' in action_name and action_name['type'] == 'key' and 'key' in action_name:
                key = action_name.get('key')
                duration = action_name.get('duration', 0.1)
                logger.info(f"Converting 'key' action format to key_press with key={key}, duration={duration}")
                try:
                    # Verify input controller is ready before executing
                    if not hasattr(self, 'keyboard') or self.keyboard is None:
                        logger.error("Keyboard controller not available")
                        return False
                    
                    # Use the correct keyboard_input method 
                    from .keyboard import KeyboardInput
                    if isinstance(self.keyboard, KeyboardInput):
                        # Short delay before key press to ensure window is ready
                        time.sleep(0.1)
                        # Execute with retry if needed
                        for attempt in range(2):
                            result = self.keyboard.key_press(key, duration)
                            if result:
                                logger.debug(f"Key press {key} successful on attempt {attempt+1}")
                                return True
                            elif attempt < 1:  # Don't wait after last attempt
                                logger.warning(f"Key press for {key} failed, retrying after delay")
                                time.sleep(0.5)
                        logger.error(f"All key press attempts for {key} failed")
                        return False
                    else:
                        logger.error(f"Keyboard controller doesn't support key_press method")
                        return False
                except Exception as e:
                    logger.error(f"Error executing key press action for key={key}: {e}")
                    return False
            
            # Handle "type" + "mouse" format - convert to appropriate mouse action
            elif 'type' in action_name and action_name['type'] == 'mouse' and 'action' in action_name:
                mouse_action = action_name.get('action')
                button = action_name.get('button', 'left')
                position = action_name.get('position', None)
                
                # Extract normalized position if provided and convert to screen coordinates
                if position and len(position) == 2:
                    screen_width, screen_height = self._get_screen_dimensions()
                    x = int(position[0] * screen_width)
                    y = int(position[1] * screen_height)
                    logger.info(f"Converted normalized position {position} to screen coordinates ({x}, {y})")
                    
                    try:
                        # Verify mouse controller is ready
                        if not hasattr(self, 'mouse') or self.mouse is None:
                            logger.error("Mouse controller not available")
                            return False
                            
                        from .mouse import MouseInput
                        if isinstance(self.mouse, MouseInput):
                            # Short delay before mouse action to ensure window is ready
                            time.sleep(0.1)
                            
                            # Execute with retry if needed
                            for attempt in range(2):
                                if mouse_action == 'click':
                                    # First ensure mouse is at the target position
                                    move_result = self.mouse.mouse_move(x, y)
                                    if not move_result:
                                        logger.warning(f"Failed to move mouse to ({x}, {y}) before click")
                                        time.sleep(0.2)
                                        # Try one more time
                                        move_result = self.mouse.mouse_move(x, y)
                                        
                                    time.sleep(0.2)  # Wait for mouse to settle
                                    result = self.mouse.mouse_click(x, y, button=button, double=False)
                                elif mouse_action == 'double_click':
                                    # First ensure mouse is at the target position
                                    move_result = self.mouse.mouse_move(x, y)
                                    if not move_result:
                                        logger.warning(f"Failed to move mouse to ({x}, {y}) before double-click")
                                        time.sleep(0.2)
                                        # Try one more time
                                        move_result = self.mouse.mouse_move(x, y)
                                        
                                    time.sleep(0.2)  # Wait for mouse to settle
                                    result = self.mouse.mouse_click(x, y, button=button, double=True)
                                elif mouse_action == 'move':
                                    result = self.mouse.mouse_move(x, y)
                                else:
                                    logger.error(f"Unknown mouse action: {mouse_action}")
                                    return False
                                    
                                if result:
                                    logger.debug(f"Mouse action {mouse_action} at ({x}, {y}) successful on attempt {attempt+1}")
                                    return True
                                elif attempt < 1:  # Don't wait after last attempt
                                    logger.warning(f"Mouse action {mouse_action} at ({x}, {y}) failed, retrying after delay")
                                    time.sleep(0.5)
                            logger.error(f"All {mouse_action} attempts at ({x}, {y}) failed")
                            return False
                        else:
                            logger.error(f"Mouse controller doesn't support the required methods")
                            return False
                    except Exception as e:
                        logger.error(f"Error executing mouse action {mouse_action} at ({x}, {y}): {e}")
                        return False
                else:
                    logger.warning(f"Mouse action {mouse_action} requires position, but none provided or invalid")
                    return False
            
            # Legacy format with 'action' key
            elif 'action' in action_name:
                params = {k: v for k, v in action_name.items() if k != 'action'}
                action_str = action_name['action']
                logger.debug(f"Extracted action '{action_str}' from dictionary with params: {params}")
                return self.execute_action(action_str, **params)
            else:
                logger.error(f"Invalid action dictionary format: {action_name}")
                return False
            
        # Check if the action exists as a direct method
        action_str = str(action_name)
        
        if hasattr(self.actions, action_str):
            try:
                # Get the action method
                action_method = getattr(self.actions, action_str)
                
                # Execute action with appropriate parameters
                logger.info(f"Executing action: {action_str} with params: {kwargs}")
                start_time = time.time()
                
                # Add a small delay before action to ensure window is ready
                time.sleep(0.1)
                
                # Execute with retry if needed
                for attempt in range(2):
                    result = action_method(**kwargs)
                    if result:
                        execution_time = time.time() - start_time
                        # Record action with execution time
                        self._record_action(action_str, kwargs, result, execution_time)
                        logger.debug(f"Action {action_str} executed successfully in {execution_time:.3f}s")
                        return True
                    elif attempt < 1:  # Don't wait after last attempt
                        logger.warning(f"Action {action_str} failed, retrying after delay")
                        time.sleep(0.5)
                
                # If we get here, all attempts failed
                execution_time = time.time() - start_time
                self._record_action(action_str, kwargs, False, execution_time)
                logger.warning(f"Action {action_str} reported failure after {execution_time:.3f}s")
                return False
                
            except Exception as e:
                logger.error(f"Error executing action {action_str}: {e}")
                self._record_action(action_str, kwargs, False, 0, error=str(e))
                return False
        else:
            # For undocumented actions, try to handle based on name conventions
            # This allows for more flexible action handling
            try:
                if action_str.startswith('key_'):
                    # Handle key_X actions (where X is a key name)
                    key = action_str[4:]  # Extract key name
                    logger.info(f"Converting action name '{action_str}' to key press for key '{key}'")
                    # Add a small delay before action to ensure window is ready
                    time.sleep(0.1)
                    result = self.keyboard.key_press(key, kwargs.get('duration', 0.1))
                    self._record_action(action_str, kwargs, result, 0)
                    return result
                    
                elif action_str.startswith('mouse_'):
                    # Handle mouse_X actions (click, move, etc.)
                    logger.info(f"Attempting to handle undocumented mouse action: {action_str}")
                    # Try to find a similar method in mouse controller
                    for method_name in dir(self.mouse):
                        if method_name.lower() == action_str.lower() or method_name.lower() == action_str[6:].lower():
                            logger.info(f"Found matching mouse method: {method_name}")
                            method = getattr(self.mouse, method_name)
                            # Add a small delay before action to ensure window is ready
                            time.sleep(0.1)
                            result = method(**kwargs)
                            self._record_action(action_str, kwargs, result, 0)
                            return result
                
                logger.error(f"Unknown action: {action_str}")
                return False
                
            except Exception as e:
                logger.error(f"Error handling undocumented action {action_str}: {e}")
                self._record_action(action_str, kwargs, False, 0, error=str(e))
                return False
    
    def _get_screen_dimensions(self) -> Tuple[int, int]:
        """Get screen dimensions for position calculations.
        
        Returns:
            Tuple[int, int]: Width and height of screen
        """
        try:
            # Default dimensions
            default_width, default_height = 1920, 1080
            
            # Try to get actual screen dimensions
            import win32api
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            
            # Return valid dimensions or defaults
            if screen_width > 0 and screen_height > 0:
                return screen_width, screen_height
                
            return default_width, default_height
        except Exception as e:
            logger.error(f"Error getting screen dimensions: {e}")
            return 1920, 1080
    
    def _record_action(self, action_name: str, params: dict, success: bool, execution_time: float = 0, error: str = None):
        """Record action execution to history.
        
        Args:
            action_name: Name of the action
            params: Parameters used
            success: Whether action was successful
            execution_time: Time taken to execute the action
            error: Error message if action failed
        """
        timestamp = time.time()
        record = {
            'timestamp': timestamp,
            'action': action_name,
            'params': params,
            'success': success,
            'execution_time': execution_time,
            'error': error
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
        from .keyboard import KeyboardInput
        self.keyboard_controller = KeyboardInput()
        self.keyboard_input = self.keyboard_controller
        
        # Create mouse controller
        from .mouse import MouseInput
        self.mouse_controller = MouseInput()
        
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
    
    def press_key(self, key: str) -> bool:
        """Press a key without holding it.
        
        Args:
            key: Key to press
            
        Returns:
            bool: Success status
        """
        try:
            # Press and quickly release the key using the keyboard input
            if hasattr(self, 'keyboard_input') and self.keyboard_input:
                return self.keyboard_input.key_press(key, duration=0.1)
            return False
        except Exception as e:
            logger.error(f"Error in press_key: {e}")
            return False
        
    def cleanup(self):
        """Cleanup resources used by the input simulator."""
        if self.keyboard_controller:
            self.keyboard_controller.cleanup()
        if self.mouse_controller:
            self.mouse_controller.cleanup()
        logger.info("Input simulator cleaned up") 