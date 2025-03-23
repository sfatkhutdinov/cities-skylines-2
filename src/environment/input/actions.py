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
        
        # Track last execution time for rate limiting
        self.last_action_time = 0
        self.min_action_interval = 0.2  # Minimum time between actions in seconds
        
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
        
    def execute_action(self, action_info: Any) -> Tuple[bool, Dict[str, Any]]:
        """Execute an action based on the provided information.
        
        Args:
            action_info: Action name, dictionary, or index
            
        Returns:
            Tuple of (success status, additional information about the action execution)
        """
        # Add debugging timestamp
        start_time = time.time()
        action_id = str(time.time())[-6:]  # Use last 6 digits of timestamp as action ID
        
        # Apply rate limiting to ensure more consistent input
        time_since_last = start_time - self.last_action_time
        if time_since_last < self.min_action_interval:
            sleep_time = self.min_action_interval - time_since_last
            logger.info(f"[ACTION-{action_id}] Rate limiting: sleeping for {sleep_time:.4f}s")
            time.sleep(sleep_time)
        
        # Store action info for results
        action_results = {
            "action_id": action_id,
            "start_time": start_time,
            "action_info": str(action_info)[:100]  # Truncate long action info
        }
        
        logger.info(f"[ACTION-{action_id}] Starting execution with info: {action_info}")
        
        # Ensure input controllers are available
        if not hasattr(self, 'keyboard') or self.keyboard is None:
            logger.error(f"[ACTION-{action_id}] Keyboard controller not available")
            action_results["error"] = "Keyboard controller not available"
            return False, action_results
            
        if not hasattr(self, 'mouse') or self.mouse is None:
            logger.error(f"[ACTION-{action_id}] Mouse controller not available")
            action_results["error"] = "Mouse controller not available"
            return False, action_results
            
        # Check for and fix any stuck keys before executing a new action
        try:
            # Use our new method if available
            if hasattr(self.keyboard, 'check_stuck_keys'):
                self.keyboard.check_stuck_keys()
        except Exception as e:
            logger.warning(f"[ACTION-{action_id}] Error checking for stuck keys: {e}")
        
        # Handle different action formats
        success = False
        
        try:
            # Dictionary format with type information
            if isinstance(action_info, dict):
                action_type = action_info.get('type')
                
                # Handle speed setting actions
                if action_type == 'speed':
                    speed = action_info.get('speed', 0.5)
                    # Map speed to key actions for game speed
                    if speed <= 0.0:
                        success = self.keyboard.key_press('1', 0.2)
                    elif speed <= 0.25:
                        success = self.keyboard.key_press('1', 0.2)
                    elif speed <= 0.5:
                        success = self.keyboard.key_press('2', 0.2)
                    elif speed <= 0.75:
                        success = self.keyboard.key_press('2', 0.2)
                    else:
                        success = self.keyboard.key_press('3', 0.2)
                    
                    logger.info(f"[ACTION-{action_id}] Set game speed to {speed} with key press: success={success}")
                    action_results["action_type"] = "speed"
                    action_results["speed"] = speed
                
                # Handle keyboard actions
                elif action_type == 'key':
                    key = action_info.get('key')
                    duration = action_info.get('duration', 0.1)
                    
                    # Ensure minimum press duration
                    duration = max(duration, 0.1)
                    
                    logger.info(f"[ACTION-{action_id}] Executing key press for '{key}' with duration {duration}")
                    
                    # Retry mechanism for key presses
                    for attempt in range(3):
                        # Try to ensure key is released first
                        try:
                            self.keyboard.ensure_key_released(key)
                        except Exception as e:
                            logger.warning(f"[ACTION-{action_id}] Error ensuring key '{key}' is released: {e}")
                            
                        success = self.keyboard.key_press(key, duration)
                        if success:
                            logger.info(f"[ACTION-{action_id}] Key press for '{key}' successful on attempt {attempt+1}")
                            break
                        
                        logger.warning(f"[ACTION-{action_id}] Key press for '{key}' failed on attempt {attempt+1}, retrying...")
                        time.sleep(0.2)
                    
                    action_results["action_type"] = "key"
                    action_results["key"] = key
                    action_results["attempts"] = attempt + 1
                
                # Handle mouse actions
                elif action_type == 'mouse':
                    mouse_action = action_info.get('action')
                    button = action_info.get('button', 'left')
                    position = action_info.get('position', None)
                    direction = action_info.get('direction', None)
                    duration = action_info.get('duration', 0.5)
                    
                    action_results["action_type"] = "mouse"
                    action_results["mouse_action"] = mouse_action
                    
                    # Get screen dimensions for position calculations
                    screen_width, screen_height = self._get_screen_dimensions()
                    
                    # Handle click actions
                    if mouse_action in ['click', 'double_click', 'right_click']:
                        # Default to center if no position
                        if not position:
                            position = (0.5, 0.5)
                            logger.info(f"[ACTION-{action_id}] No position for {mouse_action}, using center")
                        
                        # Convert normalized position to screen coordinates
                        x = int(position[0] * screen_width)
                        y = int(position[1] * screen_height)
                        
                        logger.info(f"[ACTION-{action_id}] Mouse {mouse_action} at position ({x}, {y})")
                        
                        # Retry mechanism
                        for attempt in range(3):
                            if mouse_action == 'click':
                                success = self.mouse.mouse_click(x, y, button=button)
                            elif mouse_action == 'double_click':
                                success = self.mouse.mouse_double_click(x, y, button=button)
                            elif mouse_action == 'right_click':
                                success = self.mouse.mouse_click(x, y, button='right')
                            
                            if success:
                                logger.info(f"[ACTION-{action_id}] {mouse_action} at ({x}, {y}) successful on attempt {attempt+1}")
                                break
                                
                            logger.warning(f"[ACTION-{action_id}] {mouse_action} at ({x}, {y}) failed on attempt {attempt+1}, retrying...")
                            time.sleep(0.2)
                        
                        action_results["position"] = (x, y)
                        action_results["attempts"] = attempt + 1
                    
                    # Handle scroll actions
                    elif mouse_action == 'scroll':
                        if not position:
                            position = (0.5, 0.5)
                        
                        x = int(position[0] * screen_width)
                        y = int(position[1] * screen_height)
                        
                        scroll_amount = direction or 1
                        
                        logger.info(f"[ACTION-{action_id}] Mouse scroll at ({x}, {y}) with amount {scroll_amount}")
                        
                        for attempt in range(3):
                            success = self.mouse.mouse_scroll(scroll_amount, x, y)
                            if success:
                                logger.info(f"[ACTION-{action_id}] Scroll at ({x}, {y}) successful on attempt {attempt+1}")
                                break
                                
                            logger.warning(f"[ACTION-{action_id}] Scroll at ({x}, {y}) failed on attempt {attempt+1}, retrying...")
                            time.sleep(0.2)
                        
                        action_results["position"] = (x, y)
                        action_results["scroll_amount"] = scroll_amount
                        action_results["attempts"] = attempt + 1
                    
                    # Handle drag actions
                    elif mouse_action == 'drag':
                        # Need start and end positions
                        start_pos = action_info.get('start_position', position)
                        end_pos = action_info.get('end_position', None)
                        
                        if not start_pos or not end_pos:
                            logger.error(f"[ACTION-{action_id}] Drag action requires both start and end positions")
                            action_results["error"] = "Missing positions for drag"
                            return False, action_results
                        
                        start_x = int(start_pos[0] * screen_width)
                        start_y = int(start_pos[1] * screen_height)
                        end_x = int(end_pos[0] * screen_width)
                        end_y = int(end_pos[1] * screen_height)
                        
                        logger.info(f"[ACTION-{action_id}] Mouse drag from ({start_x}, {start_y}) to ({end_x}, {end_y})")
                        
                        for attempt in range(3):
                            success = self.mouse.mouse_drag(start_x, start_y, end_x, end_y, duration)
                            if success:
                                logger.info(f"[ACTION-{action_id}] Drag successful on attempt {attempt+1}")
                                break
                                
                            logger.warning(f"[ACTION-{action_id}] Drag failed on attempt {attempt+1}, retrying...")
                            time.sleep(0.3)
                        
                        action_results["start_pos"] = (start_x, start_y)
                        action_results["end_pos"] = (end_x, end_y)
                        action_results["attempts"] = attempt + 1
                        
                    # Handle edge scrolling
                    elif mouse_action == 'edge_scroll':
                        edge_direction = action_info.get('direction', 'up')
                        scroll_duration = action_info.get('duration', 0.5)
                        
                        # Map direction to edge positions
                        direction_map = {
                            'up': (screen_width // 2, 5),
                            'down': (screen_width // 2, screen_height - 5),
                            'left': (5, screen_height // 2),
                            'right': (screen_width - 5, screen_height // 2)
                        }
                        
                        if edge_direction not in direction_map:
                            logger.error(f"[ACTION-{action_id}] Invalid edge scroll direction: {edge_direction}")
                            action_results["error"] = f"Invalid edge scroll direction: {edge_direction}"
                            return False, action_results
                        
                        x, y = direction_map[edge_direction]
                        
                        logger.info(f"[ACTION-{action_id}] Edge scroll in direction {edge_direction} at ({x}, {y})")
                        
                        for attempt in range(3):
                            # Move mouse to edge position and hold
                            self.mouse.move_mouse(x, y)
                            time.sleep(0.1)  # Ensure mouse arrived at position
                            success = self.mouse.mouse_click(x, y, hold=True)
                            if success:
                                # Hold for the specified duration
                                time.sleep(scroll_duration)
                                # Release
                                self.mouse.mouse_release(x, y)
                                logger.info(f"[ACTION-{action_id}] Edge scroll successful on attempt {attempt+1}")
                                success = True
                                break
                            
                            logger.warning(f"[ACTION-{action_id}] Edge scroll failed on attempt {attempt+1}, retrying...")
                            time.sleep(0.2)
                        
                        action_results["direction"] = edge_direction
                        action_results["position"] = (x, y)
                        action_results["attempts"] = attempt + 1
                    
                    else:
                        logger.error(f"[ACTION-{action_id}] Unknown mouse action: {mouse_action}")
                        action_results["error"] = f"Unknown mouse action: {mouse_action}"
                        return False, action_results
                
                # Unknown action type
                else:
                    logger.error(f"[ACTION-{action_id}] Unknown action type: {action_type}")
                    action_results["error"] = f"Unknown action type: {action_type}"
                    return False, action_results
            
            # String action name - pass to the actions object if available
            elif isinstance(action_info, str) and self.actions:
                action_name = action_info
                
                # Check if the action exists as a method
                if hasattr(self.actions, action_name):
                    method = getattr(self.actions, action_name)
                    logger.info(f"[ACTION-{action_id}] Executing named action: {action_name}")
                    success = method()
                    action_results["action_type"] = "named_action"
                    action_results["action_name"] = action_name
                else:
                    logger.error(f"[ACTION-{action_id}] Unknown action name: {action_name}")
                    action_results["error"] = f"Unknown action name: {action_name}"
                    return False, action_results
            
            # Integer action - treat as action index from action space
            elif isinstance(action_info, (int, np.integer)):
                # For now, just log this case - would need action space integration to handle properly
                logger.warning(f"[ACTION-{action_id}] Raw action index {action_info} - not directly supported")
                action_results["error"] = f"Raw action index not supported, use action dictionary instead"
                return False, action_results
            
            # Unknown action format
            else:
                logger.error(f"[ACTION-{action_id}] Unsupported action format: {type(action_info)}")
                action_results["error"] = f"Unsupported action format: {type(action_info)}"
                return False, action_results
        
        except Exception as e:
            logger.error(f"[ACTION-{action_id}] Exception during action execution: {e}")
            # Include stack trace for debugging
            import traceback
            logger.error(f"[ACTION-{action_id}] Stack trace: {traceback.format_exc()}")
            action_results["error"] = f"Exception: {str(e)}"
            success = False
        
        # Record final execution time and update last action time
        end_time = time.time()
        execution_time = end_time - start_time
        action_results["execution_time"] = execution_time
        action_results["success"] = success
        
        self.last_action_time = end_time
        
        # Record in action history
        self._record_action(
            str(action_info)[:50], 
            {}, 
            success, 
            execution_time,
            error=action_results.get("error", None)
        )
        
        logger.info(f"[ACTION-{action_id}] Completed execution with success={success} in {execution_time:.3f}s")
        return success, action_results
    
    def _get_screen_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the screen.
        
        Returns:
            Tuple of (width, height)
        """
        if hasattr(self, 'mouse') and self.mouse:
            return self.mouse.screen_width, self.mouse.screen_height
        else:
            # Default resolution if mouse controller not available
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