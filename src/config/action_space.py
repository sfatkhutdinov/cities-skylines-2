"""Action space definition for the agent."""

from typing import Dict, Any, List

def get_action_space() -> Dict[int, Dict[str, Any]]:
    """Get the primitive action space for the agent.
    
    This defines a low-level action space that only contains primitive inputs
    (keypresses, mouse movements, mouse clicks) without any game-specific knowledge.
    The agent will need to learn which combinations of these actions are useful.
    
    Returns:
        Dict[int, Dict[str, Any]]: Dictionary of action indices mapped to action details
    """
    action_space = {}
    action_idx = 0
    
    # === KEYBOARD ACTIONS ===
    # Generate alphabet key presses (a-z)
    for key in "abcdefghijklmnopqrstuvwxyz":
        action_space[action_idx] = {"type": "key", "key": key}
        action_idx += 1
    
    # Number keys (0-9)
    for key in "0123456789":
        action_space[action_idx] = {"type": "key", "key": key}
        action_idx += 1
    
    # Function keys (F1-F12)
    for i in range(1, 13):
        action_space[action_idx] = {"type": "key", "key": f"f{i}"}
        action_idx += 1
    
    # Special keys
    special_keys = [
        "escape", "space", "tab", "enter", "backspace",
        "shift", "ctrl", "alt", 
        "left", "right", "up", "down",
        "page_up", "page_down", "home", "end",
        "delete", "insert",
        "`", "-", "=", "[", "]", "\\", ";", "'", ",", ".", "/"
    ]
    
    for key in special_keys:
        action_space[action_idx] = {"type": "key", "key": key}
        action_idx += 1
    
    # === MOUSE ACTIONS ===
    # Grid of mouse positions (5x5 grid)
    grid_size = 5
    for y in range(grid_size):
        for x in range(grid_size):
            # Normalized position (0.0 to 1.0)
            norm_x = x / (grid_size - 1)
            norm_y = y / (grid_size - 1)
            action_space[action_idx] = {
                "type": "mouse", 
                "action": "move", 
                "x": norm_x, 
                "y": norm_y
            }
            action_idx += 1
    
    # Mouse clicks
    buttons = ["left", "right", "middle"]
    click_types = [
        {"action": "click", "double": False},
        {"action": "click", "double": True},
        {"action": "down", "double": False},
        {"action": "up", "double": False},
    ]
    
    for button in buttons:
        for click_type in click_types:
            action_space[action_idx] = {
                "type": "mouse",
                "action": click_type["action"],
                "button": button,
                "double": click_type.get("double", False)
            }
            action_idx += 1
    
    # Mouse scroll
    scroll_directions = ["up", "down"]
    for direction in scroll_directions:
        action_space[action_idx] = {
            "type": "mouse",
            "action": "scroll",
            "direction": direction,
            "amount": 5  # Default scroll amount
        }
        action_idx += 1
    
    # Mouse drag (from current position to grid points)
    for y in range(grid_size):
        for x in range(grid_size):
            # Normalized position (0.0 to 1.0)
            norm_x = x / (grid_size - 1)
            norm_y = y / (grid_size - 1)
            action_space[action_idx] = {
                "type": "mouse", 
                "action": "drag", 
                "to_x": norm_x, 
                "to_y": norm_y
            }
            action_idx += 1
    
    # === COMBINED ACTIONS ===
    # Key + mouse for common combinations
    common_key_combos = [
        {"key": "shift", "button": "left"},
        {"key": "ctrl", "button": "left"},
        {"key": "alt", "button": "left"},
    ]
    
    for combo in common_key_combos:
        action_space[action_idx] = {
            "type": "combined",
            "key": combo["key"],
            "mouse_action": "click",
            "button": combo["button"]
        }
        action_idx += 1
    
    # Wait action (do nothing)
    action_space[action_idx] = {"type": "wait", "duration": 0.5}
    action_idx += 1
    
    return action_space 