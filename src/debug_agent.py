"""
Debug script to test window switching and action execution in Cities Skylines 2.

This script will:
1. Find and focus the game window
2. Perform a sequence of mouse and keyboard actions
3. Log all results for debugging
"""

import os
import time
import logging
import sys
from pathlib import Path
import win32api
from environment.optimized_capture import OptimizedScreenCapture
from environment.input.keyboard import KeyboardInput
from environment.input.mouse import MouseInput

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def debug_window_switching_and_actions():
    """Debug window switching and action execution."""
    logger.info("Starting window switching and action debug test")
    
    # Initialize screen capture to find game window
    logger.info("Initializing screen capture with game window search")
    screen = OptimizedScreenCapture(config={
        'window_title': "Cities: Skylines II",
        'search_variations': True,
        'focus_attempts': 3,
        'focus_delay': 0.5
    })
    
    # Initialize input components
    keyboard = KeyboardInput()
    mouse = MouseInput()
    
    # Test window focus
    logger.info("Testing window focus")
    start_time = time.time()
    success = screen.focus_game_window()
    focus_time = time.time() - start_time
    logger.info(f"Window focus {'successful' if success else 'failed'} (took {focus_time:.2f}s)")
    
    if not success:
        logger.error("Failed to focus game window")
        return
    
    # Test action sequence
    logger.info("Starting action sequence test")
    
    # Move mouse to center
    logger.info("Moving mouse to center")
    start_time = time.time()
    success = mouse.mouse_move(960, 540)
    move_time = time.time() - start_time
    logger.info(f"Mouse move {'successful' if success else 'failed'} (took {move_time:.2f}s)")
    
    # Test key presses
    for key in ['w', 'a', 's', 'd']:
        logger.info(f"Testing key press: {key}")
        start_time = time.time()
        success = keyboard.key_press(key)
        press_time = time.time() - start_time
        logger.info(f"Key press {'successful' if success else 'failed'} (took {press_time:.2f}s)")
        time.sleep(0.1)  # Small delay between keys
    
    # Test mouse click
    logger.info("Testing mouse click")
    start_time = time.time()
    success = mouse.mouse_click(960, 540, 'left')
    click_time = time.time() - start_time
    logger.info(f"Mouse click {'successful' if success else 'failed'} (took {click_time:.2f}s)")
    
    # Test tool selection
    logger.info("Testing tool selection")
    start_time = time.time()
    success = keyboard.key_press('1')
    tool_time = time.time() - start_time
    logger.info(f"Tool selection {'successful' if success else 'failed'} (took {tool_time:.2f}s)")
    
    # Get final cursor position
    try:
        final_pos = win32api.GetCursorPos()
        logger.info(f"Final cursor position: {final_pos}")
    except Exception as e:
        logger.error(f"Error getting final cursor position: {e}")

if __name__ == "__main__":
    logger.info("Starting agent debugging")
    debug_window_switching_and_actions()
    logger.info("Debugging completed") 