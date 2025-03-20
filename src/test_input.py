import time
import sys
import logging
import win32api
from src.environment.input.actions import InputSimulator
from src.environment.input.mouse import MouseInput
from src.environment.input.keyboard import KeyboardInput
from src.environment.core.environment import Environment

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_input():
    """Test basic input functionality."""
    logger.info("Starting basic input test")
    
    # Create input directly
    keyboard_input = KeyboardInput()
    mouse_input = MouseInput()
    
    # Also create input simulator for comparison
    input_simulator = InputSimulator()
    
    # Wait for user to position the game window
    logger.info("Please ensure the Cities Skylines 2 window is open and visible")
    logger.info("You have 5 seconds to make the window active...")
    time.sleep(5)
    
    # Press some keys using keyboard input
    logger.info("Testing keyboard input...")
    keys_to_test = ['1', '2', '3', 'w', 'a', 's', 'd']
    for key in keys_to_test:
        logger.info(f"Pressing key: {key}")
        keyboard_input.key_press(key)
        time.sleep(1)
    
    # Get screen dimensions
    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)
    center_x, center_y = screen_width // 2, screen_height // 2
    
    # Try to find the game window
    logger.info("Attempting to find the game window")
    mouse_input.find_game_window("Cities: Skylines II")
    
    # Test mouse movement
    logger.info("Testing mouse movement")
    positions = [
        (center_x - 200, center_y - 200),
        (center_x + 200, center_y - 200),
        (center_x + 200, center_y + 200),
        (center_x - 200, center_y + 200),
        (center_x, center_y)
    ]
    
    for x, y in positions:
        logger.info(f"Moving mouse to ({x}, {y})")
        try:
            mouse_input.mouse_move(x, y)
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error moving mouse: {e}")
    
    # Test mouse click
    logger.info(f"Testing mouse click at screen center ({center_x}, {center_y})")
    try:
        mouse_input.mouse_click(center_x, center_y, 'left')
        time.sleep(1)
    except Exception as e:
        logger.error(f"Error clicking mouse: {e}")
    
    # Test keyboard shortcut and click
    logger.info("Testing tool selection (Key 1)")
    keyboard_input.key_press('1')
    time.sleep(1)
    logger.info("Testing click after tool selection")
    mouse_input.mouse_click(center_x, center_y + 100)
    
    logger.info("Basic input test complete")

def test_environment_focus():
    """Test environment window focusing."""
    logger.info("Testing environment window focusing")
    
    # Create environment with just the focus capability
    env = Environment(skip_game_check=True)
    
    logger.info("Attempting to focus game window...")
    for i in range(3):
        logger.info(f"Focus attempt {i+1}")
        env._ensure_window_focused()
        time.sleep(1)
    
    logger.info("Environment focus test complete")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "focus":
        test_environment_focus()
    else:
        test_basic_input() 