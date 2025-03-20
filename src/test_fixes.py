"""
Test script to verify the fixes made to the Cities Skylines 2 agent environment.

This script provides test functions for:
1. Window focus handling
2. Mouse input operations
3. Action execution
4. Full environment step execution
"""

import os
import time
import logging
import win32gui
import win32api
import argparse
import numpy as np
from pathlib import Path
import random
import sys

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(Path("logs/test_fixes.log")),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

# Add src to path if running directly
if __name__ == '__main__' and not __package__:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)

# Import environment components
from src.environment.input.keyboard import KeyboardInput
from src.environment.input.mouse import MouseInput
from src.environment.input.actions import InputActions, ActionExecutor, InputSimulator
from src.environment.optimized_capture import OptimizedScreenCapture
from src.environment.core.environment import Environment

def test_window_focus(window_title="Cities: Skylines II"):
    """Test window focus functionality."""
    logger.info(f"Testing window focus for '{window_title}'")
    
    # Initialize screen capture
    screen_capture = OptimizedScreenCapture(config={
        'window_title': window_title,
        'search_variations': True
    })
    
    # Try focusing the window
    result = screen_capture.focus_game_window()
    logger.info(f"Window focus result: {result}")
    
    # Get window info if available
    if hasattr(screen_capture, 'game_rect') and screen_capture.game_rect:
        logger.info(f"Window rectangle: {screen_capture.game_rect}")
        width = screen_capture.game_rect[2] - screen_capture.game_rect[0]
        height = screen_capture.game_rect[3] - screen_capture.game_rect[1]
        logger.info(f"Window size: {width}x{height}")
    elif hasattr(screen_capture, 'window_rect') and screen_capture.window_rect:
        logger.info(f"Window rectangle: {screen_capture.window_rect}")
        width = screen_capture.window_rect[2] - screen_capture.window_rect[0]
        height = screen_capture.window_rect[3] - screen_capture.window_rect[1]
        logger.info(f"Window size: {width}x{height}")
    
    return result

def test_mouse_input(no_clicks=False):
    """Test mouse input operations."""
    logger.info("Testing mouse input")
    
    # Initialize mouse input
    mouse = MouseInput()
    
    # Test move to center
    center_x, center_y = mouse.screen_width // 2, mouse.screen_height // 2
    logger.info(f"Moving mouse to center: ({center_x}, {center_y})")
    move_result = mouse.mouse_move(center_x, center_y)
    logger.info(f"Mouse move result: {move_result}")
    
    # Test random movement
    for i in range(5):
        x = random.randint(100, mouse.screen_width - 100)
        y = random.randint(100, mouse.screen_height - 100)
        logger.info(f"Moving mouse to random position {i+1}: ({x}, {y})")
        mouse.mouse_move(x, y)
        time.sleep(0.2)
    
    # Test click if not disabled
    if not no_clicks:
        logger.info("Testing mouse click")
        click_result = mouse.mouse_click(center_x, center_y)
        logger.info(f"Mouse click result: {click_result}")
        
        # Test right click
        logger.info("Testing right click")
        right_click_result = mouse.mouse_click(
            center_x + 100, 
            center_y, 
            button='right'
        )
        logger.info(f"Right click result: {right_click_result}")
    
    # Test drag
    logger.info("Testing mouse drag")
    start_x, start_y = center_x - 200, center_y
    end_x, end_y = center_x + 200, center_y
    drag_result = mouse.mouse_drag((start_x, start_y), (end_x, end_y))
    logger.info(f"Drag result: {drag_result}")
    
    return True

def test_keyboard_input():
    """Test keyboard input operations."""
    logger.info("Testing keyboard input")
    
    # Initialize keyboard input
    keyboard = KeyboardInput()
    
    # Test some key presses (non-UI affecting keys)
    test_keys = ['a', 's', 'd', 'f']
    
    for key in test_keys:
        logger.info(f"Testing key press for '{key}'")
        result = keyboard.key_press(key, duration=0.1)
        logger.info(f"Key press result for {key}: {result}")
        time.sleep(0.3)
    
    return True

def test_action_execution():
    """Test action execution functionality."""
    logger.info("Testing action execution")
    
    # Initialize input components
    keyboard = KeyboardInput()
    mouse = MouseInput()
    
    # Test direct key presses
    test_keys = ['1', '2', '3', 'w', 'a', 's', 'd', 'escape']
    success_count = 0
    
    logger.info("Testing key press actions")
    for key in test_keys:
        logger.info(f"Testing key press for '{key}'")
        
        # Test through action format
        action = {'type': 'key', 'key': key, 'duration': 0.1}
        action_executor = ActionExecutor(keyboard, mouse)
        action_result = action_executor.execute_action(action)
        logger.info(f"Action result for key {key}: {action_result}")
        
        # Test direct key press
        direct_result = keyboard.key_press(key, 0.1)
        logger.info(f"Direct key press result for key {key}: {direct_result}")
        
        if direct_result:
            success_count += 1
        
        time.sleep(0.5)
    
    logger.info(f"Key press test completed: {success_count}/{len(test_keys)} successful ({success_count/len(test_keys)*100:.1f}%)")
    
    # Test mouse actions
    logger.info("Testing mouse actions")
    action_executor = ActionExecutor(keyboard, mouse)
    
    # Test mouse click action
    mouse_action = {
        'type': 'mouse', 
        'action': 'click', 
        'position': (0.5, 0.5)
    }
    mouse_result = action_executor.execute_action(mouse_action)
    logger.info(f"Mouse click action result: {mouse_result}")
    
    time.sleep(0.5)
    
    # Test direct click action
    direct_result = action_executor.execute_action('click', x=960, y=540)
    logger.info(f"Direct click action result: {direct_result}")
    
    # Test mouse drag
    start = (mouse.screen_width//2 - 100, mouse.screen_height//2)
    end = (mouse.screen_width//2 + 100, mouse.screen_height//2)
    drag_result = mouse.mouse_drag(start, end)
    logger.info(f"Mouse drag result: {drag_result}")
    
    logger.info("")
    logger.info("All tests completed")
    
    return True

def test_environment_step():
    """Test environment step function."""
    logger.info("Testing environment step function")
    
    # Create environment with simpler initialization that doesn't require HardwareConfig
    env = Environment(
        mock_mode=False,
        window_title="Cities: Skylines II",
        skip_game_check=True,
        disable_menu_detection=True
    )
    
    # Initialize environment
    env.reset()
    
    # Test a few random actions
    for i in range(5):
        action = random.choice([
            {'type': 'key', 'key': '1'},
            {'type': 'key', 'key': '2'},
            {'type': 'mouse', 'action': 'move', 'position': (0.5, 0.5)},
            {'type': 'mouse', 'action': 'click', 'position': (0.5, 0.5)},
        ])
        
        logger.info(f"Step {i+1}: Executing action {action}")
        observation, reward, done, info = env.step(action)
        logger.info(f"Step result: reward={reward}, done={done}")
        
        if 'error' in info:
            logger.warning(f"Error in step: {info['error']}")
        
        time.sleep(1)
    
    # Close environment
    env.close()
    logger.info("Environment test completed")
    
    return True

def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description='Test fix implementation.')
    parser.add_argument('--focus', action='store_true', help='Test window focus')
    parser.add_argument('--mouse', action='store_true', help='Test mouse input')
    parser.add_argument('--keyboard', action='store_true', help='Test keyboard input')
    parser.add_argument('--action', action='store_true', help='Test action execution')
    parser.add_argument('--step', action='store_true', help='Test environment step')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--no-clicks', action='store_true', help='Disable mouse clicks in tests')
    
    args = parser.parse_args()
    
    # If no specific tests are specified, print help
    if not any([args.focus, args.mouse, args.keyboard, args.action, args.step, args.all]):
        parser.print_help()
        sys.exit(1)
    
    # Run tests based on arguments
    if args.all or args.focus:
        test_window_focus()
        
    if args.all or args.mouse:
        test_mouse_input(args.no_clicks)
        
    if args.all or args.keyboard:
        test_keyboard_input()
        
    if args.all or args.action:
        test_action_execution()
        
    if args.all or args.step:
        test_environment_step()

if __name__ == "__main__":
    main() 