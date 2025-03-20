"""
Error recovery module for Cities Skylines 2 environment.

This module implements error detection and recovery mechanisms to handle
game crashes, freezes, and other errors during training.
"""

import logging
import time
import os
import subprocess
import psutil
from typing import Callable, Optional, Dict, Any, List, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ErrorRecovery:
    """Handles error detection and recovery for the environment."""
    
    def __init__(
        self,
        process_name: str = "CitiesSkylines2",
        game_path: Optional[str] = None,
        max_restart_attempts: int = 3,
        timeout_seconds: int = 120,
        input_simulator = None
    ):
        """Initialize error recovery module.
        
        Args:
            process_name: Name of the game process to monitor
            game_path: Path to the game executable
            max_restart_attempts: Maximum number of restart attempts
            timeout_seconds: Timeout for game start/restart
            input_simulator: Input simulator for menu navigation
        """
        self.process_name = process_name
        self.game_path = game_path
        self.max_restart_attempts = max_restart_attempts
        self.timeout_seconds = timeout_seconds
        self.input_simulator = input_simulator
        
        # Error tracking
        self.restart_attempts = 0
        self.total_restarts = 0
        self.last_restart_time = 0
        self.error_history: List[Dict[str, Any]] = []
        
        # State tracking
        self.last_observation: Optional[np.ndarray] = None
        self.last_observation_time = 0
        self.freeze_count = 0
        self.freeze_threshold = 10  # Number of identical frames to consider a freeze
        
        # Recovery callbacks
        self.pre_restart_callback: Optional[Callable] = None
        self.post_restart_callback: Optional[Callable] = None
        
        logger.info(f"Initialized error recovery module for process '{process_name}'")
    
    def register_callbacks(self, pre_restart: Optional[Callable] = None, post_restart: Optional[Callable] = None):
        """Register callbacks for restart events.
        
        Args:
            pre_restart: Function to call before restarting the game
            post_restart: Function to call after restarting the game
        """
        self.pre_restart_callback = pre_restart
        self.post_restart_callback = post_restart
    
    def check_game_running(self) -> bool:
        """Check if the game process is running.
        
        Returns:
            bool: True if the game is running, False otherwise
        """
        for proc in psutil.process_iter(['name']):
            try:
                if self.process_name.lower() in proc.info['name'].lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False
    
    def check_game_frozen(self, observation: np.ndarray) -> bool:
        """Check if the game appears to be frozen.
        
        Args:
            observation: Current observation from the environment
            
        Returns:
            bool: True if the game appears to be frozen, False otherwise
        """
        if self.last_observation is None:
            self.last_observation = observation
            self.last_observation_time = time.time()
            return False
        
        # Check if the observation has changed
        if np.array_equal(observation, self.last_observation):
            self.freeze_count += 1
        else:
            self.freeze_count = 0
            self.last_observation = observation
            self.last_observation_time = time.time()
        
        # Check if the freeze count exceeds the threshold
        if self.freeze_count >= self.freeze_threshold:
            elapsed = time.time() - self.last_observation_time
            logger.warning(f"Game appears to be frozen (no change in {elapsed:.1f}s, frame count: {self.freeze_count})")
            return True
        
        return False
    
    def focus_game_window(self) -> bool:
        """Explicitly focus the game window.
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Try to import Windows-specific libraries
        try:
            import win32gui
            import win32con
        except ImportError:
            logger.warning("Win32 libraries not available for window focusing")
            return False
            
        try:
            # Find the game window
            hwnd = win32gui.FindWindow(None, self.process_name)
            if hwnd == 0:
                # Try partial match if exact name not found
                def callback(hwnd, windows):
                    window_title = win32gui.GetWindowText(hwnd)
                    if self.process_name in window_title or "Cities: Skylines II" in window_title:
                        windows.append(hwnd)
                    return True
                    
                windows = []
                win32gui.EnumWindows(callback, windows)
                
                if windows:
                    hwnd = windows[0]
            
            if hwnd == 0:
                logger.warning(f"Game window '{self.process_name}' not found")
                return False
                
            # Check if window is minimized
            if win32gui.IsIconic(hwnd):
                # Show the window if it's minimized
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                
            # Check if window is already in foreground
            foreground_hwnd = win32gui.GetForegroundWindow()
            if foreground_hwnd == hwnd:
                logger.debug("Game window is already in focus")
                return True
                
            # Set window as foreground window
            result = win32gui.SetForegroundWindow(hwnd)
            
            # Try setting focus too as a fallback
            try:
                win32gui.SetFocus(hwnd)
            except Exception:
                pass
                
            logger.info(f"Set game window to foreground manually (result: {result})")
            
            # Give OS time to process the focus change
            time.sleep(0.5)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to focus game window: {e}")
            return False
    
    def restart_game(self) -> bool:
        """Restart the game process.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.restart_attempts >= self.max_restart_attempts:
            logger.error(f"Maximum restart attempts ({self.max_restart_attempts}) reached")
            return False
            
        self.restart_attempts += 1
        self.total_restarts += 1
        self.last_restart_time = time.time()
        
        # Track error
        self.error_history.append({
            'timestamp': self.last_restart_time,
            'type': 'restart',
            'attempt': self.restart_attempts
        })
            
        logger.warning(f"Attempting to restart game (attempt {self.restart_attempts}/{self.max_restart_attempts})")
        
        try:
            # Call pre-restart callback if registered
            if hasattr(self, 'pre_restart_callback') and self.pre_restart_callback:
                try:
                    self.pre_restart_callback()
                except Exception as e:
                    logger.error(f"Error in pre-restart callback: {e}")
            
            # Kill any existing game processes
            self._kill_game_processes()
            
            # Start the game
            if not self._start_game():
                logger.error("Failed to start game")
                return False
                
            # Wait for the game to initialize
            logger.info(f"Waiting for game to initialize (timeout: {self.timeout_seconds}s)")
            start_time = time.time()
            
            # Give game time to start
            time.sleep(10)
            
            # Try to find and focus the game window
            self.focus_game_window()
            
            # Wait a bit more for game to be fully initialized
            time.sleep(5)
            
            # Check if game is running
            if not self.check_game_running():
                logger.error("Game failed to start")
                return False
                
            # Call post-restart callback if registered
            if hasattr(self, 'post_restart_callback') and self.post_restart_callback:
                try:
                    self.post_restart_callback()
                except Exception as e:
                    logger.error(f"Error in post-restart callback: {e}")
            
            # Reset restart attempts on successful restart
            self.restart_attempts = 0
            logger.info("Game successfully restarted")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during game restart: {e}")
            return False
    
    def handle_menu_detection(self) -> bool:
        """Handle menu detection and navigation.
        
        Returns:
            bool: True if successfully handled menu, False otherwise
        """
        if self.input_simulator is None:
            logger.warning("No input simulator available for menu recovery")
            return False
        
        try:
            # Try to recover from menu state
            logger.info("Attempting to recover from menu state")
            self.input_simulator.handle_menu_recovery(retries=2)
            return True
        except Exception as e:
            logger.error(f"Failed to recover from menu state: {e}")
            
            # Record menu recovery failure
            self.error_history.append({
                'time': time.time(),
                'type': 'menu_recovery',
                'success': False,
                'error': str(e)
            })
            
            return False
    
    def reset_error_state(self):
        """Reset error tracking state."""
        self.restart_attempts = 0
        self.freeze_count = 0
        self.last_observation = None
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get statistics about errors and recovery attempts.
        
        Returns:
            Dict with error statistics
        """
        return {
            'total_restarts': self.total_restarts,
            'last_restart_time': self.last_restart_time,
            'restart_attempts': self.restart_attempts,
            'freeze_count': self.freeze_count,
            'error_history_size': len(self.error_history),
            'recent_errors': self.error_history[-5:] if self.error_history else []
        }
    
    def _kill_game_processes(self):
        """Kill any running game processes."""
        for proc in psutil.process_iter(['name', 'pid']):
            try:
                if self.process_name.lower() in proc.info['name'].lower():
                    logger.info(f"Killing process {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logger.warning(f"Error killing process: {e}")
        
        # Wait for processes to terminate
        time.sleep(5)
    
    def _start_game(self) -> bool:
        """Start the game process.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.game_path:
            logger.error("No game path provided, cannot restart game")
            return False
        
        try:
            # Start the game
            game_path = Path(self.game_path)
            logger.info(f"Starting game: {game_path}")
            
            if not game_path.exists():
                logger.error(f"Game executable not found: {game_path}")
                return False
            
            subprocess.Popen([str(game_path)], shell=True)
            
            # Wait for the game to start
            start_time = time.time()
            while time.time() - start_time < self.timeout_seconds:
                if self.check_game_running():
                    # Game is running, wait a bit for it to initialize
                    logger.info(f"Game started successfully, waiting for initialization")
                    time.sleep(10)
                    return True
                time.sleep(1)
            
            logger.error(f"Game failed to start within {self.timeout_seconds} seconds")
            return False
            
        except Exception as e:
            logger.error(f"Error starting game: {e}")
            return False 