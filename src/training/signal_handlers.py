"""
Signal handling for training process.

This module provides functions for handling various signals like SIGINT, SIGTERM.
"""

import signal
import logging
import sys
import atexit
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Global flag to track exit request state
_exit_requested = False

def setup_signal_handlers(
    cleanup_function: Optional[Callable] = None,
    save_function: Optional[Callable] = None
):
    """Set up signal handlers for graceful termination.
    
    Args:
        cleanup_function: Function to call for cleanup at exit
        save_function: Function to call to save progress at exit
    """
    global _exit_requested
    
    def request_exit(signum, frame):
        """Signal handler that sets the exit flag."""
        global _exit_requested
        
        if _exit_requested:
            logger.warning("Received second exit signal, forcing immediate exit")
            sys.exit(1)
            
        logger.info(f"Received signal {signum}, gracefully stopping...")
        _exit_requested = True
        
        # If save function provided, try to save before exiting
        if save_function:
            try:
                logger.info("Attempting to save progress before exit...")
                save_function()
                logger.info("Progress saved successfully")
            except Exception as e:
                logger.error(f"Failed to save progress: {e}")
    
    def exit_handler():
        """Function called at exit to perform cleanup."""
        if not _exit_requested:
            logger.info("Exit detected, running cleanup...")
            _exit_requested = True
            
        # Call the cleanup function if provided
        if cleanup_function:
            try:
                logger.info("Running cleanup tasks...")
                cleanup_function()
                logger.info("Cleanup completed")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    # Register the exit handler
    atexit.register(exit_handler)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, request_exit)  # Ctrl+C
    signal.signal(signal.SIGTERM, request_exit)  # Termination signal
    
    # If on Windows, handle CTRL_CLOSE_EVENT and CTRL_BREAK_EVENT
    try:
        # Windows-specific signals
        signal.signal(signal.CTRL_CLOSE_EVENT, request_exit)
        signal.signal(signal.CTRL_BREAK_EVENT, request_exit)
    except (AttributeError, ValueError, OSError):
        # These signals might not be available on all platforms
        pass
    
    logger.info("Signal handlers installed")

def create_autosave_thread(
    save_function: Callable,
    interval_minutes: float = 15,
    name: str = "AutosaveThread"
) -> threading.Thread:
    """Create a thread that periodically calls the save function.
    
    Args:
        save_function: Function to call periodically
        interval_minutes: Interval in minutes between calls
        name: Name for the thread
        
    Returns:
        Thread object that can be started
    """
    def autosave_worker():
        """Worker function for the autosave thread."""
        logger.info(f"Autosave thread started with {interval_minutes} minute interval")
        
        while not _exit_requested:
            # Sleep for the specified interval
            for _ in range(int(interval_minutes * 60 / 10)):
                if _exit_requested:
                    break
                time.sleep(10)
            
            if not _exit_requested:
                try:
                    logger.info("Performing autosave...")
                    save_function()
                    logger.info("Autosave completed")
                except Exception as e:
                    logger.error(f"Error during autosave: {e}")
        
        logger.info("Autosave thread exiting")
    
    thread = threading.Thread(target=autosave_worker, name=name, daemon=True)
    return thread

def is_exit_requested() -> bool:
    """Check if an exit has been requested via signals.
    
    Returns:
        Whether an exit has been requested
    """
    return _exit_requested

def request_exit():
    """Manually request an exit."""
    global _exit_requested
    _exit_requested = True 