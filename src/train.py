import os
import sys
import time
import glob
import signal
import logging
import threading
import argparse
import traceback
import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import wandb
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from pynput import keyboard

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from environment.game_env import CitiesEnvironment
from agent.ppo_agent import PPOAgent
from config.hardware_config import HardwareConfig

# Define helper functions
def current_timestamp():
    """Return a formatted timestamp string for the current time."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Add a watchdog timer to force-kill the program if it becomes unresponsive
class WatchdogTimer:
    """Watchdog timer that forcibly kills the program if it becomes unresponsive"""
    def __init__(self, timeout=5):
        self.timeout = timeout
        self.active = False
        self.reset_event = threading.Event()
        self.thread = None
        
    def start(self):
        """Start the watchdog timer"""
        if self.thread and self.thread.is_alive():
            return  # Already running
            
        self.active = True
        self.reset_event.clear()
        self.thread = threading.Thread(target=self._watchdog_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Watchdog timer started (timeout: {self.timeout}s)")
        
    def _watchdog_loop(self):
        """Main watchdog loop"""
        while self.active:
            # Wait for the reset event or timeout
            triggered = not self.reset_event.wait(self.timeout)
            if triggered and self.active:
                # Timeout occurred and watchdog is still active
                logger.critical("WATCHDOG TIMEOUT - Program appears frozen. Forcibly terminating!")
                os._exit(2)  # Force terminate with error code
            self.reset_event.clear()
    
    def reset(self):
        """Reset the watchdog timer - must be called regularly to prevent termination"""
        self.reset_event.set()
        
    def stop(self):
        """Stop the watchdog timer"""
        self.active = False
        self.reset_event.set()  # Wake up the thread to exit
        if self.thread:
            self.thread.join(1.0)  # Wait for thread to exit, with timeout

# Start a global watchdog timer that will be reset in the main loop
watchdog = WatchdogTimer(timeout=15)  # 15-second timeout

# Add key listener for emergency shutdown
def setup_emergency_key_listener():
    """Set up a keyboard listener to detect Ctrl+C and other emergency exit key combinations"""
    try:
        from pynput import keyboard
        
        def on_press(key):
            # Check for Ctrl+C combination
            try:
                # Handle Ctrl+C (both as Key.ctrl + 'c' and directly as keyboard.Key.ctrl_l + 'c')
                if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                    # Track Ctrl press
                    on_press.ctrl_pressed = True
                elif hasattr(key, 'char') and key.char == 'c' and hasattr(on_press, 'ctrl_pressed') and on_press.ctrl_pressed:
                    # Ctrl+C detected
                    logger.info("Keyboard hook detected Ctrl+C - initiating emergency shutdown")
                    signal_handler(signal.SIGINT, None)
                # Also detect Escape key for emergency exit
                elif key == keyboard.Key.esc:
                    logger.info("Keyboard hook detected ESC key - initiating emergency shutdown")
                    signal_handler(signal.SIGINT, None)
            except Exception as e:
                logger.error(f"Error in keyboard hook: {e}")
                
        def on_release(key):
            # Track the release of Ctrl key
            if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                on_press.ctrl_pressed = False
                
        # Initialize Ctrl press tracking
        on_press.ctrl_pressed = False
        
        # Start keyboard listener in non-blocking mode
        keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        keyboard_listener.daemon = True  # Make it a daemon so it doesn't block program exit
        keyboard_listener.start()
        logger.info("Emergency keyboard hook activated")
        
        return keyboard_listener
    except Exception as e:
        logger.warning(f"Failed to set up emergency key listener: {e}")
        return None

# Set up emergency key listener
emergency_listener = setup_emergency_key_listener()

# Set up Windows-specific Ctrl+C handler (more reliable than signals on Windows)
import platform
if platform.system() == 'Windows':
    try:
        import win32api
        import win32con
        
        # Windows-specific handler for Ctrl+C events
        def windows_ctrl_handler(ctrl_type):
            if ctrl_type == win32con.CTRL_C_EVENT or ctrl_type == win32con.CTRL_BREAK_EVENT:
                logger.info("Windows Ctrl+C event detected")
                signal_handler(signal.SIGINT, None)  # Call our standard handler
                return True  # Indicate we've handled it
            return False
            
        # Register the Windows handler
        win32api.SetConsoleCtrlHandler(windows_ctrl_handler, True)
        logger.info("Windows-specific Ctrl+C handler registered")
    except Exception as e:
        logger.warning(f"Failed to register Windows Ctrl+C handler: {e}")

logger.debug("Starting script...")

try:
    logger.debug("Importing environment...")
    from environment.game_env import CitiesEnvironment
    logger.debug("Environment imported successfully")
    
    logger.debug("Importing agent...")
    from agent.ppo_agent import PPOAgent
    logger.debug("Agent imported successfully")
    
    logger.debug("Importing config...")
    from config.hardware_config import HardwareConfig
    logger.debug("Config imported successfully")
except Exception as e:
    logger.error(f"Error during imports: {str(e)}")
    raise

from typing import Dict, List, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description="Train Cities: Skylines 2 RL agent")
    parser.add_argument("--num_episodes", type=int, default=10000, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=3000, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updates")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=100, help="Episodes between checkpoints")
    parser.add_argument("--time_checkpoint_freq", type=int, default=30, help="Minutes between time-based checkpoints")
    parser.add_argument("--step_checkpoint_freq", type=int, default=10000, help="Steps between step-based checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--mock", action="store_true", help="Use mock environment for training without the actual game")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--resolution", type=str, default="320x240", help="Resolution for screen capture (WxH)")
    parser.add_argument("--action_delay", type=float, default=0.5, help="Minimum delay between actions (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--resume_best", action="store_true", help="Resume training from best checkpoint")
    parser.add_argument("--no_auto_resume", action="store_true", help="Disable automatic checkpoint resumption")
    parser.add_argument("--menu_screenshot", type=str, default=None, help="Path to a screenshot of the menu for reference-based detection")
    parser.add_argument("--capture_menu", action="store_true", help="Capture a menu screenshot at startup (assumes you're starting from the menu)")
    return parser.parse_args()

# Global variable for graceful interruption
stop_training = False

# Signal handler for graceful interruption
def signal_handler(sig, frame):
    global stop_training
    logger.info("Received interrupt signal (CTRL+C). Will save checkpoint and exit after current episode.")
    stop_training = True

    # Emergency release of all input devices
    try:
        import win32api
        from pynput.keyboard import Controller as KeyboardController, Key
        from pynput.mouse import Controller as MouseController, Button
        import ctypes
        import os
        
        logger.info("EMERGENCY SHUTDOWN - Releasing all input controls")
        
        # Release any keys that might be pressed
        keyboard = KeyboardController()
        # First try specific key release
        try:
            # Release common game keys
            for key in ['w', 'a', 's', 'd', 'q', 'e', 'r', 'f', 'escape', 'space']:
                keyboard.release(key)
            # Release modifier keys
            keyboard.release(Key.shift)
            keyboard.release(Key.ctrl)
            keyboard.release(Key.alt)
        except:
            pass
        
        # Then try to release all pressed keys the pynput way
        try:
            for key in keyboard._pressed:
                keyboard.release(key)
        except:
            pass
        
        # Release mouse buttons
        mouse = MouseController()
        try:
            mouse.release(Button.left)
            mouse.release(Button.right)
            mouse.release(Button.middle)
        except:
            pass
        
        # Try to block/unblock input as a last resort (Windows specific)
        try:
            # Block inputs temporarily to clear any stuck inputs
            ctypes.windll.user32.BlockInput(True)
            time.sleep(0.1)
            ctypes.windll.user32.BlockInput(False)
        except:
            pass
            
        # Reset mouse position to center of the screen
        try:
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            win32api.SetCursorPos((screen_width // 2, screen_height // 2))
        except:
            pass
            
        logger.info("Input controls released - FORCIBLY TERMINATING in 1 second")
        
        # Force terminate after a short delay to allow logging to complete
        def force_exit():
            time.sleep(1)
            os._exit(1)  # Force terminate the process
            
        # Start a thread to force exit
        exit_thread = threading.Thread(target=force_exit)
        exit_thread.daemon = True
        exit_thread.start()
        
    except Exception as e:
        logger.error(f"Failed to release input controls: {e}")
        # Force terminate anyway
        os._exit(1)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def collect_trajectory(
    env: CitiesEnvironment,
    agent: PPOAgent,
    max_steps: int,
    device: torch.device,
    args: argparse.Namespace
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, Dict]:
    """Collect a trajectory of experience."""
    # Reset watchdog timer at the start of trajectory collection
    if 'watchdog' in globals():
        watchdog.reset()
    
    # Initialize storage
    states = []
    next_states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    total_reward = 0.0
    last_info = {}
    
    # Get initial state and create frame history
    state = env.reset()
    frame_history = [state.clone() for _ in range(4)]  # Always use 4 frames for stacking
    
    # Create stacked initial state
    stacked_state = torch.cat(frame_history, dim=0)
    
    # Loop through steps
    for step in range(max_steps):
        # Make sure to reset the watchdog every iteration to prevent false triggers
        if 'watchdog' in globals():
            watchdog.reset()
        
        # Select action using current policy
        action, log_prob, value = agent.select_action(stacked_state.unsqueeze(0))
        
        # Execute action in environment
        next_state, reward, done, info = env.step(action.item())
        last_info = info  # Store the last info dictionary
        total_reward += reward
        
        # Update frame history for next state
        frame_history.pop(0)
        frame_history.append(next_state.clone())
        stacked_next_state = torch.cat(frame_history, dim=0)
        
        # Store transition
        states.append(stacked_state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        next_states.append(stacked_next_state)
        
        # Update current state
        stacked_state = stacked_next_state
        
        # End episode if done
        if done:
            break
    
    return states, actions, log_probs, rewards, values, dones, next_states, total_reward, last_info

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in the given directory."""
    # Check both episode and step-based checkpoints
    episode_checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
    step_checkpoints = list(checkpoint_dir.glob("step_checkpoint_*.pt"))
    time_checkpoints = list(checkpoint_dir.glob("time_checkpoint_*.pt"))
    
    all_checkpoints = []
    
    # Process episode checkpoints
    for cp_file in episode_checkpoints:
        try:
            episode = int(cp_file.stem.split('_')[1])
            timestamp = cp_file.stat().st_mtime
            all_checkpoints.append((timestamp, episode, cp_file, "episode"))
        except ValueError:
            continue
    
    # Process step checkpoints
    for cp_file in step_checkpoints:
        try:
            step = int(cp_file.stem.split('_')[2])
            timestamp = cp_file.stat().st_mtime
            all_checkpoints.append((timestamp, step, cp_file, "step"))
        except ValueError:
            continue
    
    # Process time checkpoints
    for cp_file in time_checkpoints:
        try:
            timestamp = cp_file.stat().st_mtime
            all_checkpoints.append((timestamp, 0, cp_file, "time"))
        except:
            continue
    
    if not all_checkpoints:
        return None
    
    # Sort by timestamp (most recent first)
    all_checkpoints.sort(reverse=True, key=lambda x: x[0])
    latest = all_checkpoints[0]
    
    if latest[3] == "episode":
        return (latest[1], latest[2])  # Return (episode_number, file_path)
    else:
        # For step or time checkpoints, return episode 0 since we don't know the episode
        return (0, latest[2])  # Return (0, file_path)

# For Windows systems, add direct process exit handler
def setup_windows_exit_handler():
    """
    Setup a direct exit handler for Windows systems using Win32 API.
    This ensures the program can be forcibly stopped with Ctrl+C or Ctrl+Break.
    """
    try:
        import win32api
        
        def handle_win32_exit(sig, hook_type):
            """Handle Windows Ctrl events directly using Win32 API"""
            logger.info(f"Windows Ctrl event received: {sig} - Shutting down program immediately")
            if 'watchdog' in globals():
                watchdog.stop()
                
            # Force immediate exit to prevent hanging
            os._exit(1)
            return 1  # True in Win32 API to indicate the signal was handled
        
        # Register handlers for Ctrl+C (0), Ctrl+Break (1), and Close (2)
        win32api.SetConsoleCtrlHandler(handle_win32_exit, True)
        logger.info("Windows-specific exit handler installed")
    except ImportError:
        logger.warning("win32api not available, Windows-specific exit handler not installed")

# Exit flag for signaling termination
exit_flag = threading.Event()

def handle_exit_signal(signum, frame):
    """Handle exit signals (SIGINT, SIGTERM) by setting the exit flag"""
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    logger.info(f"Received signal {sig_name}, initiating graceful shutdown...")
    exit_flag.set()
    # Give the program 5 seconds to exit gracefully, then force exit
    threading.Timer(5.0, lambda: os._exit(1)).start()

# Setup signal handlers
try:
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)
    logger.info("Standard signal handlers installed")
except (ValueError, AttributeError) as e:
    logger.warning(f"Failed to set up signal handlers: {e}")

def main():
    """Main training loop"""
    args = parse_args()
    
    # Setup Windows-specific exit handler
    if sys.platform.startswith('win'):
        setup_windows_exit_handler()
    
    # Start the watchdog timer at the beginning of main
    if 'watchdog' in globals():
        watchdog.start()
    
    try:
        # Create hardware config with specific device & dtype
        config = HardwareConfig()
        
        # Set device based on command line args
        if args.device is not None:
            config.device = args.device
            
        # Apply batch size override if provided
        if args.batch_size is not None:
            config.batch_size = args.batch_size
            
        # Apply step checkpoint frequency if provided
        if args.step_checkpoint_freq is not None:
            config.step_checkpoint_freq = args.step_checkpoint_freq
        
        # Configuration log
        logger.info(f"Using device: {config.device}")
        # GPU information is already logged in the HardwareConfig initialization
        
        # Create environment
        logger.info("Initializing environment...")
        
        # Default resolution
        if args.resolution:
            width, height = map(int, args.resolution.split('x'))
            config.resolution = (width, height)
            
        # Create environment with command line options
        env = CitiesEnvironment(
            config=config, 
            mock_mode=args.mock,
            menu_screenshot_path=args.menu_screenshot
        )
        
        # Create agent
        logger.info("Initializing agent...")
        agent = PPOAgent(config)
        
        # Ensure checkpoint directory exists
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Try to capture menu screenshot if requested
        if args.capture_menu:
            logger.info("Attempting to capture menu screenshot...")
            env.capture_menu_reference("menu_reference.png")
            
        # Set up model compilation if PyTorch version supports it (2.0+)
        if hasattr(torch, 'compile') and hasattr(agent.network, 'model'):
            logger.info("Applying PyTorch 2.0 model compilation")
            try:
                agent.network.model = torch.compile(agent.network.model)
            except Exception as e:
                logger.warning(f"Module compilation failed: {e}")
        
        # Resume from checkpoints if available and not explicitly disabled
        load_checkpoint = not args.no_auto_resume
        use_best = args.resume_best
        if load_checkpoint:
            logger.info(f"Looking for checkpoints to resume training (use_best={use_best})...")
            agent.load_checkpoint(use_best=use_best)
        
        # Initialize episode counter and best reward tracking
        start_episode = 0
        best_reward = float('-inf')
        last_time_checkpoint = time.time()
        
        logger.info(f"Starting training with model dimensions: {agent.network.expected_width}x{agent.network.expected_height}")
        logger.info(f"Frame stack size: {config.frame_stack}")
        
        # Initialize wandb logging if available
        use_wandb = False
        try:
            import wandb
            use_wandb = True
            logger.info("Initializing wandb for experiment tracking")
            wandb.init(
                project="cs2-agent",
                config={
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "device": args.device,
                    "resolution": args.resolution,
                    "max_steps": args.max_steps,
                    "frame_stack": config.frame_stack
                }
            )
        except ImportError:
            logger.warning("wandb not available, skipping experiment tracking")
        
        # Training loop
        best_reward = float('-inf')
        episode_times = []
        total_steps = 0
        
        for episode in range(start_episode, args.num_episodes):
            # Check exit flag at start of each episode
            if exit_flag.is_set():
                logger.info("Exit flag detected, stopping training")
                break
            
            # Reset watchdog timer at the start of each episode
            if 'watchdog' in globals():
                watchdog.reset()
            
            logger.info(f"Starting episode {episode+1}/{args.num_episodes}")
            
            # Implement dynamic resolution scheduling
            if episode == 0:
                # Start with lower resolution for faster initial learning
                env.screen_capture.process_resolution = (320, 180)
                agent.network.expected_width = 320
                agent.network.expected_height = 180
                logger.info("Using initial low resolution (320x180) for faster learning")
            elif episode == 1000:
                # Increase to medium resolution
                env.screen_capture.process_resolution = (384, 216)
                agent.network.expected_width = 384
                agent.network.expected_height = 216
                logger.info("Increasing to medium resolution (384x216)")
            elif episode == 3000:
                # Increase to target resolution
                env.screen_capture.process_resolution = (480, 270)
                agent.network.expected_width = 480
                agent.network.expected_height = 270
                logger.info("Increasing to target resolution (480x270)")
            
            # Wait for previous async update to complete if active
            if hasattr(agent, 'is_updating') and agent.is_updating:
                logger.info("Waiting for previous async update to complete...")
                while agent.is_updating:
                    time.sleep(0.1)  # Small sleep to prevent CPU spin
            
            # Collect trajectory
            trajectory_start_time = time.time()
            states, actions, log_probs, rewards, values, dones, next_states, episode_reward, last_info = collect_trajectory(
                env=env,
                agent=agent,
                max_steps=args.max_steps,
                device=config.device,
                args=args
            )
            trajectory_time = time.time() - trajectory_start_time
            
            # Convert lists to tensors for the agent
            if states and isinstance(states[0], torch.Tensor):
                states_tensor = torch.stack(states)
            else:
                states_tensor = torch.FloatTensor(np.array(states)) if states else torch.FloatTensor([])
            
            if next_states and isinstance(next_states[0], torch.Tensor):
                next_states_tensor = torch.stack(next_states)
            else:
                next_states_tensor = torch.FloatTensor(np.array(next_states)) if next_states else torch.FloatTensor([])
            
            # Store experience in agent's memory
            for i in range(len(states)):
                agent.remember(
                    states[i], 
                    actions[i], 
                    log_probs[i], 
                    rewards[i], 
                    values[i], 
                    dones[i], 
                    next_states[i] if i < len(next_states) else None
                )
            
            # Start asynchronous update
            logger.info(f"Starting async update for episode {episode+1}")
            agent.update_async()
            
            # While update is happening asynchronously, we can do other work
            # like logging, checkpointing, etc.
            
            # Save checkpoint if it's time
            if (episode + 1) % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{episode+1}.pt")
                agent.save(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Occasionally save time-based checkpoint
            current_time = time.time()
            if args.time_checkpoint_freq > 0 and current_time - last_time_checkpoint >= args.time_checkpoint_freq * 60:
                time_checkpoint_path = os.path.join(args.checkpoint_dir, f"time_checkpoint_{int(current_time)}.pt")
                agent.save(time_checkpoint_path)
                logger.info(f"Saved time-based checkpoint to {time_checkpoint_path}")
                last_time_checkpoint = current_time
            
            # Log metrics
            logger.info(f"Episode {episode+1}: Reward={episode_reward:.2f}, Steps={len(states)}")
            
            # Check exit flag after each trajectory
            if exit_flag.is_set():
                logger.info("Exit flag detected, stopping training")
                break
            
            # Check if episode ended stuck in a menu - using last_info from trajectory
            menu_detected = False
            if last_info and last_info.get('menu_detected', False):
                menu_detected = True
            
            if menu_detected:
                logger.warning("Episode ended while stuck in a menu - taking corrective action")
                
                # Try clicking the RESUME GAME button directly with exact coordinates
                width, height = env._get_screen_dimensions()
                
                # Exact coordinates from user: (720, 513) for 1920x1080 resolution
                resume_x, resume_y = (720, 513)
                
                # Scale for different resolutions
                if width != 1920 or height != 1080:
                    x_scale = width / 1920
                    y_scale = height / 1080
                    resume_x = int(resume_x * x_scale)
                    resume_y = int(resume_y * y_scale)
                
                logger.info(f"Clicking RESUME GAME at exact coordinates: ({resume_x}, {resume_y})")
                
                # Click multiple times with delays
                for _ in range(3):
                    env.input_simulator.mouse_click(resume_x, resume_y)
                    time.sleep(1.0)
                
                # If still in menu, try the safe handling method
                if env.visual_estimator.detect_main_menu(env.screen_capture.capture_frame()):
                    logger.info("Still in menu, trying safe menu handling method")
                    env.input_simulator.safe_menu_handling()
            
            # Save checkpoint if best reward
            if episode_reward > best_reward:
                best_reward = episode_reward
                # Create checkpoint directory if it doesn't exist
                checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                agent.save(checkpoint_path)
                logger.info(f"Saved new best model with reward: {best_reward:.2f}")
            
            # Add a delay between episodes to prevent system resource issues
            time.sleep(2.0)
            
            # Reset watchdog before post-episode operations
            if 'watchdog' in globals():
                watchdog.reset()
        
        # Save final model
        agent.save(os.path.join(args.checkpoint_dir, "final_model.pt"))
        logger.info("Saved final model")
        env.close()
        logger.info("Training completed")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        traceback.print_exc()
    finally:
        # Stop the watchdog timer before cleanup
        if 'watchdog' in globals():
            watchdog.stop()
            
        # Ensure environment is properly closed regardless of how we exit
        try:
            if 'env' in locals():
                logger.info("Ensuring environment is properly closed...")
                env.close()
        except Exception as closing_error:
            logger.error(f"Error while closing environment: {closing_error}")

if __name__ == "__main__":
    main() 