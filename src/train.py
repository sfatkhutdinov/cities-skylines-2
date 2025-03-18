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
    def __init__(self, timeout=60):  # Increased from 5 to 60 seconds
        self.timeout = timeout
        self.active = False
        self.reset_event = threading.Event()
        self.thread = None
        self.warning_triggered = False
        self.warning_time = None
        
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
            triggered = not self.reset_event.wait(self.timeout / 2 if self.warning_triggered else self.timeout)
            
            if triggered and self.active:
                # First trigger a warning
                if not self.warning_triggered:
                    self.warning_triggered = True
                    self.warning_time = time.time()
                    logger.warning(f"WATCHDOG WARNING - Program may be unresponsive. Will terminate in {self.timeout / 2} seconds if no reset.")
                else:
                    # Only terminate if the warning period has also elapsed
                    if time.time() - self.warning_time >= self.timeout / 2:
                        # Timeout occurred twice and watchdog is still active
                        logger.critical("WATCHDOG TIMEOUT - Program appears frozen. Forcibly terminating!")
                        os._exit(2)  # Force terminate with error code
                    
            else:
                # Reset was received, clear warning state
                self.warning_triggered = False
                
            self.reset_event.clear()
    
    def reset(self):
        """Reset the watchdog timer - must be called regularly to prevent termination"""
        if self.active:
            self.reset_event.set()
            if self.warning_triggered:
                logger.info("Watchdog timer reset after warning - continuing execution")
                self.warning_triggered = False
        
    def stop(self):
        """Stop the watchdog timer"""
        self.active = False
        self.reset_event.set()  # Wake up the thread to exit
        if self.thread:
            self.thread.join(1.0)  # Wait for thread to exit, with timeout

# Start a global watchdog timer that will be reset in the main loop
watchdog = WatchdogTimer(timeout=60)  # Increased from 15 to 60 seconds

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
    parser.add_argument("--watchdog_timeout", type=int, default=60, help="Watchdog timeout in seconds (set to 0 to disable)")
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
    """Collect a trajectory of experience to train the agent.
    
    Args:
        env: The environment to collect the trajectory from
        agent: The agent to use for action selection
        max_steps: The maximum number of steps to collect
        device: The device to use for tensor operations
        args: The command line arguments
    
    Returns:
        A tuple containing:
        - states: Tensor of state observations
        - actions: Tensor of actions
        - log_probs: Tensor of log probabilities
        - values: Tensor of value estimates
        - rewards: Tensor of rewards
        - dones: Tensor of done flags
        - next_states: Tensor of next state observations
        - total_reward: The total reward for the trajectory
        - stats: Dictionary of trajectory statistics
    """
    # Reset environment and prepare for collection
    obs = env.reset()
    
    # Set default movement speed (can be adjusted during training)
    # Setting it to a moderate value (0.6) which is faster than default but not too fast
    env.input_simulator.set_movement_speed(0.6)
    
    # Track speed adjustment
    speed_adjustment_counter = 0
    adjust_speed_every = 100  # Adjust speed every 100 steps

    # Track performance and reward for adaptive speed
    recent_rewards = []
    recent_fps = []
    
    # Initialize stats tracking
    episode_reward = 0
    total_reward = 0
    total_intrinsic_reward = 0
    total_extrinsic_reward = 0
    step_count = 0
    episode_count = 0
    stats = {
        'fps': 0,
        'steps_per_second': 0,
        'avg_reward': 0,
        'intrinsic_reward': 0,
        'extrinsic_reward': 0,
        'episodes': 0,
        'steps': 0,
        'avg_episode_length': 0,
        'intrinsic_reward_ratio': 0
    }
    
    # Time tracking
    start_time = time.time()
    last_time = start_time
    
    # Clear agent memory buffers
    agent.states = []
    agent.actions = []
    agent.action_probs = []
    agent.values = []
    agent.rewards = []
    agent.dones = []
    agent.next_states = []
    
    # Collect the trajectory
    for step in range(max_steps):
        # Reset the watchdog timer for each step
        watchdog.reset()
        
        # Get agent to select an action
        with torch.no_grad():
            action, log_prob, value = agent.act(obs)
        
        # Execute the action in the environment and get next observation, reward, done flag
        next_obs, reward, done, info = env.step(action.item())
        
        # Store step information
        agent.states.append(obs)
        agent.actions.append(action)
        agent.action_probs.append(log_prob)
        agent.values.append(value)
        agent.rewards.append(torch.tensor([reward], device=device))
        agent.dones.append(torch.tensor([done], device=device))
        agent.next_states.append(next_obs)
        
        # Update observation
        obs = next_obs
        
        # Update episode statistics
        episode_reward += reward
        total_reward += reward
        step_count += 1
        
        # Track rewards for adaptive speed adjustment
        recent_rewards.append(reward)
        if len(recent_rewards) > 50:  # Keep last 50 rewards
            recent_rewards.pop(0)
        
        # Calculate current FPS
        current_time = time.time()
        frame_time = current_time - last_time
        current_fps = 1.0 / max(frame_time, 1e-6)  # Avoid division by zero
        last_time = current_time
        
        # Track FPS for adaptive speed
        recent_fps.append(current_fps)
        if len(recent_fps) > 20:  # Keep last 20 FPS readings
            recent_fps.pop(0)
        
        # Adaptively adjust mouse and keyboard speed
        speed_adjustment_counter += 1
        if speed_adjustment_counter >= adjust_speed_every:
            speed_adjustment_counter = 0
            
            # Calculate average recent reward and FPS
            avg_recent_reward = sum(recent_rewards) / max(len(recent_rewards), 1)
            avg_recent_fps = sum(recent_fps) / max(len(recent_fps), 1)
            
            # Base speed on performance:
            # 1. If rewards are positive and FPS is good, increase speed
            # 2. If rewards are negative or FPS is low, decrease speed
            current_speed = env.input_simulator.mouse_speed
            
            # Speed adjustment logic
            if avg_recent_fps < 20:  # FPS is low
                # Reduce speed to improve performance
                new_speed = max(0.3, current_speed - 0.1)
                logger.info(f"FPS low ({avg_recent_fps:.1f}), reducing speed to {new_speed:.1f}")
            elif avg_recent_reward > 0 and avg_recent_fps > 30:
                # Performing well, increase speed
                new_speed = min(0.9, current_speed + 0.05)
                logger.info(f"Performance good (reward: {avg_recent_reward:.1f}, FPS: {avg_recent_fps:.1f}), increasing speed to {new_speed:.1f}")
            elif avg_recent_reward < -5:
                # Performing poorly, reduce speed
                new_speed = max(0.3, current_speed - 0.05)
                logger.info(f"Performance poor (reward: {avg_recent_reward:.1f}), reducing speed to {new_speed:.1f}")
            else:
                # Maintain current speed
                new_speed = current_speed
            
            # Apply the speed adjustment if it changed
            if new_speed != current_speed:
                env.input_simulator.set_movement_speed(new_speed)
        
        # Check if episode is done
        if done:
            # Reset environment for the next episode
            obs = env.reset()
            episode_count += 1
            
            # Log episode statistics
            logger.info(f"Episode {episode_count} completed with reward {episode_reward:.2f} after {step_count} steps")
            episode_reward = 0
            
            # Store intrinsic/extrinsic reward information if available in info
            if 'intrinsic_reward' in info:
                total_intrinsic_reward += info['intrinsic_reward']
            if 'extrinsic_reward' in info:
                total_extrinsic_reward += info['extrinsic_reward']
    
    # Calculate final statistics
    elapsed_time = time.time() - start_time
    stats['fps'] = step_count / elapsed_time if elapsed_time > 0 else 0
    stats['steps_per_second'] = step_count / elapsed_time if elapsed_time > 0 else 0
    stats['avg_reward'] = total_reward / max(step_count, 1)
    stats['intrinsic_reward'] = total_intrinsic_reward
    stats['extrinsic_reward'] = total_extrinsic_reward
    stats['episodes'] = episode_count
    stats['steps'] = step_count
    stats['avg_episode_length'] = step_count / max(episode_count, 1)
    
    total_ir_er = total_intrinsic_reward + total_extrinsic_reward
    if total_ir_er != 0:
        stats['intrinsic_reward_ratio'] = total_intrinsic_reward / total_ir_er
    
    # Convert agent memory to tensors for training
    states = torch.cat(agent.states).detach()
    actions = torch.cat(agent.actions).detach()
    log_probs = torch.cat(agent.action_probs).detach()
    values = torch.cat(agent.values).detach()
    rewards = torch.cat(agent.rewards).detach()
    dones = torch.cat(agent.dones).detach()
    next_states = torch.cat(agent.next_states).detach()
    
    return states, actions, log_probs, values, rewards, dones, next_states, total_reward, stats

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
    
    # Configure and start watchdog timer based on command line args
    if 'watchdog' in globals():
        if args.watchdog_timeout > 0:
            watchdog.timeout = args.watchdog_timeout
            logger.info(f"Watchdog timer configured with {args.watchdog_timeout}s timeout")
            watchdog.start()
        else:
            logger.info("Watchdog timer disabled by command line argument")
    
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
        
        # Reset watchdog before potentially long environment initialization
        if 'watchdog' in globals():
            watchdog.reset()
        
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
        
        # Reset watchdog after environment initialization
        if 'watchdog' in globals():
            watchdog.reset()
        
        # Create agent
        logger.info("Initializing agent...")
        agent = PPOAgent(config)
        
        # Reset watchdog after agent initialization
        if 'watchdog' in globals():
            watchdog.reset()
        
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
                
        # Reset watchdog after model compilation
        if 'watchdog' in globals():
            watchdog.reset()
        
        # Resume from checkpoints if available and not explicitly disabled
        load_checkpoint = not args.no_auto_resume
        use_best = args.resume_best
        if load_checkpoint:
            logger.info(f"Looking for checkpoints to resume training (use_best={use_best})...")
            agent.load_checkpoint(use_best=use_best)
            # Reset watchdog after potentially long checkpoint loading
            if 'watchdog' in globals():
                watchdog.reset()
                
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
                wait_start = time.time()
                while agent.is_updating:
                    time.sleep(0.1)  # Small sleep to prevent CPU spin
                    
                    # Reset watchdog during long waits
                    if time.time() - wait_start > 10:
                        if 'watchdog' in globals():
                            watchdog.reset()
                        wait_start = time.time()
            
            # Collect trajectory
            trajectory_start_time = time.time()
            states, actions, log_probs, values, rewards, dones, next_states, episode_reward, stats = collect_trajectory(
                env=env,
                agent=agent,
                max_steps=args.max_steps,
                device=config.device,
                args=args
            )
            trajectory_time = time.time() - trajectory_start_time
            
            # Reset watchdog after trajectory collection
            if 'watchdog' in globals():
                watchdog.reset()
            
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
                
                # Reset watchdog periodically during memory storage
                if i > 0 and i % 1000 == 0 and 'watchdog' in globals():
                    watchdog.reset()
            
            # Start asynchronous update
            logger.info(f"Starting async update for episode {episode+1}")
            
            # Add a callback to reset watchdog during async update
            original_update_thread = agent._update_thread
            
            def watchdog_update_thread():
                update_start_time = time.time()
                update_thread = threading.Thread(target=original_update_thread)
                update_thread.daemon = True
                update_thread.start()
                
                # Reset watchdog periodically while update is running
                while update_thread.is_alive():
                    if 'watchdog' in globals():
                        watchdog.reset()
                    time.sleep(10)  # Check every 10 seconds
                    
            # Replace the update thread with our watchdog-enabled version
            if hasattr(agent, '_update_thread'):
                agent._original_update_thread = agent._update_thread
                agent._update_thread = watchdog_update_thread
                
            agent.update_async()
            
            # While update is happening asynchronously, we can do other work
            # like logging, checkpointing, etc.
            
            # Save checkpoint if it's time
            if (episode + 1) % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{episode+1}.pt")
                agent.save(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                # Reset watchdog after saving checkpoint
                if 'watchdog' in globals():
                    watchdog.reset()
            
            # Occasionally save time-based checkpoint
            current_time = time.time()
            if args.time_checkpoint_freq > 0 and current_time - last_time_checkpoint >= args.time_checkpoint_freq * 60:
                time_checkpoint_path = os.path.join(args.checkpoint_dir, f"time_checkpoint_{int(current_time)}.pt")
                agent.save(time_checkpoint_path)
                logger.info(f"Saved time-based checkpoint to {time_checkpoint_path}")
                last_time_checkpoint = current_time
                # Reset watchdog after saving checkpoint
                if 'watchdog' in globals():
                    watchdog.reset()
            
            # Log metrics
            logger.info(f"Episode {episode+1}: Reward={episode_reward:.2f}, Steps={len(states)}")
            
            # Check exit flag after each trajectory
            if exit_flag.is_set():
                logger.info("Exit flag detected, stopping training")
                break
            
            # Check if episode ended stuck in a menu - using last_info from trajectory
            menu_detected = False
            if stats and stats.get('menu_detected', False):
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