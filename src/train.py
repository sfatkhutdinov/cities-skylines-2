import torch
import numpy as np
from pathlib import Path
import argparse
import wandb
import logging
import sys
import time
import signal
import datetime
import os

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def collect_trajectory(
    env: CitiesEnvironment,
    agent: PPOAgent,
    max_steps: int,
    device: torch.device,
    args
) -> Tuple[list, list, list, list, list, list, list, float]:
    """Collect a trajectory of experience.
    
    Args:
        env: Environment to collect from
        agent: Agent to collect with
        max_steps: Maximum steps to collect
        device: Device to use
        args: Command line arguments
        
    Returns:
        Tuple of (states, actions, log_probs, rewards, values, dones, next_states, total_reward)
    """
    states = []
    next_states = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    dones = []
    total_reward = 0.0
    
    state = env.reset()
    
    # Define mini-batch size for parallel processing
    mini_batch_size = min(8, max_steps)  # Process up to 8 steps at once
    
    for step in range(0, max_steps, mini_batch_size):
        # Limit batch size to remaining steps
        current_batch_size = min(mini_batch_size, max_steps - step)
        
        # Initialize batch storage
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_values = []
        batch_next_states = []
        batch_rewards = []
        batch_dones = []
        
        # Collect experience for the mini-batch
        for i in range(current_batch_size):
            batch_states.append(state)
            
            # If this is the last step in the batch, we'll use the results
            # Otherwise we're just pre-filling the batch
            if i == current_batch_size - 1:
                # Process the entire batch at once using the batch action selection
                batch_tensor = torch.stack(batch_states)
                with torch.inference_mode():
                    batch_actions_tensor, batch_log_probs_tensor, batch_values_tensor = agent.select_action_batch(batch_tensor)
                
                # Now execute each action in the environment and collect results
                for j in range(current_batch_size):
                    # Get the action for this position in the batch
                    action = batch_actions_tensor[j].item()
                    log_prob = batch_log_probs_tensor[j]
                    value = batch_values_tensor[j]
                    
                    # Execute in environment
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    
                    # Store transition
                    states.append(batch_states[j])
                    actions.append(action)
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(reward)
                    dones.append(done)
                    next_states.append(next_state)
                    
                    # Prepare for next iteration
                    state = next_state
                    
                    if done:
                        # If episode is done, don't continue
                        break
            
        # If episode ended in the middle of the batch, break out of main loop
        if done:
            break
    
    return states, actions, log_probs, rewards, values, dones, next_states, total_reward

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

def main():
    """Main training loop."""
    global stop_training
    logger.info("Starting training...")
    args = parse_args()
    
    # Set debug level if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Parse resolution 
    try:
        width, height = map(int, args.resolution.split('x'))
        logger.info(f"Using resolution: {width}x{height}")
    except:
        logger.warning(f"Invalid resolution format: {args.resolution}, using default 320x240")
        width, height = 320, 240
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created checkpoint directory at {checkpoint_dir}")
    
    # Initialize environment and config
    logger.info("Initializing environment...")
    logger.info("Initializing config...")
    config = HardwareConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
        resolution=(height, width),  # Use resolution from args
        frame_stack=1  # Explicitly set frame_stack to 1 to avoid channel mismatch
    )
    # Add mock_mode flag to config for the model to use
    config.mock_mode = args.mock
    logger.info(f"Config initialized with device: {config.device}, mock mode: {args.mock}")
    
    # Initialize environment with config
    env = CitiesEnvironment(
        config=config, 
        mock_mode=args.mock, 
        menu_screenshot_path=args.menu_screenshot
    )
    # Set minimum delay between actions
    env.min_action_delay = args.action_delay
    logger.info(f"Environment initialized with mock mode: {args.mock}, action delay: {args.action_delay}s")
    
    # If user requested to capture menu screenshot at startup
    if args.capture_menu and not args.mock:
        logger.info("Capturing menu screenshot at startup (assuming game is showing menu)")
        menu_path = "menu_reference.png"
        if env.capture_menu_reference(menu_path):
            logger.info(f"Successfully captured menu screenshot to {menu_path}")
        else:
            logger.warning("Failed to capture menu screenshot")
    
    logger.info("Initializing agent...")
    agent = PPOAgent(config)
    logger.info("Agent initialized")
    
    # Handle checkpoint resumption if requested
    start_episode = 0
    checkpoint_loaded = False
    best_reward = float("-inf")
    
    # Auto-resume logic: Automatically load checkpoints unless explicitly disabled
    should_auto_resume = not args.no_auto_resume and not args.resume and not args.resume_best
    
    if args.resume or args.resume_best or should_auto_resume:
        # Priority order for loading:
        # 1. If --resume_best is specified, try loading best_model.pt
        # 2. If --resume is specified, try loading the latest checkpoint
        # 3. If auto-resume is enabled (default), try best_model.pt first, then latest checkpoint
        
        if (args.resume_best or should_auto_resume) and (checkpoint_dir / "best_model.pt").exists():
            # Resume from best model
            try:
                logger.info("Loading best model checkpoint...")
                agent.load(checkpoint_dir / "best_model.pt")
                checkpoint_loaded = True
                logger.info("Successfully loaded best model")
            except Exception as e:
                logger.error(f"Failed to load best model: {str(e)}")
                
        # If best model loading was not requested or failed, try latest checkpoint
        if not checkpoint_loaded and (args.resume or should_auto_resume):
            # Find and load the latest checkpoint
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                episode_num, checkpoint_path = latest_checkpoint
                try:
                    logger.info(f"Loading checkpoint from {checkpoint_path}...")
                    agent.load(checkpoint_path)
                    start_episode = episode_num
                    checkpoint_loaded = True
                    logger.info(f"Successfully loaded checkpoint from {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {str(e)}")
        
        if not checkpoint_loaded:
            logger.warning("No checkpoint found or loading failed. Starting training from scratch.")
        elif should_auto_resume:
            logger.info("Auto-resume successfully loaded checkpoint.")
    
    # Initialize wandb if available
    use_wandb = False
    try:
        wandb.init(project="cs2-agent", config={
            "num_episodes": args.num_episodes,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "resolution": args.resolution,
            "device": args.device,
            "mock_mode": args.mock,
            "use_curiosity": True  # Flag that we're using curiosity-driven exploration
        })
        use_wandb = True
        logger.info("Initialized wandb for experiment tracking")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {str(e)}")
    
    # Training loop
    best_reward = float('-inf')
    episode_times = []
    total_steps = 0
    last_time_checkpoint = time.time()
    
    for episode in range(start_episode, args.num_episodes):
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
        states, actions, log_probs, rewards, values, dones, next_states, episode_reward = collect_trajectory(
            env=env,
            agent=agent,
            max_steps=args.max_steps,
            device=args.device,
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
        
        # Optional: wait for update to complete before next episode
        # Uncomment if you want to ensure update completes before next trajectory
        # while agent.is_updating:
        #     time.sleep(0.1)
            
        # Check if we should stop training
        if stop_training:
            logger.info("Stopping training due to interrupt signal")
            break
        
        # Check if episode ended stuck in a menu
        if experiences.get('info', {}).get('menu_detected', False):
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
            agent.save(checkpoint_dir / "best_model.pt")
            logger.info(f"Saved new best model with reward: {best_reward:.2f}")
        
        # Add a delay between episodes to prevent system resource issues
        time.sleep(2.0)
    
    # Save final model
    agent.save(checkpoint_dir / "final_model.pt")
    logger.info("Saved final model")
    env.close()
    logger.info("Training completed")

if __name__ == "__main__":
    main() 