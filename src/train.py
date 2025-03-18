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
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for updates")
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
    total_steps: int,
    checkpoint_dir: Path,
    step_checkpoint_freq: int
) -> Tuple[Dict[str, torch.Tensor], float, int]:
    """Collect a trajectory of experiences.
    
    Args:
        env (CitiesEnvironment): Game environment
        agent (PPOAgent): RL agent
        max_steps (int): Maximum steps to collect
        total_steps (int): Current total steps completed
        checkpoint_dir (Path): Directory to save checkpoints
        step_checkpoint_freq (int): Steps between checkpoints
        
    Returns:
        Tuple[Dict[str, torch.Tensor], float, int]:
            - experiences: Dictionary of experiences
            - episode_reward: Total reward for episode
            - episode_length: Length of episode
    """
    states = []
    next_states = []
    actions = []
    rewards = []
    intrinsic_rewards = []
    total_rewards = []  # Combined extrinsic and intrinsic rewards
    log_probs = []
    values = []
    dones = []
    action_infos = []  # Store action info for better debugging
    
    state = env.reset()
    episode_reward = 0
    episode_intrinsic_reward = 0
    episode_length = 0
    
    # Track previous actions and their results for menu detection
    previous_action_idx = None
    was_in_menu = False
    
    for step in range(max_steps):
        # Step-based checkpoint
        current_total_steps = total_steps + step
        if step_checkpoint_freq > 0 and current_total_steps > 0 and current_total_steps % step_checkpoint_freq == 0:
            checkpoint_path = checkpoint_dir / f"step_checkpoint_{current_total_steps}.pt"
            agent.save(checkpoint_path)
            logger.info(f"Saved step-based checkpoint at step {current_total_steps}")
        
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0) if state.dim() == 3 else state
        
        # Select action
        action, log_prob, value = agent.select_action(state_tensor)
        
        # Take step in environment
        action_idx = action.item()
        action_info = env.actions[action_idx]
        next_state, reward, done, info = env.step(action_idx)
        
        # Store next state in agent for curiosity module
        agent.store_next_state(next_state)
        
        # Compute intrinsic curiosity reward
        intrinsic_reward = agent.compute_intrinsic_reward(state, next_state, action_idx)
        
        # Combine extrinsic and intrinsic rewards
        total_reward = reward + intrinsic_reward
        
        # Update agent's tracking of rewards for this action
        agent.update_from_reward(action_idx, reward)
        
        # Check if we transitioned into a menu and register the action that caused it
        current_in_menu = info.get('menu_detected', False)
        
        if current_in_menu and not was_in_menu and previous_action_idx is not None:
            # We just entered a menu, and we know the action that did it
            logger.warning(f"Detected menu transition after action {previous_action_idx} ({env.actions[previous_action_idx]})")
            
            # Register the action with the agent to avoid it in the future
            # Increased penalty from 0.7 to 0.8 for more aggressive avoidance
            agent.register_menu_action(previous_action_idx, penalty=0.8)
        
        # Store current menu state for next iteration
        was_in_menu = current_in_menu
        previous_action_idx = action_idx
        
        # Decay menu penalties occasionally to allow for exploration
        if episode_length % 25 == 0:  # More frequent decay (was 50)
            agent.decay_menu_penalties()
        
        # Store experience
        states.append(state)
        next_states.append(next_state)
        actions.append(action_idx)
        rewards.append(reward)
        intrinsic_rewards.append(intrinsic_reward)
        total_rewards.append(total_reward)
        log_probs.append(log_prob.item())
        values.append(value.item())
        dones.append(done)
        action_infos.append(action_info)
        
        episode_reward += reward
        episode_intrinsic_reward += intrinsic_reward
        episode_length += 1
        
        # Log interesting actions (UI exploration)
        if action_info["type"] in ["ui", "ui_position"]:
            logger.info(f"UI Action: {action_info}")
        
        # Log high curiosity reward actions
        if intrinsic_reward > 0.1:
            logger.info(f"High curiosity reward ({intrinsic_reward:.4f}) for action: {action_info}")
        
        # If we get an extremely negative reward, log and consider early termination
        if reward < -500:
            logger.warning(f"Extremely negative reward: {reward} - possible menu penalty")
            
            # If we're stuck in a menu for too long (10+ steps), try to reset
            if info.get('menu_stuck_counter', 0) > 10:
                logger.error(f"Stuck in menu for {info.get('menu_stuck_counter')} steps - forcing exit")
                # Try clicking the resume button more aggressively
                env.input_simulator.safe_menu_handling()
        
        if done:
            break
            
        state = next_state
    
    # Log summary of intrinsic rewards
    logger.info(f"Episode intrinsic reward: {episode_intrinsic_reward:.4f}")
    
    # Convert lists to tensors
    if isinstance(states[0], torch.Tensor):
        states_tensor = torch.stack(states)
    else:
        states_tensor = torch.FloatTensor(np.array(states))
    
    if isinstance(next_states[0], torch.Tensor):
        next_states_tensor = torch.stack(next_states)
    else:
        next_states_tensor = torch.FloatTensor(np.array(next_states))
        
    experiences = {
        "states": states_tensor,
        "next_states": next_states_tensor,
        "actions": torch.LongTensor(actions),
        "rewards": torch.FloatTensor(rewards),
        "intrinsic_rewards": torch.FloatTensor(intrinsic_rewards),
        "total_rewards": torch.FloatTensor(total_rewards),
        "log_probs": torch.FloatTensor(log_probs),
        "values": torch.FloatTensor(values),
        "dones": torch.BoolTensor(dones),
        "action_infos": action_infos,  # For analysis
        "info": info  # Store final info
    }
    
    return experiences, episode_reward, episode_length

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

def train():
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
        if stop_training:
            logger.info("Training interrupted by user. Saving final checkpoint...")
            # Save interrupt checkpoint
            interrupt_path = checkpoint_dir / f"interrupt_checkpoint.pt"
            agent.save(interrupt_path)
            logger.info(f"Saved interrupt checkpoint to {interrupt_path}")
            break
            
        episode_start_time = time.time()
        
        # Check if it's time for a time-based checkpoint
        current_time = time.time()
        time_elapsed = (current_time - last_time_checkpoint) / 60  # Convert to minutes
        if args.time_checkpoint_freq > 0 and time_elapsed >= args.time_checkpoint_freq:
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoint_dir / f"time_checkpoint_{time_str}.pt"
            agent.save(checkpoint_path)
            logger.info(f"Saved time-based checkpoint after {time_elapsed:.1f} minutes")
            last_time_checkpoint = current_time
        
        # Collect experience
        logger.info(f"Starting episode {episode}...")
        experiences, episode_reward, episode_length = collect_trajectory(
            env=env,
            agent=agent,
            max_steps=args.max_steps,
            total_steps=total_steps,
            checkpoint_dir=checkpoint_dir,
            step_checkpoint_freq=args.step_checkpoint_freq
        )
        
        # Update total steps
        total_steps += episode_length
        
        # Calculate average intrinsic reward
        avg_intrinsic_reward = experiences["intrinsic_rewards"].mean().item()
        
        # Get total combined reward
        total_episode_reward = episode_reward + experiences["intrinsic_rewards"].sum().item()
        
        # Update the agent
        update_metrics = agent.update()
        
        # Calculate episode time
        episode_time = time.time() - episode_start_time
        episode_times.append(episode_time)
        avg_time = sum(episode_times[-10:]) / min(len(episode_times), 10)
        
        # Log metrics
        metrics = {
            "episode": episode,
            "reward": episode_reward,
            "intrinsic_reward": avg_intrinsic_reward,
            "total_reward": total_episode_reward,
            "episode_length": episode_length,
            "time": episode_time,
            "avg_time": avg_time,
            "total_steps": total_steps,
            **update_metrics
        }
        
        logger.info(f"Episode {episode}: " 
                    f"reward={episode_reward:.2f}, "
                    f"intrinsic={avg_intrinsic_reward:.4f}, "
                    f"total={total_episode_reward:.2f}, "
                    f"length={episode_length}, "
                    f"time={episode_time:.2f}s, "
                    f"total_steps={total_steps}")
        
        if use_wandb:
            wandb.log(metrics)
        
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
            
        # Save periodic checkpoint
        if (episode + 1) % args.checkpoint_freq == 0:
            agent.save(checkpoint_dir / f"checkpoint_{episode+1}.pt")
            logger.info(f"Saved periodic checkpoint at episode {episode+1}")
            
        print(f"Episode {episode+1}/{args.num_episodes} - Reward: {episode_reward:.2f}")
        
        # Add a delay between episodes to prevent system resource issues
        time.sleep(2.0)
    
    # Save final model
    agent.save(checkpoint_dir / "final_model.pt")
    logger.info("Saved final model")
    env.close()
    logger.info("Training completed")

if __name__ == "__main__":
    train() 