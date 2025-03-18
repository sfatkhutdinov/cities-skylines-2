import torch
import numpy as np
from pathlib import Path
import argparse
import wandb
import logging
import sys
import time

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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--mock", action="store_true", help="Use mock environment for training without the actual game")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--resolution", type=str, default="320x240", help="Resolution for screen capture (WxH)")
    parser.add_argument("--action_delay", type=float, default=1.0, help="Minimum delay between actions (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--resume_best", action="store_true", help="Resume training from best checkpoint")
    parser.add_argument("--menu_screenshot", type=str, default=None, help="Path to a screenshot of the menu for reference-based detection")
    parser.add_argument("--capture_menu", action="store_true", help="Capture a menu screenshot at startup (assumes you're starting from the menu)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    return parser.parse_args()

def collect_trajectory(
    env: CitiesEnvironment,
    agent: PPOAgent,
    max_steps: int
) -> Tuple[Dict[str, torch.Tensor], float, int]:
    """Collect a trajectory of experiences.
    
    Args:
        env (CitiesEnvironment): Game environment
        agent (PPOAgent): RL agent
        max_steps (int): Maximum steps to collect
        
    Returns:
        Tuple[Dict[str, torch.Tensor], float, int]:
            - experiences: Dictionary of experiences
            - episode_reward: Total reward for episode
            - episode_length: Length of episode
    """
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    dones = []
    action_infos = []  # Store action info for better debugging
    
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    
    # Track previous actions and their results for menu detection
    previous_action_idx = None
    was_in_menu = False
    
    for _ in range(max_steps):
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
        actions.append(action_idx)
        rewards.append(reward)
        log_probs.append(log_prob.item())
        values.append(value.item())
        dones.append(done)
        action_infos.append(action_info)
        
        episode_reward += reward
        episode_length += 1
        
        # Log interesting actions (UI exploration)
        if action_info["type"] in ["ui", "ui_position"]:
            logger.info(f"UI Action: {action_info}")
        
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
    
    # Convert lists to tensors
    if isinstance(states[0], torch.Tensor):
        states_tensor = torch.stack(states)
    else:
        states_tensor = torch.FloatTensor(np.array(states))
        
    # Create next_states by shifting the states list and repeating the last state
    next_states_list = states[1:] + [states[-1]]
    if isinstance(next_states_list[0], torch.Tensor):
        next_states = torch.stack(next_states_list)
    else:
        next_states = torch.FloatTensor(np.array(next_states_list))
    
    experiences = {
        "states": states_tensor,
        "next_states": next_states,
        "actions": torch.LongTensor(actions),
        "rewards": torch.FloatTensor(rewards),
        "log_probs": torch.FloatTensor(log_probs),
        "values": torch.FloatTensor(values),
        "dones": torch.BoolTensor(dones),
        "action_infos": action_infos  # For analysis
    }
    
    return experiences, episode_reward, episode_length

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file in the given directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        return None
        
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not checkpoint_files:
        return None
        
    # Extract episode numbers and find the highest
    episodes = []
    for cp_file in checkpoint_files:
        try:
            episode = int(cp_file.stem.split('_')[1])
            episodes.append((episode, cp_file))
        except ValueError:
            continue
            
    if not episodes:
        return None
        
    # Sort by episode number and return the file with the highest number
    episodes.sort(key=lambda x: x[0])
    return episodes[-1]  # Returns (episode_number, file_path)

def train():
    """Main training loop."""
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
    
    if args.resume or args.resume_best:
        if args.resume_best:
            best_checkpoint_path = checkpoint_dir / "best_model.pt"
            if best_checkpoint_path.exists():
                # Resume from best model
                try:
                    logger.info("Loading best model checkpoint...")
                    agent.load(best_checkpoint_path)
                    checkpoint_loaded = True
                    logger.info("Successfully loaded best model")
                except Exception as e:
                    logger.error(f"Failed to load best model: {str(e)}")
            else:
                logger.warning(f"Best model checkpoint not found at {best_checkpoint_path}")
        elif args.resume:
            # Find and load the latest checkpoint
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                episode_num, checkpoint_path = latest_checkpoint
                try:
                    logger.info(f"Loading checkpoint from episode {episode_num}...")
                    agent.load(checkpoint_path)
                    start_episode = episode_num
                    checkpoint_loaded = True
                    logger.info(f"Successfully loaded checkpoint from episode {episode_num}")
                except Exception as e:
                    logger.error(f"Failed to load checkpoint: {str(e)}")
        
        if not checkpoint_loaded:
            logger.warning("No checkpoint found or loading failed. Starting training from scratch.")
    
    # Initialize wandb if enabled
    if args.use_wandb:
        try:
            logger.info("Initializing wandb...")
            wandb.init(
                project="cities-skylines-2-rl",
                config=vars(args)
            )
            logger.info("wandb initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {str(e)}")
            args.use_wandb = False
    
    # Training loop
    if checkpoint_loaded:
        # Try to find the current best reward if we're resuming
        best_model_path = checkpoint_dir / "best_model.pt"
        if best_model_path.exists():
            # Just a heuristic: assume the best model has a better reward than starting fresh
            best_reward = 0  # This will be updated on the first better reward
            logger.info(f"Found existing best model, setting initial best_reward to {best_reward}")
    
    logger.info("Starting training loop...")
    for episode in range(start_episode, args.num_episodes):
        logger.info(f"Starting episode {episode+1}/{args.num_episodes}")
        
        # Collect trajectory
        experiences, episode_reward, episode_length = collect_trajectory(
            env, agent, args.max_steps
        )
        logger.info(f"Collected trajectory - Length: {episode_length}, Reward: {episode_reward:.2f}")
        
        # Check if episode ended stuck in a menu
        current_frame = env.screen_capture.capture_frame()
        menu_detected = env.visual_estimator.detect_main_menu(current_frame)
        
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
        
        # Update agent
        metrics = agent.update()
        logger.info(f"Updated agent - Metrics: {metrics}")
        
        # Log episode results to wandb if enabled
        if args.use_wandb:
            wandb.log({
                "episode": episode,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
                **metrics
            })
        
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