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
        if args.resume_best and (checkpoint_dir / "best_model.pt").exists():
            # Resume from best model
            try:
                logger.info("Loading best model checkpoint...")
                agent.load(checkpoint_dir / "best_model.pt")
                checkpoint_loaded = True
                logger.info("Successfully loaded best model")
            except Exception as e:
                logger.error(f"Failed to load best model: {str(e)}")
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
    
    # Initialize wandb (comment out to skip)
    # logger.info("Initializing wandb...")
    # wandb.init(
    #     project="cities-skylines-2-rl",
    #     config=vars(args)
    # )
    # logger.info("wandb initialized")
    
    # Training loop
    if checkpoint_loaded:
        # Try to find the current best reward if we're resuming
        if (checkpoint_dir / "best_model.pt").exists():
            # Just a heuristic: assume the best model has a better reward than starting fresh
            best_reward = 0  # This will be updated on the first better reward
    
    logger.info("Starting training loop...")
    for episode in range(start_episode, args.num_episodes):
        logger.info(f"Starting episode {episode+1}/{args.num_episodes}")
        
        # Collect trajectory
        experiences, episode_reward, episode_length = collect_trajectory(
            env, agent, args.max_steps
        )
        logger.info(f"Collected trajectory - Length: {episode_length}, Reward: {episode_reward:.2f}")
        
        # Update agent
        metrics = agent.update()
        logger.info(f"Updated agent - Metrics: {metrics}")
        
        # Log episode results
        # wandb.log({
        #     "episode": episode,
        #     "episode_reward": episode_reward,
        #     "episode_length": episode_length,
        #     **metrics
        # })
        
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