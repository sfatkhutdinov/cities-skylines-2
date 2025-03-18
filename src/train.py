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

def collect_trajectory(env, agent, max_steps, render=False):
    """Collect trajectory of experience from environment.
    
    Args:
        env: Environment to collect trajectory from
        agent: Agent to use for actions
        max_steps: Maximum number of steps to collect
        render: Whether to render the environment
        
    Returns:
        experiences: List of experience tuples
        total_reward: Total reward for episode
        total_steps: Total steps taken
    """
    state = env.reset()
    
    experiences = []
    done = False
    total_reward = 0
    total_steps = 0
    
    # For tracking menu transitions
    in_menu = False
    menu_transition_count = 0
    consecutive_menu_steps = 0
    
    for step in range(max_steps):
        # Select action
        action, log_prob, value = agent.select_action(state)
        
        # Check if we're in a menu before taking action
        pre_action_in_menu = env.check_menu_state()
        
        if pre_action_in_menu:
            consecutive_menu_steps += 1
            
            # If stuck in menu for too long, try recovery
            if consecutive_menu_steps >= 3:
                logger.info(f"Stuck in menu for {consecutive_menu_steps} steps, attempting recovery")
                try:
                    env.input_simulator.handle_menu_recovery(retries=2)
                except Exception as e:
                    logger.error(f"Menu recovery failed: {e}")
                
                # Re-check menu state after recovery attempt
                in_menu = env.check_menu_state()
                if not in_menu:
                    logger.info("Successfully recovered from menu state")
                    consecutive_menu_steps = 0
        else:
            consecutive_menu_steps = 0
        
        # Execute action in environment
        next_state, reward, done, info = env.step(action.item())
        
        # Check if we just entered or exited a menu
        menu_detected = info.get("menu_detected", False)
        
        # Handle menu transition for learning
        if not pre_action_in_menu and menu_detected:
            # We just entered a menu
            menu_transition_count += 1
            logger.info(f"Action {action.item()} caused menu transition (count: {menu_transition_count})")
            
            # Register this action as a menu-opening action with the agent
            if hasattr(agent, 'register_menu_action'):
                agent.register_menu_action(action.item(), penalty=0.7)
        
        # Store experience
        experiences.append((state, action, reward, next_state, log_prob, value, done))
        
        # Update state
        state = next_state
        total_reward += reward
        total_steps += 1
        
        if render:
            env.render()
            
        if done:
            break
            
    return experiences, total_reward, total_steps

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
    """Train the agent."""
    args = parse_args()
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting training with arguments: {args}")
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load hardware config
    config = HardwareConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Update config with command-line args
    config.gamma = args.gamma
    config.frame_skip = args.frame_skip
    
    # Set PPO-specific parameters
    config.ppo_epochs = args.ppo_epochs
    config.clip_range = args.clip_range
    config.value_loss_coef = args.value_loss_coef
    config.entropy_coef = args.entropy_coef
    config.max_grad_norm = args.max_grad_norm
    config.gae_lambda = args.gae_lambda
    
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
    
    # Connect environment and agent for better coordination
    env.agent = agent
    
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
            env, agent, args.max_steps, render=args.render
        )
        logger.info(f"Collected trajectory - Length: {episode_length}, Reward: {episode_reward:.2f}")
        
        # Process experiences for PPO update
        if experiences:
            # Extract components
            states = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            dones = []
            
            for state, action, reward, next_state, log_prob, value, done in experiences:
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                log_probs.append(log_prob.item())
                values.append(value.item())
                dones.append(done)
            
            # Store in agent memory
            agent.states = states
            agent.actions = [torch.tensor([a]) for a in actions]  # Convert to tensor
            agent.rewards = rewards
            agent.dones = dones
            
            # Update agent policy
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
        
        # Check if episode ended stuck in a menu
        current_frame = env.screen_capture.capture_frame()
        menu_detected = env.visual_estimator.detect_main_menu(current_frame)
        
        if menu_detected:
            logger.warning("Episode ended while stuck in a menu - taking corrective action")
            env.input_simulator.handle_menu_recovery(retries=3)
        
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