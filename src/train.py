import torch
import numpy as np
from pathlib import Path
import argparse
import wandb
import logging
import sys
import time
import os
import signal
import threading
from datetime import datetime, timedelta

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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train reinforcement learning agent")
    
    # Training parameters
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updates")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clipping parameter")
    
    # Checkpointing and resumption
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=100, help="Episodes between checkpoints")
    parser.add_argument("--autosave_interval", type=int, default=15, help="Minutes between auto-saves")
    parser.add_argument("--backup_checkpoints", type=int, default=5, help="Number of backup checkpoints to keep")
    parser.add_argument("--fresh_start", action="store_true", help="Force starting training from scratch, ignoring existing checkpoints")
    parser.add_argument("--use_best", action="store_true", help="Resume training from best checkpoint instead of latest")
    
    # Environment settings
    parser.add_argument("--mock_env", action="store_true", help="Use mock environment for testing")
    parser.add_argument("--disable_menu_detection", action="store_true", help="Disable menu detection")
    
    # Monitoring and logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Hardware acceleration
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (FP16) if available")
    
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

class CheckpointManager:
    """Manages saving and loading checkpoints, including auto-saving functionality."""
    
    def __init__(self, checkpoint_dir, agent, env, autosave_interval=15, max_backups=5):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir (str): Directory to save checkpoints
            agent (PPOAgent): Agent to checkpoint
            env (CitiesEnvironment): Environment to checkpoint
            autosave_interval (int): Minutes between auto-saves
            max_backups (int): Maximum number of backup checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.agent = agent
        self.env = env
        self.autosave_interval = autosave_interval
        self.max_backups = max_backups
        
        # Create subdirectories
        self.reward_system_dir = self.checkpoint_dir / "reward_system"
        self.reward_system_dir.mkdir(exist_ok=True)
        
        # Variables for auto-saving
        self.stop_autosave = threading.Event()
        self.autosave_thread = None
        self.last_autosave_time = datetime.now()
        self.best_reward = float("-inf")
        self.current_episode = 0
    
    def save_checkpoint(self, episode, is_best=False, is_final=False, is_autosave=False):
        """Save checkpoint of the current agent and environment state.
        
        Args:
            episode (int): Current episode number
            is_best (bool): Whether this is the best checkpoint so far
            is_final (bool): Whether this is the final checkpoint
            is_autosave (bool): Whether this is an auto-save
            
        Returns:
            str: Path to the saved checkpoint
        """
        try:
            # Determine checkpoint name
            if is_final:
                checkpoint_path = self.checkpoint_dir / "final_model.pt"
            elif is_best:
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
            elif is_autosave:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_path = self.checkpoint_dir / f"autosave_{timestamp}.pt"
            else:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_{episode}.pt"
            
            # Save agent state
            self.agent.save(checkpoint_path)
            
            # Save environment's reward system state
            if hasattr(self.env, 'reward_system') and hasattr(self.env.reward_system, 'save_state'):
                rs_path = self.reward_system_dir / f"reward_system_{episode}"
                self.env.reward_system.save_state(rs_path)
            
            # Save metadata
            metadata_path = self.checkpoint_path_to_metadata_path(checkpoint_path)
            metadata = {
                'episode': episode,
                'timestamp': datetime.now().isoformat(),
                'is_best': is_best,
                'is_final': is_final,
                'is_autosave': is_autosave
            }
            torch.save(metadata, metadata_path)
            
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Clean up old auto-saves if we have too many
            if is_autosave:
                self._clean_old_autosaves()
                
            return str(checkpoint_path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {str(e)}")
            return None
    
    def load_checkpoint(self, path=None, use_best=False):
        """Load checkpoint for agent and environment.
        
        Args:
            path (str, optional): Specific checkpoint path to load
            use_best (bool): Whether to load the best checkpoint
            
        Returns:
            int: Episode number of the loaded checkpoint, or None if loading failed
        """
        try:
            # Determine which checkpoint to load
            if use_best:
                checkpoint_path = self.checkpoint_dir / "best_model.pt"
            elif path:
                checkpoint_path = Path(path)
            else:
                latest = find_latest_checkpoint(self.checkpoint_dir)
                if latest:
                    _, checkpoint_path = latest
                else:
                    logger.warning("No checkpoint found to load")
                    return None
            
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint does not exist: {checkpoint_path}")
                return None
            
            # Load agent state
            self.agent.load(checkpoint_path)
            
            # Load metadata to get episode number
            metadata_path = self.checkpoint_path_to_metadata_path(checkpoint_path)
            episode = None
            if metadata_path.exists():
                metadata = torch.load(metadata_path)
                episode = metadata.get('episode', 0)
                self.current_episode = episode
            else:
                # Try to extract episode number from filename
                try:
                    episode = int(checkpoint_path.stem.split('_')[1])
                    self.current_episode = episode
                except (IndexError, ValueError):
                    episode = 0
            
            # Load environment's reward system state
            if hasattr(self.env, 'reward_system') and hasattr(self.env.reward_system, 'load_state'):
                rs_path = self.reward_system_dir / f"reward_system_{episode}"
                if os.path.exists(rs_path):
                    self.env.reward_system.load_state(rs_path)
                    logger.info(f"Loaded reward system state from {rs_path}")
                else:
                    logger.warning(f"Reward system state not found at {rs_path}")
            
            logger.info(f"Successfully loaded checkpoint from episode {episode}")
            return episode
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {str(e)}")
            return None
    
    def start_autosave_thread(self):
        """Start a background thread that automatically saves checkpoints at regular intervals."""
        self.stop_autosave.clear()
        self.autosave_thread = threading.Thread(target=self._autosave_worker, daemon=True)
        self.autosave_thread.start()
        logger.info(f"Started auto-save thread (interval: {self.autosave_interval} minutes)")
    
    def stop_autosave_thread(self):
        """Stop the auto-save thread."""
        if self.autosave_thread:
            self.stop_autosave.set()
            self.autosave_thread.join(timeout=5.0)
            logger.info("Stopped auto-save thread")
    
    def _autosave_worker(self):
        """Worker function for auto-save thread."""
        while not self.stop_autosave.is_set():
            now = datetime.now()
            time_since_last_save = now - self.last_autosave_time
            
            # Check if it's time for an auto-save
            if time_since_last_save >= timedelta(minutes=self.autosave_interval):
                logger.info("Auto-saving checkpoint...")
                self.save_checkpoint(self.current_episode, is_autosave=True)
                self.last_autosave_time = now
            
            # Sleep for a short time before checking again
            for _ in range(30):  # Check every 2 seconds
                if self.stop_autosave.is_set():
                    break
                time.sleep(2)
    
    def _clean_old_autosaves(self):
        """Remove old auto-save files to keep only the most recent ones."""
        autosaves = sorted(self.checkpoint_dir.glob("autosave_*.pt"), key=os.path.getmtime, reverse=True)
        
        # Keep only the most recent auto-saves
        if len(autosaves) > self.max_backups:
            for old_save in autosaves[self.max_backups:]:
                try:
                    # Also remove the corresponding metadata file
                    metadata_path = self.checkpoint_path_to_metadata_path(old_save)
                    if metadata_path.exists():
                        os.remove(metadata_path)
                    
                    os.remove(old_save)
                    logger.debug(f"Removed old auto-save: {old_save}")
                except Exception as e:
                    logger.warning(f"Failed to remove old auto-save {old_save}: {str(e)}")
    
    def checkpoint_path_to_metadata_path(self, checkpoint_path):
        """Convert a checkpoint path to its corresponding metadata path."""
        return checkpoint_path.with_suffix('.meta')
    
    def update_best_reward(self, reward):
        """Update the best reward seen so far.
        
        Args:
            reward (float): New reward to compare against best
            
        Returns:
            bool: Whether this is a new best reward
        """
        if reward > self.best_reward:
            self.best_reward = reward
            return True
        return False
    
    def update_current_episode(self, episode):
        """Update the current episode number."""
        self.current_episode = episode

def setup_signal_handlers(checkpoint_manager):
    """Set up signal handlers for graceful termination.
    
    Args:
        checkpoint_manager (CheckpointManager): Manager to handle saving on termination
    """
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, saving checkpoint before exit...")
        
        # Save a final checkpoint
        checkpoint_manager.save_checkpoint(
            checkpoint_manager.current_episode, 
            is_final=True
        )
        
        # Stop the auto-save thread
        checkpoint_manager.stop_autosave_thread()
        
        logger.info("Checkpoint saved, exiting...")
        sys.exit(0)
    
    # Register handlers for common termination signals
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    # On Windows, SIGBREAK is sent when the user closes the console window
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, signal_handler)

def train():
    """Train the agent."""
    args = parse_args()
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Starting training with arguments: {args}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load hardware config
    config = HardwareConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device="cpu" if args.force_cpu else "auto",
        use_fp16=args.fp16
    )
    
    # Create environment and agent
    env = CitiesEnvironment(
        config=config,
        mock_mode=args.mock_env,
        menu_screenshot_path=None  # Use default menu detection
    )
    
    # Get state and action dimensions from the environment
    if env.mock_mode:
        # For mock mode, get dimensions from mock frame (default 3x180x320)
        state_shape = (3, 180, 320)
    else:
        # Try to get actual frame dimensions from a reset or current frame
        initial_frame = env.reset()
        state_shape = initial_frame.shape
    
    # Get action dimensions from the environment
    action_dim = env.num_actions
    
    logger.info(f"State shape: {state_shape}, Action dim: {action_dim}")
    
    # Create a config object with RL parameters if needed
    config.gamma = args.gamma
    config.gae_lambda = args.gae_lambda
    config.clip_param = args.clip_param
    config.value_coef = 0.5
    config.entropy_coef = 0.01
    
    # Initialize the agent with the config
    agent = PPOAgent(
        state_dim=state_shape,
        action_dim=action_dim,
        config=config
    )
    
    # Connect environment and agent for better coordination
    env.agent = agent
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        agent=agent,
        env=env,
        autosave_interval=args.autosave_interval,
        max_backups=args.backup_checkpoints
    )
    
    # Set up signal handlers for graceful termination
    setup_signal_handlers(checkpoint_manager)
    
    # Handle checkpoint resumption (auto-resume by default unless fresh_start is specified)
    start_episode = 0
    checkpoint_loaded = False
    best_reward = float("-inf")
    
    # Check if we have any checkpoints available
    best_exists = (checkpoint_dir / "best_model.pt").exists()
    latest = find_latest_checkpoint(checkpoint_dir)
    has_checkpoints = best_exists or latest is not None
    
    # Determine if we should try to resume
    should_resume = has_checkpoints and not args.fresh_start
    
    if should_resume:
        # First try loading the best model if specifically requested
        if args.use_best and best_exists:
            logger.info("Auto-resuming from best checkpoint...")
            episode = checkpoint_manager.load_checkpoint(use_best=True)
            if episode is not None:
                checkpoint_loaded = True
                logger.info(f"Successfully resumed from best checkpoint (episode {episode})")
        # Otherwise try loading the latest checkpoint
        elif latest:
            episode_num, _ = latest
            logger.info(f"Auto-resuming from latest checkpoint (episode {episode_num})...")
            episode = checkpoint_manager.load_checkpoint()
            if episode is not None:
                start_episode = episode
                checkpoint_loaded = True
                logger.info(f"Successfully resumed from episode {episode}")
        
        if not checkpoint_loaded:
            logger.warning("No valid checkpoints found or loading failed. Starting training from scratch.")
    else:
        if args.fresh_start and has_checkpoints:
            logger.info("Checkpoints found but --fresh_start flag specified. Starting training from scratch.")
        elif not has_checkpoints:
            logger.info("No existing checkpoints found. Starting new training run.")
    
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
    
    # Start auto-save thread
    checkpoint_manager.start_autosave_thread()
    
    # Training loop
    try:
        if checkpoint_loaded:
            # Try to find the current best reward if we're resuming
            best_model_path = checkpoint_dir / "best_model.pt"
            if best_model_path.exists():
                # Just a heuristic: assume the best model has a better reward than starting fresh
                best_reward = 0  # This will be updated on the first better reward
                logger.info(f"Found existing best model, setting initial best_reward to {best_reward}")
        
        logger.info("Starting training loop...")
        
        for episode in range(start_episode, args.num_episodes):
            checkpoint_manager.update_current_episode(episode)
            
            # Collect trajectory
            episode_start_time = time.time()
            experiences, episode_reward, steps_taken = collect_trajectory(
                env=env,
                agent=agent,
                max_steps=args.max_steps
            )
            episode_time = time.time() - episode_start_time
            
            # Update the agent
            if experiences:
                agent.update(experiences)
            
            # Log episode results
            logger.info(f"Episode {episode+1}/{args.num_episodes} - Reward: {episode_reward:.2f}, Steps: {steps_taken}, Time: {episode_time:.2f}s")
            
            # Update wandb if enabled
            if args.use_wandb:
                wandb.log({
                    "episode": episode,
                    "reward": episode_reward,
                    "steps": steps_taken,
                    "time": episode_time
                })
            
            # Save checkpoint if best reward
            if checkpoint_manager.update_best_reward(episode_reward):
                checkpoint_manager.save_checkpoint(episode, is_best=True)
            
            # Save periodic checkpoint
            if (episode + 1) % args.checkpoint_freq == 0:
                checkpoint_manager.save_checkpoint(episode + 1)
            
            print(f"Episode {episode+1}/{args.num_episodes} - Reward: {episode_reward:.2f}")
            
            # Add a delay between episodes to prevent system resource issues
            time.sleep(2.0)
        
        # Save final model
        checkpoint_manager.save_checkpoint(args.num_episodes, is_final=True)
        logger.info("Training completed")
    
    except Exception as e:
        logger.error(f"Training interrupted by exception: {str(e)}")
        # Save emergency checkpoint
        emergency_path = checkpoint_manager.save_checkpoint(
            checkpoint_manager.current_episode, 
            is_autosave=True
        )
        logger.info(f"Emergency checkpoint saved to {emergency_path}")
        raise
    
    finally:
        # Stop auto-save thread
        checkpoint_manager.stop_autosave_thread()
        
        # Close environment
        env.close()

if __name__ == "__main__":
    train() 