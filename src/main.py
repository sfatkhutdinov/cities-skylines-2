import argparse
import logging
import sys
import os
import json
import torch

from src.environment.core.environment import Environment
from src.agent.core.ppo_agent import PPOAgent
from src.agent.memory_agent import MemoryAugmentedAgent
from src.model.optimized_network import OptimizedNetwork
from src.memory.memory_augmented_network import MemoryAugmentedNetwork
from src.training.trainer import Trainer
from src.training.memory_trainer import MemoryTrainer
from src.config.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a Cities: Skylines 2 agent.')
    parser.add_argument('--config', type=str, default='src/config/defaults/default_config.json',
                        help='Path to config file')
    parser.add_argument('--mock', action='store_true',
                        help='Use mock environment for testing')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--memory', action='store_true',
                        help='Use memory-augmented agent')
    parser.add_argument('--memory_size', type=int, default=1000,
                        help='Size of episodic memory (if using memory)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to train for (overrides config)')
    return parser.parse_args()

def main():
    """Main entry point for training."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    
    # Create environment
    env = Environment(args.config, use_mock=args.mock, device=device)
    
    # Create network
    use_memory = args.memory or config.get('memory', {}).get('enabled', False)
    
    if use_memory:
        logger.info("Creating memory-augmented network")
        memory_config = config.get('memory', {})
        memory_size = args.memory_size or memory_config.get('memory_size', 1000)
        
        # Create memory-augmented network
        policy_network = MemoryAugmentedNetwork(
            input_shape=env.observation_space.shape,
            num_actions=env.action_space.n,
            memory_size=memory_size,
            device=device,
            use_lstm=config.get('model', {}).get('use_lstm', True),
            lstm_hidden_size=config.get('model', {}).get('lstm_hidden_size', 256),
            use_attention=config.get('model', {}).get('use_attention', True),
            attention_heads=config.get('model', {}).get('attention_heads', 4)
        )
        
        # Create memory-augmented agent
        agent = MemoryAugmentedAgent(
            policy_network=policy_network,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            memory_size=memory_size,
            memory_use_prob=memory_config.get('memory_use_probability', 0.8),
            lr=config.get('training', {}).get('learning_rate', 5e-5),
            gamma=config.get('training', {}).get('gamma', 0.99),
            epsilon=config.get('training', {}).get('clip_ratio', 0.2),
            value_coef=config.get('training', {}).get('value_coef', 0.5),
            entropy_coef=config.get('training', {}).get('entropy_coef', 0.01)
        )
        
        # Create memory trainer
        trainer = MemoryTrainer(
            agent=agent,
            env=env,
            config_file=args.config
        )
    else:
        logger.info("Creating standard neural network")
        # Create standard network
        policy_network = OptimizedNetwork(
            input_shape=env.observation_space.shape,
            num_actions=env.action_space.n,
            device=device
        )
        
        # Create standard agent
        agent = PPOAgent(
            policy_network=policy_network,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            lr=config.get('training', {}).get('learning_rate', 3e-4),
            gamma=config.get('training', {}).get('gamma', 0.99),
            epsilon=config.get('training', {}).get('clip_ratio', 0.2),
            value_coef=config.get('training', {}).get('value_coef', 0.5),
            entropy_coef=config.get('training', {}).get('entropy_coef', 0.01)
        )
        
        # Create standard trainer
        trainer = Trainer(
            agent=agent,
            env=env,
            config_file=args.config
        )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Get number of episodes from arguments or config
    num_episodes = args.episodes or config.get('training', {}).get('num_episodes', 1000)
    
    # Train agent
    logger.info(f"Starting training for {num_episodes} episodes")
    trainer.train(num_episodes)
    
    # Save final model
    checkpoint_path = os.path.join('checkpoints', 'final_model.pt')
    trainer.save_checkpoint(checkpoint_path)
    logger.info(f"Saved final model to {checkpoint_path}")

if __name__ == '__main__':
    main() 