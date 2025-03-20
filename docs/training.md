# Training Module

The Training module handles the reinforcement learning training process for the Cities Skylines 2 agent, including checkpoint management, signal handling, and training loop implementation.

## Core Components

### Trainer

The `Trainer` class (`training/trainer.py`) is the main component that manages the training loop. Key features include:

- End-to-end training management
- Experience collection and agent updates
- Episode tracking and logging
- Performance monitoring

### CheckpointManager

The `CheckpointManager` class (`training/checkpointing.py`) handles saving and loading model checkpoints. Features include:

- Regular checkpoint saving at specified intervals
- Best model tracking based on cumulative reward
- Automatic checkpoint pruning to manage disk space
- Loading from checkpoint for training resumption

### SignalHandlers

The `signal_handlers` module (`training/signal_handlers.py`) provides utilities for handling system signals during training. Features include:

- Graceful interruption handling (Ctrl+C)
- Automatic checkpointing before exit
- Resource cleanup on shutdown
- Automatic saving thread management

### Training Utilities

The `utils` module (`training/utils.py`) provides supporting functionality for training:

- Command-line argument parsing
- Hardware configuration
- Environment and agent setup
- Utility functions for training

## Training Process

The training process follows these steps:

1. **Initialization**:
   - Parse command-line arguments
   - Configure hardware settings
   - Set up environment and agent
   - Initialize checkpoint manager

2. **Training Loop**:
   - Reset environment to start new episode
   - Collect experiences by agent-environment interaction
   - Update agent policy and value function
   - Save checkpoints at specified intervals
   - Log training progress and metrics

3. **Termination**:
   - Detect completion criteria (episode limit, target reward)
   - Handle interruption signals
   - Save final checkpoint
   - Clean up resources

## Command-Line Options

The training script (`src/train.py`) supports the following key options:

- `--num_episodes`: Number of episodes to train for
- `--max_steps`: Maximum steps per episode
- `--batch_size`: Batch size for updates
- `--learning_rate`: Learning rate
- `--gamma`: Discount factor
- `--checkpoint_dir`: Directory to save checkpoints
- `--checkpoint_freq`: Episodes between checkpoints
- `--autosave_interval`: Minutes between auto-saves
- `--mock_env`: Use mock environment for testing
- `--use_wandb`: Use Weights & Biases for logging
- `--render`: Render environment during training
- `--force_cpu`: Force CPU usage even if GPU is available

## Usage Examples

### Basic Training

```bash
python src/train.py --num_episodes 1000 --max_steps 1000
```

### Resume from Checkpoint

```bash
python src/train.py --checkpoint_dir checkpoints
```

### Distributed Training

```python
# Example of manually setting up distributed training
from src.training.trainer import Trainer
from src.config.hardware_config import HardwareConfig
from src.environment.core import Environment
from src.agent.core import PPOAgent
import torch.distributed as dist

# Initialize distributed environment
dist.init_process_group(backend='nccl')
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)

# Set up environment and agent
config = HardwareConfig()
env = Environment(config)
agent = PPOAgent(
    state_dim=env.observation_shape,
    action_dim=env.action_space.n,
    device=f"cuda:{local_rank}"
)

# Create trainer
trainer = Trainer(agent, env, config, {"num_episodes": 1000})

# Start training
trainer.train()
``` 