# Training Module

This module contains all the code related to training the reinforcement learning agent for Cities: Skylines 2.

## Structure

The training module is organized into the following components:

- `trainer.py`: Contains the `Trainer` class which handles the core training loop and logic.
- `checkpointing.py`: Contains the `CheckpointManager` class for saving and loading model checkpoints.
- `signal_handlers.py`: Provides functionality for handling signals (Ctrl+C, etc.) and ensuring graceful exit.
- `utils.py`: Utility functions for setting up training, parsing arguments, etc.

## Usage

The main training script (`src/train.py`) uses these modules to orchestrate the training process. To start training, run:

```bash
python src/train.py [options]
```

For a list of available options, run:

```bash
python src/train.py --help
```

## Components

### Trainer

The `Trainer` class is responsible for:
- Collecting experience trajectories
- Updating the agent with collected experiences
- Managing checkpoints
- Tracking training statistics
- Handling logging and visualization

### CheckpointManager

The `CheckpointManager` class handles:
- Saving checkpoints at regular intervals
- Loading checkpoints when resuming training
- Managing disk usage by removing old checkpoints
- Finding best checkpoints based on performance

### Signal Handlers

Signal handling functionality ensures:
- Graceful termination when interrupted
- Automatic saving of progress before exiting
- Cleanup of resources
- Periodic autosaving via background threads

### Utilities

Utility functions provide:
- Command-line argument parsing
- Hardware configuration setup
- Environment and agent setup
- Helper functions for training

## Configuration

Training can be configured using command-line arguments, including:
- Number of episodes and steps
- Learning rate and other hyperparameters
- Checkpointing frequency and behavior
- Hardware acceleration options
- Logging and visualization settings 