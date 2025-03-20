# Architecture Overview

## System Components

The Cities Skylines 2 agent is designed with a modular architecture that separates concerns into distinct components:

### Core Modules

- **Environment**: Handles game interaction and state management
  - **Core**: Basic environment interfaces and state tracking
  - **Input**: Keyboard and mouse simulation
  - **Menu**: Menu detection and navigation
  - **Rewards**: Reward calculation based on visual feedback

- **Agent**: Implements the reinforcement learning algorithm
  - **Core**: PPO agent implementation with policy and value functions
  - **Memory**: Experience storage and replay
  - **Policy**: Policy network and action selection
  - **Value**: Value network for state evaluation

- **Model**: Neural network architecture
  - **OptimizedNetwork**: CNN-based network optimized for visual processing

- **Training**: Training infrastructure
  - **Trainer**: Manages the training loop
  - **Checkpointing**: Handles saving and loading model states
  - **Signal Handling**: Manages graceful interruption

- **Utils**: Utility functions and services
  - **Image Utils**: Image processing utilities
  - **Hardware Monitor**: System resource monitoring
  - **Performance Safeguards**: Ensures stable performance

## Interaction Flow

1. The **Environment** captures the screen and processes it into an observation
2. The **Agent** receives the observation and selects an action
3. The **Environment** executes the action via the input simulation
4. The **Environment** computes the reward based on visual changes
5. The training loop continues this process while the **Training** module handles checkpointing and monitoring

## Design Principles

- **Pure Visual Learning**: The agent learns only from raw pixels
- **Modular Components**: Each component has a single responsibility
- **Hardware Optimization**: Code is optimized for GPU acceleration
- **Robustness**: The system can recover from unexpected states 