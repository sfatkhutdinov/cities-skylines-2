# Agent Module

The Agent module implements the reinforcement learning algorithm used to train the Cities Skylines 2 agent. It uses the Proximal Policy Optimization (PPO) algorithm, a state-of-the-art policy gradient method.

## Core Components

### PPOAgent

The main `PPOAgent` class (`agent/core/ppo_agent.py`) serves as the primary interface for the agent. Key methods include:

- `select_action(state)`: Selects an action given the current state
- `update(batch)`: Updates the policy and value networks using collected experience
- `save(path)`: Saves the agent's state to disk
- `load(path)`: Loads the agent's state from disk

### Policy

The `Policy` class (`agent/core/policy.py`) implements the policy network, which maps observations to action probabilities. Features include:

- Actor network architecture
- Action distribution management
- Entropy calculation for exploration
- Policy clipping for stable updates

### ValueFunction

The `ValueFunction` class (`agent/core/value.py`) implements the value network, which estimates the expected return from a given state. Features include:

- Critic network architecture
- Value estimation
- Value clipping for stable updates
- Advantage estimation

### Memory

The `Memory` class (`agent/core/memory.py`) handles experience collection and batching for training. Features include:

- Experience buffer management
- Trajectory storage and retrieval
- Generalized Advantage Estimation (GAE)
- Batch creation for mini-batch training

### PPOUpdater

The `PPOUpdater` class (`agent/core/updater.py`) handles the training logic for the PPO algorithm. Features include:

- Loss calculation (policy loss, value loss, entropy loss)
- Learning rate management
- Gradient clipping
- Optimization steps

## Algorithm Overview

The PPO algorithm used by the agent follows these steps:

1. **Collect Experience**: Interact with the environment to collect trajectories
2. **Compute Advantages**: Calculate advantages using Generalized Advantage Estimation
3. **Update Policy**: Update the policy network to maximize the clipped objective
4. **Update Value Function**: Update the value network to better estimate expected returns
5. **Repeat**: Continue collecting experience and updating the networks

## Hyperparameters

The agent uses the following key hyperparameters:

- `learning_rate`: Learning rate for the optimizer
- `gamma`: Discount factor for future rewards
- `gae_lambda`: Parameter for Generalized Advantage Estimation
- `clip_param`: Clipping parameter for policy gradient
- `value_coef`: Coefficient for value loss
- `entropy_coef`: Coefficient for entropy loss
- `max_grad_norm`: Maximum gradient norm for clipping
- `num_epochs`: Number of epochs to train on each batch of experience
- `batch_size`: Batch size for training

## Usage Examples

### Basic Usage

```python
from src.agent.core import PPOAgent
from src.environment.core import Environment
from src.config.hardware_config import HardwareConfig

# Create environment and agent
config = HardwareConfig()
env = Environment(config)
agent = PPOAgent(
    state_dim=env.observation_shape,
    action_dim=env.action_space.n,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_param=0.2,
    device=config.get_device()
)

# Training loop
obs = env.reset()
for i in range(1000):
    # Select action
    action = agent.select_action(obs)
    
    # Execute action
    next_obs, reward, done, info = env.step(action)
    
    # Record experience
    agent.record(obs, action, reward, next_obs, done)
    
    # Update observation
    obs = next_obs
    
    # Update policy if enough steps have been taken
    if agent.is_update_ready():
        agent.update()
    
    if done:
        obs = env.reset()

# Save trained agent
agent.save("checkpoints/agent.pt")
``` 