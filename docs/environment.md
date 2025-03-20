# Environment Module

The Environment module is responsible for interacting with the Cities: Skylines 2 game, capturing observations, executing actions, and computing rewards.

## Core Components

### Environment

The main `Environment` class (`environment/core/environment.py`) serves as the primary interface between the agent and the game. It implements the standard reinforcement learning environment interface with the following methods:

- `reset()`: Resets the environment to an initial state
- `step(action)`: Executes an action and returns observation, reward, done flag, and info
- `close()`: Cleans up resources when the environment is no longer needed

### ObservationManager

The `ObservationManager` class (`environment/core/observation.py`) handles capturing and processing screen images from the game. Key features include:

- Frame capture using optimized screen capture methods
- Frame preprocessing (resizing, normalization, etc.)
- Frame stacking for temporal information
- Converting raw pixel data into agent-friendly observations

### ActionSpace

The `ActionSpace` class (`environment/core/action_space.py`) defines the possible actions that the agent can take. It includes:

- Definition of primitive actions (keyboard/mouse inputs)
- Compound actions that combine multiple primitives
- Action sampling for exploration
- Action validation and execution

### GameState

The `GameState` class (`environment/core/game_state.py`) tracks the state of the game, including:

- Game speed
- Current game mode (normal, pause, etc.)
- Game metrics (derived from visual information)
- Episode status

### PerformanceMonitor

The `PerformanceMonitor` class (`environment/core/performance.py`) tracks system performance during training to ensure stability:

- Frame rate monitoring
- GPU/CPU usage
- Memory usage
- Dynamic adjustment of capture and processing parameters

## Input Simulation

The input simulation components (`environment/input/`) handle sending keyboard and mouse inputs to the game:

- `KeyboardController`: Simulates keyboard inputs (keypresses, key combinations)
- `MouseController`: Simulates mouse inputs (movement, clicks, scrolling)
- `ActionExecutor`: Maps high-level actions to low-level input commands
- `InputTracker`: Tracks input history and prevents conflicting inputs

## Menu Handling

The menu handling components (`environment/menu/`) detect and navigate in-game menus:

- `MenuDetector`: Detects when the game is showing a menu
- `MenuNavigator`: Navigates through menu options
- `MenuRecovery`: Recovers from stuck menu situations
- `MenuTemplateManager`: Manages menu templates for recognition
- `MenuHandler`: Integrates all menu-related functionality

## Reward System

The reward system components (`environment/rewards/`) compute rewards based on visual changes:

- `AutonomousRewardSystem`: Main reward computation system
- `VisualChangeAnalyzer`: Detects and quantifies visual changes in the game
- `WorldModelCNN`: Predictive model for expected visual changes
- `VisualMetricsEstimator`: Estimates game metrics from visual information
- `RewardCalibrator`: Calibrates and normalizes rewards

## Usage Examples

### Basic Usage

```python
from src.environment.core import Environment
from src.config.hardware_config import HardwareConfig

# Create environment
config = HardwareConfig()
env = Environment(config)

# Reset environment
obs = env.reset()

# Run for 1000 steps
for i in range(1000):
    action = 0  # No-op action
    obs, reward, done, info = env.step(action)
    if done:
        break

# Clean up
env.close()
```

### With Random Actions

```python
# After creating environment
obs = env.reset()
for i in range(1000):
    # Take random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        break
``` 