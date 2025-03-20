# Cities Skylines 2 Autonomous Agent

![Cities Skylines 2](https://image.api.playstation.com/vulcan/ap/rnd/202306/0816/3214e62bc2f655c5417a0a3dcaafdbc62d9447ebb58919b7.jpg)

## 🌆 Overview

This project implements an autonomous reinforcement learning agent that learns to play Cities Skylines 2 through pure visual observation and keyboard/mouse inputs. The agent operates with no access to the game's internal state or API, using only what it can "see" on screen to make decisions.

## 🧠 Technical Architecture

The project uses a deep reinforcement learning approach with the following components:

### Core Components

- **Environment (src/environment/game_env.py)**: Captures game state through screen capture, processes observations, and manages interactions with the game.
- **Agent (src/agent/ppo_agent.py)**: Implements the PPO (Proximal Policy Optimization) reinforcement learning algorithm.
- **Model (src/model/optimized_network.py)**: Neural network architecture for policy and value functions.
- **Menu Management (src/environment/menu/)**: Detects and handles in-game menus automatically.
- **Input Simulation (src/environment/input/)**: Simulates keyboard and mouse inputs to control the game.
- **Reward System (src/environment/rewards/)**: Computes rewards based solely on visual changes in the environment.
- **Training Module (src/training/)**: Manages the training process, checkpoints, and signal handling.

### Training Infrastructure

- **Checkpoint Management**: Automatic saving, loading, and managing of training checkpoints.
- **Hardware Optimization**: Adapts to available hardware resources for optimal performance.
- **Performance Monitoring**: Tracks system resource usage during training.
- **Crash Recovery**: Detects and recovers from game crashes or unexpected states.

## 🚀 Installation

### Prerequisites

- Windows 10/11
- NVIDIA GPU (RTX 3080 Ti recommended)
- Cities: Skylines 2
- Python 3.10+

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sfatkhutdinov/cities-skylines-2.git
   cd cities-skylines-2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Cities: Skylines 2 and ensure it's running in windowed mode.

## 🎮 Usage

### Basic Training

Start training with default parameters:

```bash
python -m src.train
```

### Advanced Options

```bash
# Train with custom parameters
python -m src.train --num_episodes 2000 --learning_rate 3e-4 --gamma 0.995

# Resume from checkpoint
python -m src.train --checkpoint_dir checkpoints

# Use best checkpoint instead of latest
python -m src.train --checkpoint_dir checkpoints --use_best

# Enable Weights & Biases logging
python -m src.train --use_wandb

# Force CPU usage (not recommended)
python -m src.train --force_cpu
```

### Monitoring

Training metrics are logged in the `logs/` directory. If enabled, training can also be monitored through Weights & Biases.

## 🔧 Configuration

The agent's behavior can be customized through various configuration files:

- **Hardware Configuration** (src/config/hardware_config.py): Adjust resource utilization and hardware-specific settings.
- **Action Space** (src/config/action_space.py): Modify the available actions the agent can take.

## 🛡️ Safeguards

The system includes several safeguards:

- **Performance Monitoring**: Prevents resource exhaustion by adjusting capture rate and model complexity.
- **Game Crash Detection**: Identifies when the game is no longer running and handles recovery.
- **Automatic Saving**: Periodically saves training progress to prevent data loss.
- **Signal Handling**: Gracefully shuts down on user termination signals (Ctrl+C).

## 🧪 Design Principles

This agent strictly adheres to the following principles:

1. **Raw Input Only**: The agent receives only raw pixel data from the screen.
2. **Pure Autonomy**: All internal representations are learned, not engineered.
3. **Human-like Interaction**: Actions are performed through simulated keyboard and mouse inputs.
4. **End-to-End Learning**: Direct mapping from pixels to actions through reinforcement learning.
5. **Emergent Rewards**: Reward signals are derived from visual outcomes, not game metrics.

## 📊 Project Structure

The project is organized into several key directories:

- `checkpoints/` - Saved model checkpoints
- `logs/` - Training logs
- `src/` - Source code
  - `agent/` - Reinforcement learning agent
    - `core/` - Core agent functionality
  - `config/` - Configuration settings
  - `environment/` - Game environment interaction
    - `core/` - Core environment functionality
    - `input/` - Input simulation (keyboard, mouse, actions)
      - `keyboard.py` - Keyboard input simulation
      - `mouse.py` - Mouse input simulation
      - `actions.py` - High-level actions
      - `tracking.py` - Input tracking
    - `menu/` - Menu detection and handling
      - `detector.py` - Menu detection
      - `navigator.py` - Menu navigation
      - `recovery.py` - Menu recovery
      - `templates.py` - Menu templates
      - `menu_handler.py` - Menu handling integration
    - `rewards/` - Reward computation system
      - `reward_system.py` - Main reward system integration
      - `metrics.py` - Reward-related metrics tracking
      - `analyzers.py` - Visual change analysis
      - `calibration.py` - Reward normalization and calibration
      - `world_model.py` - Predictive world model for rewards
  - `model/` - Neural network architecture
  - `training/` - Training processes
    - `checkpointing.py` - Model checkpointing
    - `signal_handlers.py` - Signal handling for training
    - `trainer.py` - Main training loop
    - `utils.py` - Training utilities

## 🔄 Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Developed with ❤️ for Cities: Skylines 2 

## Modularization Progress

The codebase has been modularized to improve maintainability and organization. 
Current progress:

- ✅ Training module modularization
- ✅ Menu handling modularization
- ✅ Input simulation modularization
- ✅ Reward system modularization
- ⬜ Core environment modularization
- ⬜ Agent module modularization 