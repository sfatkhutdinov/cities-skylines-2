# Cities Skylines 2 Autonomous Agent

![Cities Skylines 2](https://image.api.playstation.com/vulcan/ap/rnd/202306/0816/3214e62bc2f655c5417a0a3dcaafdbc62d9447ebb58919b7.jpg)

## 🌆 Overview

This project implements an autonomous reinforcement learning agent that learns to play Cities Skylines 2 through pure visual observation and keyboard/mouse inputs. The agent operates with no access to the game's internal state or API, using only what it can "see" on screen to make decisions.

## 🧠 Technical Architecture

The project uses a deep reinforcement learning approach with the following components:

### Core Components

- **Environment**: Captures game state through screen capture, processes observations, and manages interactions with the game.
  - `environment/core`: Core environment interfaces and infrastructure
  - `environment/input`: Keyboard and mouse input simulation
  - `environment/menu`: Menu detection and navigation
  - `environment/rewards`: Reward computation based on visual changes
  - `environment/mock_environment.py`: Simulated environment for testing without the game
  - `environment/optimized_capture.py`: Optimized screen capture implementation
  - `environment/visual_metrics.py`: Visual metrics and measurements

- **Agent**: Implements the PPO (Proximal Policy Optimization) reinforcement learning algorithm.
  - `agent/core`: Core agent components (policy, value, memory, updater)
  - `agent/memory_agent.py`: Memory-augmented agent implementation
  - `agent/hierarchical_agent.py`: Hierarchical agent architecture

- **Memory**: Implements memory-augmented architectures for enhanced agent capabilities.
  - `memory/memory_augmented_network.py`: Neural memory architecture
  - `memory/episodic_memory.py`: Episodic memory functionality

- **Model**: Neural network architecture for policy and value functions.
  - `model/optimized_network.py`: Optimized CNN network for visual processing
  - `model/visual_understanding_network.py`: Visual scene understanding
  - `model/world_model.py`: World modeling for predictions
  - `model/error_detection_network.py`: Error detection for the game

- **Training**: Manages the training process, checkpoints, and signal handling.
  - `training/trainer.py`: Training loop and management
  - `training/checkpointing.py`: Checkpoint saving and loading
  - `training/signal_handlers.py`: Handles interrupts and signals
  - `training/memory_trainer.py`: Trainer for memory-augmented agent
  - `training/hierarchical_trainer.py`: Trainer for hierarchical agent

- **Utils**: Utility functions and services including monitoring capabilities.
  - `utils/image_utils.py`: Image processing utilities
  - `utils/hardware_monitor.py`: System resource monitoring
  - `utils/performance_safeguards.py`: Ensures stable performance
  - `utils/visualization.py`: Visualization tools
  - `utils/path_utils.py`: Path management for consistent file locations

- **Config**: Configuration for hardware and action space.
  - `config/hardware_config.py`: Hardware configuration
  - `config/action_space.py`: Action space definition
  - `config/training_config.py`: Training parameters and settings
  - `config/config_loader.py`: Configuration loading utilities

- **Benchmarks**: Tools for performance analysis and optimization.
  - `benchmarks/benchmark_agent.py`: Measures agent performance metrics

- **Tests**: Automated testing infrastructure.
  - `tests/test_mock_environment.py`: Tests for the mock environment

### Project Structure

```
cities-skylines-2/
├── src/                                # Source code directory
│   ├── agent/                          # Agent modules
│   │   ├── core/                       # Core agent components
│   │   │   ├── memory.py               # Memory buffer implementation
│   │   │   ├── policy.py               # Policy network and action selection
│   │   │   ├── ppo_agent.py            # PPO algorithm implementation
│   │   │   ├── updater.py              # Network update logic
│   │   │   └── value.py                # Value function implementation
│   │   ├── hierarchical_agent.py       # Hierarchical agent architecture
│   │   └── memory_agent.py             # Memory-augmented agent implementation
│   │
│   ├── benchmarks/                     # Benchmarking tools
│   │   └── benchmark_agent.py          # Agent performance evaluation
│   │
│   ├── config/                         # Configuration
│   │   ├── defaults/                   # Default configuration files
│   │   ├── action_space.py             # Action space definition
│   │   ├── config_loader.py            # Configuration loading utilities
│   │   ├── example_config.json         # Example configuration file
│   │   ├── hardware_config.py          # Hardware-specific configuration
│   │   └── training_config.py          # Training parameters and settings
│   │
│   ├── environment/                    # Environment modules
│   │   ├── core/                       # Core environment components
│   │   ├── input/                      # Keyboard and mouse input simulation
│   │   ├── menu/                       # Menu detection and navigation
│   │   ├── rewards/                    # Reward computation
│   │   ├── mock_environment.py         # Simulated environment
│   │   ├── optimized_capture.py        # Screen capture optimization
│   │   └── visual_metrics.py           # Visual-based metrics calculation
│   │
│   ├── memory/                         # Memory-augmented architectures
│   │   ├── episodic_memory.py          # Episodic memory implementation
│   │   └── memory_augmented_network.py # Neural network with memory capabilities
│   │
│   ├── model/                          # Neural network models
│   │   ├── error_detection_network.py  # Error detection for the game
│   │   ├── optimized_network.py        # Optimized CNN architecture
│   │   ├── visual_understanding_network.py # Visual scene understanding
│   │   └── world_model.py              # World modeling for predictions
│   │
│   ├── training/                       # Training infrastructure
│   │   ├── checkpointing.py            # Model checkpoint management
│   │   ├── hierarchical_trainer.py     # Trainer for hierarchical agent
│   │   ├── memory_trainer.py           # Trainer for memory-augmented agent
│   │   ├── signal_handlers.py          # Handles system signals during training
│   │   ├── trainer.py                  # Base trainer implementation
│   │   └── utils.py                    # Training utility functions
│   │
│   ├── tests/                          # Test scripts
│   │   └── test_mock_environment.py    # Tests for the mock environment
│   │
│   ├── utils/                          # Utility functions and monitoring
│   │   ├── hardware_monitor.py         # System resource monitoring
│   │   ├── image_utils.py              # Image processing utilities
│   │   ├── path_utils.py               # Path management for consistent file locations
│   │   ├── performance_safeguards.py   # Ensures stable performance
│   │   └── visualization.py            # Data visualization tools
│   │
│   ├── __init__.py                     # Package initialization
│   └── train.py                        # Main training entry point
│
├── docs/                               # Documentation
│   ├── agent.md                        # Agent documentation
│   ├── architecture.md                 # System architecture overview
│   ├── environment.md                  # Environment documentation
│   ├── improvements.md                 # Future improvements
│   ├── model.md                        # Model documentation
│   ├── README.md                       # Documentation overview
│   └── training.md                     # Training process documentation
│
├── scripts/                            # Utility scripts
│   ├── benchmark.py                    # Performance benchmarking
│   ├── dashboard.py                    # Real-time monitoring dashboard
│   ├── hyperparameter_tuning.py        # Hyperparameter optimization
│   ├── run_environment.py              # Run the environment standalone
│   ├── run_mock_training.py            # Training with mock environment
│   └── visualize_training.py           # Training visualization tools
│
├── .github/                            # GitHub configuration
├── .gitignore                          # Git ignore rules
├── CONTRIBUTING.md                     # Contribution guidelines
├── LICENSE                             # Project license
├── README.md                           # Main README (this file)
└── requirements.txt                    # Python dependencies
```

**Note**: The following directories are created at runtime:
- `logs/`: Generated log files and tensorboard data
- `output/`: Generated outputs, visualizations, and benchmark results
- `checkpoints/`: Model checkpoints during and after training

These directories are automatically created in the project root when needed by the application. All paths are managed by the path utilities in `src/utils/path_utils.py`, ensuring consistent file locations regardless of which directory you run the scripts from.

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
cd cities-skylines-2-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

### Training

To start training the agent:

```bash
python src/train.py
```

Common options:
```
--num_episodes=1000     # Number of episodes to train
--max_steps=1000        # Maximum steps per episode
--checkpoint_dir=checkpoints  # Directory to save checkpoints
--render                # Display visualization during training
--mock_env              # Use mock environment for testing
--mixed_precision       # Enable mixed precision training
--hardware_config=path/to/config.json  # Custom hardware configuration
```

### Testing with Mock Environment

The project includes a mock environment for testing without requiring the actual game:

```bash
python src/tests/test_mock_environment.py
```

This will run a series of tests on the mock environment, including:
- Basic functionality tests
- Complete episode simulation
- Error condition handling (crashes, freezes, menus)
- Visualization generation

The mock environment simulates:
- City building mechanics
- Population and budget dynamics
- Game crashes and freezes
- Menu interactions
- Reward computation

### Benchmarking

Run comprehensive benchmarks on the agent and environment:

```bash
python src/benchmarks/benchmark_agent.py --episodes=10 --steps=500 --output=benchmark_results
```

Common options:
```
--episodes=10           # Number of episodes to run
--steps=500             # Maximum steps per episode
--config=path/to/config.json  # Configuration file
--output=benchmark_results  # Output directory name
--gpu                   # Force GPU usage
--cpu                   # Force CPU usage
--mixed_precision       # Use mixed precision
```

The benchmark will generate:
- Performance metrics (rewards, episode lengths, success rates)
- Hardware utilization statistics (CPU, GPU, memory)
- Visualizations of agent performance
- Detailed JSON and text reports

### Utility Scripts

The project includes several utility scripts to help with development, analysis, and monitoring:

#### Hyperparameter Tuning

Optimize agent hyperparameters:

```bash
python scripts/hyperparameter_tuning.py --method=random --trials=10 --visualize
```

Common options:
```
--output=hyperparameter_results  # Output directory for results
--method=[grid|random]  # Search method (grid or random search)
--trials=10             # Number of trials for random search
--mock                  # Use mock environment
--epochs=5              # Training epochs per trial
--episodes=5            # Episodes per epoch
--parallel=4            # Number of parallel trials to run
--visualize             # Generate visualizations of results
```

#### Dashboard

Run a real-time dashboard to monitor agent performance:

```bash
streamlit run scripts/dashboard.py -- --log_dir=logs
```

Common options:
```
--log_dir=logs          # Directory containing training logs
--port=8501             # Port to run the dashboard on
--host=localhost        # Host to run the dashboard on
--refresh_interval=30   # Dashboard refresh interval in seconds
```

## 📋 Documentation

See the [docs](docs/) directory for detailed documentation on each component.

## 🧪 Testing

Run the tests with:

```bash
python -m unittest discover tests
```

## 📊 Performance Optimization Features

The agent includes several performance optimization features:

- **Adaptive Resource Management**: Automatically adjusts resource usage based on system capabilities
- **Mixed Precision Training**: Reduces memory usage and increases training speed on compatible GPUs
- **Error Recovery**: Handles game crashes and freezes gracefully to continue training
- **Hardware Monitoring**: Tracks system resource usage to identify bottlenecks
- **Performance Benchmarking**: Tools to measure and optimize agent performance

## 📈 Monitoring

Training progress is logged to the `logs` directory. You can also use Weights & Biases for more detailed monitoring:

```bash
python src/train.py --use_wandb
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. 