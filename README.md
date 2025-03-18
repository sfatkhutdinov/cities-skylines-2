# Autonomous Cities: Skylines 2 Agent

An end-to-end reinforcement learning agent that learns to play Cities: Skylines 2 using only raw pixel inputs and simulated keyboard/mouse actions, with no access to game internals or metrics.

## Project Overview

This project implements a fully autonomous agent for Cities: Skylines 2 using deep reinforcement learning. Key features:

- **Pure Visual Learning**: Only uses raw screen pixels as input (no engineered features)
- **Human-like Interaction**: Outputs simulated keyboard and mouse actions through Windows APIs
- **End-to-End Architecture**: Complete processing pipeline from visual perception to action execution
- **Self-supervised Learning**: Uses intrinsic curiosity and autonomous reward signals

## System Architecture

The system consists of several interconnected components:

- **Environment (src/environment/)**
  - `game_env.py`: Main interface to the game world
  - `optimized_capture.py`: High-performance screen capture
  - `input_simulator.py`: Keyboard/mouse simulation
  - `autonomous_reward_system.py`: Self-supervised reward generation
  - `visual_metrics.py`: Computer vision for UI detection

- **Agent (src/agent/)**
  - `ppo_agent.py`: Proximal Policy Optimization implementation
  - `curiosity_module.py`: Intrinsic curiosity and exploration

- **Model (src/model/)**
  - `optimized_network.py`: Neural network architecture with GPU optimization

- **Configuration (src/config/)**
  - `hardware_config.py`: Hardware and training configurations

## Setup Instructions

### Prerequisites

- Windows 10 or later
- NVIDIA GPU with CUDA support (RTX 3080 or better recommended)
- Cities: Skylines 2 (legally purchased)
- Python 3.9+

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/cities-skylines-2-agent.git
cd cities-skylines-2-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Prepare a screenshot of the game menu (optional but recommended):
```
Place a screenshot of the game menu as "menu_reference.png" in the root directory
```

### Hardware Configuration

The system automatically detects your hardware capabilities and configures for optimal performance. You can adjust settings in `src/config/hardware_config.py`.

## Usage

### Basic Training

1. Launch Cities: Skylines 2 and ensure the game window is visible
2. Start the training process:

```bash
python -m src.train --num_episodes 1000 --checkpoint_freq 100
```

### Advanced Options

```bash
python -m src.train --help
```

Common parameters:
- `--num_episodes`: Number of training episodes
- `--max_steps`: Maximum steps per episode
- `--resolution`: Screen capture resolution (default: "1920x1080")
- `--device`: Computing device (default: "cuda" if available)
- `--checkpoint_dir`: Directory for saving model checkpoints
- `--resume`: Resume from the last checkpoint
- `--capture_menu`: Capture a menu screenshot during startup

## Implementation Details

### Vision System

- Screen is captured at full resolution (1920x1080)
- Downsampled to processing resolution (typically 320x240)
- Processed through convolutional layers for feature extraction
- Features feed into policy and value networks

### Action Space

The agent can perform a wide range of in-game actions:
- Camera navigation (pan, zoom, rotate)
- Menu interactions
- Building placement and zoning
- Road construction
- Service management

### Learning Process

- Uses Proximal Policy Optimization (PPO) algorithm
- Implements Generalized Advantage Estimation (GAE)
- Features intrinsic motivation through a curiosity module
- Autonomous reward system based on visual progress indicators

### Optimizations

- Tensor core utilization for NVIDIA RTX GPUs
- Mixed-precision training for faster computation
- Parallel environment processing
- Adaptive frame skipping based on scene complexity

## Performance Tuning

Adjust these parameters for your specific hardware:

- Lower `resolution` on less powerful GPUs
- Increase `frame_skip` to reduce processing load
- Reduce `batch_size` if experiencing memory issues
- Enable/disable `mixed_precision` based on GPU capabilities

## Troubleshooting

### Common Issues

- **Game Window Not Found**: Ensure Cities: Skylines 2 is running and visible
- **CUDA Out of Memory**: Lower batch size or processing resolution
- **Agent Not Learning**: Check reward signals and increase exploration rate

### Logging

Detailed logs are saved to help diagnose issues. Check console output for real-time information.

## Future Work

- Implement curriculum learning for progressive skill acquisition
- Enhance visual understanding for better city planning
- Add memory mechanisms for long-term strategy development
- Incorporate imitation learning from human gameplay examples

## Project Principles

This project adheres to these core principles:
1. The agent must only use raw pixels as input (no game metrics)
2. All learning must be done autonomously (no human-engineered rewards)
3. Interaction must simulate human input devices (keyboard/mouse only)
4. The system must be end-to-end with no shortcuts or simplifications

## License

MIT License

Copyright (c) 2025 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgments

- OpenAI for reinforcement learning foundations
- PyTorch team for deep learning framework
- Colossal Order for creating Cities: Skylines 2 