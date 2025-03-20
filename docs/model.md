# Model Module

The Model module contains neural network architectures used by the agent to process observations and make decisions in the Cities Skylines 2 environment.

## Core Components

### OptimizedNetwork

The `OptimizedNetwork` class (`model/optimized_network.py`) provides an optimized neural network architecture for visual processing. Features include:

- Convolutional neural network (CNN) architecture
- Optimized for processing screen captures from the game
- Efficient memory usage and computation
- Hardware acceleration support (CUDA, mixed precision)

## Network Architecture

The network architecture is composed of several components:

### Feature Extractor

The feature extractor processes raw pixel inputs through a series of convolutional layers:

1. **ConvBlock 1**: Input → 32 channels
2. **ConvBlock 2**: 32 → 64 channels
3. **ConvBlock 3**: 64 → 128 channels
4. **ConvBlock 4**: 128 → 256 channels

Each ConvBlock consists of:
- 2D convolution
- Batch normalization
- SiLU (Swish) activation function

### Policy Head

The policy head maps features to action probabilities:

1. Flattened features
2. Fully connected layer: features → 512 units
3. Fully connected layer: 512 → 256 units
4. Fully connected layer: 256 → action_dim units
5. Softmax activation for probability distribution

### Value Head

The value head estimates state values:

1. Flattened features
2. Fully connected layer: features → 512 units
3. Fully connected layer: 512 → 256 units
4. Fully connected layer: 256 → 1 unit

## Optimizations

The neural network includes several optimizations for performance:

- **Batch Normalization**: Improves training stability and speed
- **SiLU Activation**: Better gradient flow than ReLU
- **Residual Connections**: Helps with gradient propagation in deeper networks
- **Dynamic Shape Adaptation**: Automatically adapts to different input shapes
- **Mixed Precision**: Support for FP16 training when available
- **Memory Efficiency**: Optimized architecture to reduce VRAM usage

## Usage Examples

### Basic Usage

```python
import torch
from src.model.optimized_network import OptimizedNetwork

# Create network
input_shape = (3, 84, 84)  # (channels, height, width)
num_actions = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

network = OptimizedNetwork(input_shape, num_actions, device)

# Process observation
observation = torch.randn(1, *input_shape).to(device)
action_probs, value = network(observation)

# Select action
action = torch.multinomial(action_probs, 1).item()
```

### Manual Feature Extraction

```python
# Create network
network = OptimizedNetwork(input_shape, num_actions, device)

# Extract features only
observation = torch.randn(1, *input_shape).to(device)
features = network.extract_features(observation)

# Use features for custom processing
custom_output = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
``` 