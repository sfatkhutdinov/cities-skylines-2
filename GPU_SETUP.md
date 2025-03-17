# GPU Acceleration for Cities: Skylines 2 Agent

This document explains how to use the GPU acceleration features in the Cities: Skylines 2 agent.

## Hardware Requirements

- NVIDIA GPU with CUDA support (RTX 3080 Ti or similar recommended)
- CUDA toolkit 11.x or later installed
- 16+ GB system RAM recommended

## Testing GPU Acceleration

Several test scripts are provided to verify that GPU acceleration is working correctly:

1. **Basic GPU Test**:
   ```bash
   python test_gpu.py
   ```
   This tests that PyTorch can access the GPU and the hardware configuration is working.

2. **Visual Metrics Test**:
   ```bash
   python test_visual_metrics.py
   ```
   This tests that the visual metrics estimator is correctly using the GPU.

3. **Game Environment Test**:
   ```bash
   python test_game_env.py
   ```
   This tests the full game environment with GPU acceleration in mock mode.

## GPU Acceleration Features

### Mixed Precision Training

The agent uses automatic mixed precision (AMP) to speed up training and reduce memory usage. This uses FP16 (half-precision) operations where possible while maintaining FP32 (single-precision) for critical operations.

### Tensor Cores

On NVIDIA GPUs with Tensor Cores (Volta, Turing, Ampere, or newer architectures), the agent automatically uses TF32 precision for matrix multiplications, providing a significant speedup.

### Memory Optimization

The hardware configuration includes memory optimization features:
- Automatic caching allocator for efficient GPU memory usage
- Memory fraction control to avoid out-of-memory errors
- Automatic cleanup for unused tensors

## Using GPU Acceleration in Your Code

To use GPU acceleration in your code:

```python
from src.config.hardware_config import HardwareConfig

# Create a hardware config with GPU acceleration
config = HardwareConfig()

# Check if CUDA is available
if config.use_cuda:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Get device and dtype for tensor creation
device = config.get_device()
dtype = config.get_dtype()

# Create tensors on the GPU with the right precision
tensor = torch.randn(100, 100, device=device, dtype=dtype)

# Use automatic mixed precision for operations
with config.get_amp_context():
    result = tensor @ tensor.T
```

## Troubleshooting

If you encounter issues with GPU acceleration:

1. Check that your CUDA drivers are up to date
2. Ensure PyTorch is installed with CUDA support
3. Run `python test_gpu.py` to verify basic GPU functionality
4. Check GPU memory usage with `nvidia-smi` while running the agent

For more advanced configurations, modify parameters in `src/config/hardware_config.py`. 