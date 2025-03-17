"""
Direct GPU test that checks only the hardware_config component.
This avoids importing the entire environment to prevent circular dependencies.
"""

import sys
import os
import time
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import hardware_config directly
try:
    # Direct import with no intermediate imports that might cause conflicts
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    from config.hardware_config import HardwareConfig
    print("Successfully imported HardwareConfig module")
except ImportError as e:    2r
    print(f"Error importing HardwareConfig: {e}")
    sys.exit(1)

def test_gpu():
    print("=== PyTorch GPU Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Test tensor creation on GPU
        x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        print(f"Test tensor on GPU: {x}")
        print(f"Tensor device: {x.device}")
    else:
        print("CUDA is not available. Using CPU only.")
    
    print("\n=== Hardware Config Test ===")
    config = HardwareConfig()
    print(f"Device: {config.device}")
    print(f"CUDA available: {config.use_cuda}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Tensor cores: {config.tensor_cores}")
    print(f"Parallel environments: {config.num_parallel_envs}")
    
    # Test tensor creation with config
    device = config.get_device()
    dtype = config.get_dtype()
    print(f"\nConfig device: {device}")
    print(f"Config dtype: {dtype}")
    
    y = torch.tensor([4.0, 5.0, 6.0], device=device, dtype=dtype)
    print(f"Test tensor with config: {y}")
    print(f"Tensor device: {y.device}")
    print(f"Tensor dtype: {y.dtype}")
    
    # Test autocast/amp context
    print("\n=== Testing AMP Context ===")
    with config.get_amp_context():
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        start_time = time.time()
        c = a @ b
        end_time = time.time()
    
    print(f"Matrix operation time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Result shape: {c.shape}")
    print(f"Result device: {c.device}")
    
    # Test GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated() / 1e9
        print(f"\nInitial GPU memory: {initial_mem:.4f} GB")
        
        # Optimize for inference
        config.optimize_for_inference()
        print("Optimized for inference")
        
        # Test large allocation
        large_tensor = torch.randn(5000, 5000, device=device, dtype=dtype)
        used_mem = torch.cuda.memory_allocated() / 1e9
        print(f"Memory usage after allocation: {used_mem:.4f} GB")
        
        # Clean up
        del large_tensor
        torch.cuda.empty_cache()
        final_mem = torch.cuda.memory_allocated() / 1e9
        print(f"Memory after cleanup: {final_mem:.4f} GB")

if __name__ == "__main__":
    test_gpu() 