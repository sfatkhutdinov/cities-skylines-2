"""
Standalone GPU test script for testing CUDA acceleration.
This script avoids importing any modules that might cause circular dependencies.
"""

import os
import sys
import time
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

class HardwareConfigStandalone:
    """Simplified standalone hardware configuration for testing."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision: bool = True,
        tensor_cores: bool = True
    ):
        self.device = device
        self.use_cuda = 'cuda' in device
        self.mixed_precision = mixed_precision and self.use_cuda
        self.tensor_cores = tensor_cores and self.use_cuda
        
        logger.info(f"Auto-selected device: {self.device}")
        
        # Configure CUDA if available
        if self.use_cuda:
            logger.info("Configuring CUDA...")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Detected device count: {torch.cuda.device_count()}")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            
            # Check GPU compute capability
            capability = torch.cuda.get_device_capability(0)
            logger.info(f"Compute capability: {capability[0]}.{capability[1]}")
            
            # Get memory info
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / (1024**3)
            logger.info(f"Memory: {memory_gb:.2f} GB")
            
            # Configure PyTorch
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = self.tensor_cores
            torch.backends.cudnn.allow_tf32 = self.tensor_cores
            
        logger.info(f"Running in {'GPU' if self.use_cuda else 'CPU'}-only mode with device: {self.device}")
    
    def get_device(self):
        """Get device object."""
        return torch.device(self.device)
    
    def get_dtype(self):
        """Get data type for mixed precision."""
        if self.use_cuda and self.mixed_precision:
            return torch.float16
        return torch.float32

def test_gpu():
    """Run comprehensive GPU tests."""
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
    config = HardwareConfigStandalone()
    print(f"Device: {config.device}")
    print(f"CUDA available: {config.use_cuda}")
    print(f"Mixed precision: {config.mixed_precision}")
    
    # Test tensor creation with config
    device = config.get_device()
    dtype = config.get_dtype()
    print(f"\nConfig device: {device}")
    print(f"Config dtype: {dtype}")
    
    y = torch.tensor([4.0, 5.0, 6.0], device=device, dtype=dtype)
    print(f"Test tensor with config: {y}")
    print(f"Tensor device: {y.device}")
    print(f"Tensor dtype: {y.dtype}")
    
    # Performance test
    print("\n=== GPU Operation Test ===")
    start_time = time.time()
    
    # Create large matrices and perform multiplication
    a = torch.randn(1000, 1000, device=device, dtype=dtype)
    b = torch.randn(1000, 1000, device=device, dtype=dtype)
    
    # Use context manager for mixed precision if applicable
    if config.mixed_precision:
        with torch.cuda.amp.autocast():
            c = a @ b
    else:
        c = a @ b
    
    end_time = time.time()
    print(f"Matrix multiplication time: {end_time - start_time:.4f} seconds")
    print(f"Result shape: {c.shape}")
    print(f"Result device: {c.device}")
    
    # Test memory usage
    if torch.cuda.is_available():
        # Clear memory
        torch.cuda.empty_cache()
        print(f"\nInitial GPU memory: {torch.cuda.memory_allocated()/1e9:.4f} GB")
        
        # Allocate large tensor
        large_tensor = torch.randn(8000, 8000, device=device)
        print(f"Memory after allocation: {torch.cuda.memory_allocated()/1e9:.4f} GB")
        
        # Clean up
        del large_tensor
        torch.cuda.empty_cache()
        print(f"Memory after cleanup: {torch.cuda.memory_allocated()/1e9:.4f} GB")

if __name__ == "__main__":
    test_gpu() 