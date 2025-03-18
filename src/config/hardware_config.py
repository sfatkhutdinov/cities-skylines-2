"""
Hardware and training configuration for Cities: Skylines 2.
"""

import torch
import torch.cuda.amp as amp
from typing import Tuple, Dict, Any, Optional

class HardwareConfig:
    """Configuration for hardware and training parameters."""
    
    def __init__(
        self,
        learning_rate: float = 3e-4,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        resolution: Tuple[int, int] = (1080, 1920),
        ppo_epochs: int = 4,
        clip_range: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        mixed_precision: bool = True,
        tensor_cores: bool = True,
        cudnn_benchmark: bool = True,
        pin_memory: bool = True,
        frame_stack: int = 4,
        frame_skip: int = 2,
        min_fps: float = 30.0,
        target_fps: float = 60.0,
        num_parallel_envs: int = 8
    ):
        """Initialize configuration.
        
        Args:
            learning_rate (float): Learning rate for optimizer
            batch_size (int): Batch size for training
            device (str): Device to use (cuda/cpu)
            resolution (tuple): Input resolution (height, width)
            ppo_epochs (int): Number of PPO epochs per update
            clip_range (float): PPO clip range
            value_loss_coef (float): Value loss coefficient
            entropy_coef (float): Entropy coefficient
            max_grad_norm (float): Maximum gradient norm
            gamma (float): Discount factor
            gae_lambda (float): GAE lambda parameter
            mixed_precision (bool): Whether to use mixed precision training
            tensor_cores (bool): Whether to use tensor cores if available
            cudnn_benchmark (bool): Whether to use cuDNN benchmark mode
            pin_memory (bool): Whether to pin memory for faster GPU transfer
            frame_stack (int): Number of frames to stack
            frame_skip (int): Number of frames to skip between observations
            min_fps (float): Minimum FPS to maintain
            target_fps (float): Target FPS for training
            num_parallel_envs (int): Number of parallel environments to run
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        self.resolution = resolution
        self.ppo_epochs = ppo_epochs
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Hardware settings
        self.use_cuda = 'cuda' in device and torch.cuda.is_available() and torch.cuda.device_count() > 0
        self.mixed_precision = mixed_precision and self.use_cuda
        self.tensor_cores = tensor_cores and self.use_cuda
        self.cudnn_benchmark = cudnn_benchmark and self.use_cuda
        self.pin_memory = pin_memory and self.use_cuda
        
        # Frame processing
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        self.min_fps = min_fps
        self.target_fps = target_fps
        
        # Environment settings
        self.num_parallel_envs = num_parallel_envs if self.use_cuda else 1

        # CUDA Acceleration settings
        self.amp_enabled = self.mixed_precision
        self.amp_dtype = torch.float16
        self.scaler = None
        
        # Configure CUDA if available
        if self.use_cuda:
            try:
                # Check CUDA device count
                cuda_device_count = torch.cuda.device_count()
                if cuda_device_count == 0:
                    print("Warning: CUDA is available but no CUDA devices found.")
                    self.use_cuda = False
                    self.device = "cpu"
                    self.mixed_precision = False
                    self.tensor_cores = False
                    self.pin_memory = False
                    self.num_parallel_envs = 1
                else:
                    # Get and print detailed GPU information
                    print(f"GPU: {torch.cuda.get_device_name(0)}")
                    print(f"CUDA Version: {torch.version.cuda}")
                    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
                    print(f"Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
                    
                    # Set thread configuration for optimal performance
                    torch.set_num_threads(16)  # Adjust based on CPU core count
                    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                        torch.cuda.set_per_process_memory_fraction(0.9)  # Reserve 90% of GPU memory
                    
                    # Check GPU compute capability
                    capability = torch.cuda.get_device_capability(0)
                    if capability[0] < 7:
                        print("Warning: GPU compute capability < 7.0. Some optimizations will be disabled.")
                        self.tensor_cores = False
                        self.mixed_precision = False
                    else:
                        print(f"GPU Compute Capability {capability[0]}.{capability[1]} supports tensor cores and mixed precision")
                    
                    # Configure PyTorch CUDA optimizations
                    torch.backends.cudnn.benchmark = self.cudnn_benchmark
                    torch.backends.cudnn.deterministic = False  # Better performance, less reproducibility
                    torch.backends.cuda.matmul.allow_tf32 = self.tensor_cores
                    torch.backends.cudnn.allow_tf32 = self.tensor_cores
                    
                    # Setup Automatic Mixed Precision if enabled
                    if self.mixed_precision:
                        self.scaler = torch.amp.GradScaler()
                        print("Automatic Mixed Precision (AMP) enabled")
                    
                    # Optimize memory allocation strategy
                    if hasattr(torch.cuda, 'memory_stats'):
                        print("CUDA memory allocation strategy: caching allocator enabled")
                    
                    # Empty cache to start fresh
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Warning: CUDA initialization failed: {e}")
                self.device = "cpu"
                self.use_cuda = False
                self.mixed_precision = False
                self.tensor_cores = False
                self.pin_memory = False
                self.num_parallel_envs = 1
        
        # Print configuration summary
        print(f"Running in {'GPU' if self.use_cuda else 'CPU'}-only mode.")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Tensor cores: {self.tensor_cores}")
        print(f"Number of parallel environments: {self.num_parallel_envs}")
        
    def get_device(self) -> torch.device:
        """Get PyTorch device."""
        return torch.device(self.device)
        
    def get_dtype(self) -> torch.dtype:
        """Get the appropriate dtype based on configuration."""
        if self.use_cuda and self.mixed_precision:
            return self.amp_dtype
        return torch.float32
        
    def optimize_for_inference(self) -> None:
        """Configure PyTorch for optimal inference performance."""
        if self.use_cuda:
            # Set inference mode specific optimizations
            torch.backends.cudnn.benchmark = True
            with torch.cuda.device(self.get_device()):
                torch.cuda.empty_cache()
                
    def get_amp_context(self):
        """Get the appropriate autocast context for mixed precision."""
        if self.use_cuda and self.mixed_precision:
            return torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype)
        else:
            # Return a dummy context manager that does nothing
            return torch.no_grad()