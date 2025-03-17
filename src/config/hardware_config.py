"""
Hardware and training configuration for Cities: Skylines 2.
"""

import torch
from typing import Tuple

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
        num_parallel_envs: int = 4
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
        self.use_cuda = 'cuda' in device
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
        
        # Configure CUDA if available
        if self.use_cuda:
            try:
                # Check GPU compute capability
                if torch.cuda.get_device_properties(0).major < 7:
                    print("Warning: GPU compute capability < 7.0. Some optimizations will be disabled.")
                    self.tensor_cores = False
                    self.mixed_precision = False
                
                # Configure PyTorch
                torch.backends.cudnn.benchmark = self.cudnn_benchmark
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
            except Exception as e:
                print(f"Warning: CUDA initialization failed: {e}")
                self.device = "cpu"
                self.use_cuda = False
                self.mixed_precision = False
                self.tensor_cores = False
                self.pin_memory = False
                self.num_parallel_envs = 1
        
        # Print configuration
        print(f"Running in {'GPU' if self.use_cuda else 'CPU'}-only mode. Adjusting configuration...")
        
    def get_device(self) -> torch.device:
        """Get PyTorch device."""
        return torch.device(self.device)
        
    def get_dtype(self) -> torch.dtype:
        """Get the appropriate dtype based on configuration."""
        if self.use_cuda and self.mixed_precision:
            return torch.float16
        return torch.float32