"""
Hardware and training configuration for Cities: Skylines 2.
"""

import torch
import torch.cuda.amp as amp
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class HardwareConfig:
    """Configuration for hardware resources and training parameters."""
    
    def __init__(
        self,
        batch_size: int = 64,
        learning_rate: float = 1e-4, 
        device: str = "auto",
        resolution: Tuple[int, int] = (320, 240),
        frame_stack: int = 1,
        frame_skip: int = 2,
        use_fp16: bool = False,
        cpu_threads: int = 0,  # 0 means use all available
        force_cpu: bool = False
    ):
        """Initialize hardware configuration.
        
        Args:
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate for optimizer
            device (str): Device to use for training ('auto', 'cuda', 'cuda:0', 'cpu', etc)
            resolution (Tuple[int, int]): Resolution to scale frames to (height, width)
            frame_stack (int): Number of frames to stack
            frame_skip (int): Number of frames to skip between actions
            use_fp16 (bool): Whether to use half precision (FP16)
            cpu_threads (int): Number of CPU threads to use (0 = all available)
            force_cpu (bool): Whether to force CPU usage
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._device_str = device
        self.resolution = resolution
        self.frame_stack = max(1, frame_stack)
        self.frame_skip = max(1, frame_skip)
        self.use_fp16 = use_fp16 and self._supports_fp16()
        self.cpu_threads = cpu_threads
        self.force_cpu = force_cpu
        
        # RL-specific parameters
        self.gamma = 0.99
        self.ppo_epochs = 4
        self.clip_range = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.gae_lambda = 0.95
        
        # Auto-configure device
        self._device = self._configure_device()
        self._dtype = torch.float16 if self.use_fp16 else torch.float32
        
        # Log configuration
        logger.info(f"Hardware config initialized - Device: {self._device}, Resolution: {resolution}, "
                   f"Batch size: {batch_size}, FP16: {self.use_fp16}")
        
        # Set CPU threads if specified
        if self.cpu_threads > 0:
            torch.set_num_threads(self.cpu_threads)
            logger.info(f"Set CPU threads to {self.cpu_threads}")
    
    def _configure_device(self) -> torch.device:
        """Configure the device to use based on availability.
        
        Returns:
            torch.device: Configured device
        """
        if self._device_str == "auto":
            # Auto-detect optimal device
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
        else:
            # Use specified device
            try:
                device = torch.device(self._device_str)
                if device.type == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    device = torch.device("cpu")
            except Exception as e:
                logger.error(f"Invalid device {self._device_str}: {e}, falling back to CPU")
                device = torch.device("cpu")
                
        return device
    
    def _supports_fp16(self) -> bool:
        """Check if FP16 is supported on the current device.
        
        Returns:
            bool: True if FP16 is supported, False otherwise
        """
        if not torch.cuda.is_available():
            return False
            
        try:
            # Check specific GPU compute capability for FP16 support
            cuda_device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(cuda_device)
            major, minor = capability
            
            # Pascal (6.x) and higher architectures support FP16 well
            return major >= 6
        except Exception as e:
            logger.warning(f"Error checking FP16 support: {e}")
            return False
    
    def get_device(self) -> torch.device:
        """Get the appropriate device for computation.
        
        Returns:
            torch.device: Device to use for computation
        """
        if torch.cuda.is_available() and not self.force_cpu:
            # Always use cuda:0 for consistency
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    
    def get_dtype(self) -> torch.dtype:
        """Get the data type to use for training.
        
        Returns:
            torch.dtype: Data type to use
        """
        return self._dtype
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": str(self._device),
            "resolution": self.resolution,
            "frame_stack": self.frame_stack,
            "frame_skip": self.frame_skip,
            "use_fp16": self.use_fp16,
            "cpu_threads": self.cpu_threads,
            "gamma": self.gamma,
            "ppo_epochs": self.ppo_epochs,
            "clip_range": self.clip_range,
            "value_loss_coef": self.value_loss_coef,
            "entropy_coef": self.entropy_coef,
            "max_grad_norm": self.max_grad_norm,
            "gae_lambda": self.gae_lambda
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HardwareConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict (Dict[str, Any]): Configuration dictionary
            
        Returns:
            HardwareConfig: Configuration object
        """
        # Extract core parameters
        instance = cls(
            batch_size=config_dict.get("batch_size", 64),
            learning_rate=config_dict.get("learning_rate", 1e-4),
            device=config_dict.get("device", "auto"),
            resolution=config_dict.get("resolution", (320, 240)),
            frame_stack=config_dict.get("frame_stack", 1),
            frame_skip=config_dict.get("frame_skip", 2),
            use_fp16=config_dict.get("use_fp16", False),
            cpu_threads=config_dict.get("cpu_threads", 0),
            force_cpu=config_dict.get("force_cpu", False)
        )
        
        # Set additional parameters
        if "gamma" in config_dict:
            instance.gamma = config_dict["gamma"]
        if "ppo_epochs" in config_dict:
            instance.ppo_epochs = config_dict["ppo_epochs"]
        if "clip_range" in config_dict:
            instance.clip_range = config_dict["clip_range"]
        if "value_loss_coef" in config_dict:
            instance.value_loss_coef = config_dict["value_loss_coef"]
        if "entropy_coef" in config_dict:
            instance.entropy_coef = config_dict["entropy_coef"]
        if "max_grad_norm" in config_dict:
            instance.max_grad_norm = config_dict["max_grad_norm"]
        if "gae_lambda" in config_dict:
            instance.gae_lambda = config_dict["gae_lambda"]
            
        return instance

    def optimize_for_inference(self) -> None:
        """Configure PyTorch for optimal inference performance."""
        if self._device.type == "cuda":
            # Set inference mode specific optimizations
            torch.backends.cudnn.benchmark = True
            with torch.cuda.device(self.get_device()):
                torch.cuda.empty_cache()
                
    def get_amp_context(self):
        """Get the appropriate autocast context for mixed precision."""
        if self._device.type == "cuda" and self.use_fp16:
            return torch.amp.autocast(device_type='cuda', dtype=self._dtype)
        else:
            # Return a dummy context manager that does nothing
            return torch.no_grad()