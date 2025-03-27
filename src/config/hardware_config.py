"""
Hardware and training configuration for Cities: Skylines 2.
"""

import torch
import torch.cuda.amp as amp
from typing import Tuple, Dict, Any, Optional
import logging
import contextlib

logger = logging.getLogger(__name__)

class HardwareConfig:
    """Configuration for hardware resources and training parameters."""
    
    def __init__(
        self,
        # Hardware params
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        device: str = "auto",
        resolution: Tuple[int, int] = (84, 84),
        frame_stack: int = 4,
        frame_skip: int = 2,
        use_fp16: bool = True,
        cpu_threads: int = 0,
        force_cpu: bool = False,

        # Training loop params
        num_episodes: int = 1000,
        max_steps: int = 1000,
        early_stop_reward: Optional[float] = None,
        update_frequency: int = 2048,

        # Optimizer params
        optimizer: str = "adam",
        weight_decay: float = 0.0,

        # PPO params
        gamma: float = 0.99,
        ppo_epochs: int = 4,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,

        # Checkpointing params
        checkpoint_dir: str = "checkpoints",
        checkpoint_freq: int = 100,
        max_checkpoints: int = 5,

        # Logging / Visualization
        tensorboard_dir: str = "logs",
        visualizer_update_interval: int = 10,
        use_wandb: bool = False,

        # Performance Safeguards
        monitor_hardware: bool = True,
        min_fps: int = 10,
        max_memory_usage: float = 0.9,
        safeguard_cooldown: int = 60,

        # Memory settings
        memory: Optional[Dict[str, Any]] = None,

        # Hierarchical settings
        hierarchical: Optional[Dict[str, Any]] = None,
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
            num_episodes (int): Number of training episodes
            max_steps (int): Maximum number of steps per episode
            early_stop_reward (Optional[float]): Reward threshold for early stopping
            update_frequency (int): Frequency of updates in training loop
            optimizer (str): Optimizer to use
            weight_decay (float): Weight decay for optimizer
            gamma (float): Discount factor for PPO
            ppo_epochs (int): Number of PPO epochs
            clip_param (float): Clipping parameter for PPO
            value_loss_coef (float): Value loss coefficient for PPO
            entropy_coef (float): Entropy coefficient for PPO
            max_grad_norm (float): Maximum gradient norm for PPO
            gae_lambda (float): Lambda parameter for GAE
            checkpoint_dir (str): Directory for saving checkpoints
            checkpoint_freq (int): Frequency of checkpoint saving
            max_checkpoints (int): Maximum number of checkpoints to keep
            tensorboard_dir (str): Directory for TensorBoard logs
            visualizer_update_interval (int): Interval for visualizer updates
            use_wandb (bool): Whether to use Weights & Biases for logging
            monitor_hardware (bool): Whether to monitor hardware resources
            min_fps (int): Minimum frames per second for monitoring
            max_memory_usage (float): Maximum memory usage for monitoring
            safeguard_cooldown (int): Cooldown period for safeguard actions
            memory (Optional[Dict[str, Any]]): Memory settings
            hierarchical (Optional[Dict[str, Any]]): Hierarchical settings
        """
        # Hardware params
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._device_str = device
        self.resolution = resolution
        self.frame_stack = max(1, frame_stack)
        self.frame_skip = max(1, frame_skip)
        self.use_fp16 = use_fp16 and self._supports_fp16()
        self.use_mixed_precision = self.use_fp16
        self.cpu_threads = cpu_threads
        self.force_cpu = force_cpu

        # Training loop params
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.early_stop_reward = early_stop_reward
        self.update_frequency = update_frequency

        # Optimizer params
        self.optimizer = optimizer
        self.weight_decay = weight_decay

        # PPO params
        self.gamma = gamma
        self.ppo_epochs = ppo_epochs
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda

        # Checkpointing params
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints

        # Logging / Visualization
        self.tensorboard_dir = tensorboard_dir
        self.visualizer_update_interval = visualizer_update_interval
        self.use_wandb = use_wandb

        # Performance Safeguards
        self.monitor_hardware = monitor_hardware
        self.min_fps = min_fps
        self.max_memory_usage = max_memory_usage
        self.safeguard_cooldown = safeguard_cooldown

        # Memory settings
        self.memory = memory if memory is not None else {
            'enabled': True,
            'memory_size': 2000,
            'key_size': 128,
            'value_size': 256,
            'retrieval_threshold': 0.5,
            'warmup_episodes': 10,
            'use_curriculum': True,
            'curriculum_phases': {
                'observation': 10,
                'retrieval': 30,
                'integration': 50,
                'refinement': 100
            },
            'memory_use_probability': 0.8
        }

        # Hierarchical settings
        self.hierarchical = hierarchical if hierarchical is not None else {
            'enabled': True,
            'feature_dim': 512,
            'latent_dim': 256,
            'prediction_horizon': 5,
            'adaptive_memory_use': True,
            'adaptive_memory_threshold': 0.7,
            'training_schedules': {
                'visual_network': 10,
                'world_model': 5,
                'error_detection': 20
            },
            'batch_sizes': {
                'visual_network': 32,
                'world_model': 64,
                'error_detection': 32
            },
            'progressive_training': True,
            'progressive_phases': {
                'visual_network': 50,
                'world_model': 100,
                'error_detection': 150
            }
        }

        # Auto-configure device, dtype, threads
        self._device = self._configure_device()
        self._dtype = torch.float16 if self.use_fp16 else torch.float32
        if self.cpu_threads > 0:
            torch.set_num_threads(self.cpu_threads)
            logger.info(f"Set CPU threads to {self.cpu_threads}")

        logger.info(f"Hardware config initialized - Device: {self._device}, Resolution: {self.resolution}, "
                   f"Batch size: {self.batch_size}, FP16: {self.use_fp16}")
    
    def _configure_device(self) -> torch.device:
        """Configure the device to use based on availability.
        
        Returns:
            torch.device: Configured device
        """
        if self.force_cpu:
            return torch.device("cpu")

        if self._device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
        else:
            try:
                device = torch.device(self._device_str)
                if device.type == "cuda" and not torch.cuda.is_available():
                    logger.warning("CUDA requested but not available, falling back to CPU")
                    device = torch.device("cpu")
                else:
                    logger.info(f"Using specified device: {device}")
            except Exception as e:
                logger.error(f"Invalid device string {self._device_str}: {e}, falling back to CPU")
                device = torch.device("cpu")
                
        return device
    
    def _supports_fp16(self) -> bool:
        """Check if FP16 is supported on the current device.
        
        Returns:
            bool: True if FP16 is supported, False otherwise
        """
        device = self._configure_device()
        if device.type != "cuda":
            return False
            
        try:
            capability = torch.cuda.get_device_capability(device)
            return capability[0] >= 7  # Volta and newer have good FP16 support
        except Exception as e:
            logger.warning(f"Error checking FP16 support: {e}")
            return False
    
    def get_device(self) -> torch.device:
        """Get the configured device."""
        return self._device
    
    def get_dtype(self) -> torch.dtype:
        """Get the data type based on mixed precision setting."""
        return self._dtype
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HardwareConfig':
        """Create configuration from dictionary.
        
        Args:
            config_dict (Dict[str, Any]): Configuration dictionary
            
        Returns:
            HardwareConfig: Configuration object
        """
        init_keys = cls.__init__.__code__.co_varnames[1:cls.__init__.__code__.co_argcount]
        filtered_dict = {k: v for k, v in config_dict.items() if k in init_keys}
        return cls(**filtered_dict)

    def optimize_for_inference(self) -> None:
        """Configure PyTorch for optimal inference performance."""
        if self._device.type == "cuda":
            # Set inference mode specific optimizations
            torch.backends.cudnn.benchmark = True
            with torch.cuda.device(self.get_device()):
                torch.cuda.empty_cache()
                
    def get_amp_context(self):
        """Get the appropriate autocast context for mixed precision."""
        if self.use_mixed_precision and self._device.type == 'cuda':
            return torch.cuda.amp.autocast()
        else:
            @contextlib.contextmanager
            def nullcontext():
                yield
            return nullcontext()

    def is_memory_enabled(self) -> bool:
        """Check if memory augmentation is enabled."""
        return self.memory.get('enabled', False)

    def is_hierarchical_enabled(self) -> bool:
        """Check if hierarchical learning is enabled."""
        return self.hierarchical.get('enabled', False)