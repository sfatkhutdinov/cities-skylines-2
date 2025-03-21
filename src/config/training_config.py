"""
Training configuration management for the Cities Skylines 2 agent.

This module defines the TrainingConfig class that encapsulates all 
training-related parameters and hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union


@dataclass
class TrainingConfig:
    """Configuration class for training parameters.
    
    This class contains all hyperparameters and settings related to the 
    training process, including learning rates, batch sizes, and optimization settings.
    """
    
    # Episode parameters
    num_episodes: int = 1000
    max_steps: int = 1000
    early_stop_reward: Optional[float] = 500.0
    
    # Learning parameters
    batch_size: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    
    # PPO parameters
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    
    # Checkpoint parameters
    checkpoint_dir: str = "checkpoints"
    checkpoint_freq: int = 100
    autosave_interval: int = 15  # minutes
    max_checkpoints: int = 10
    
    # Logging and visualization
    log_dir: str = "logs"
    render: bool = False
    visualize: bool = True
    log_interval: int = 10
    
    # Advanced parameters
    use_mixed_precision: bool = True
    optimizer: str = "adam"
    weight_decay: float = 0.0
    
    def __post_init__(self):
        """Initialize derived fields after initialization."""
        # Any post-initialization logic can go here
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create a TrainingConfig instance from a dictionary."""
        return cls(**config_dict)
    
    def update(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown training config parameter: {key}") 