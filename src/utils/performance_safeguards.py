"""
Performance safeguards to prevent memory issues and optimize training.
"""

import torch
import psutil
import gc
from typing import Optional, Dict
from src.config.hardware_config import HardwareConfig

class PerformanceSafeguards:
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.warning_threshold = 0.85  # 85% memory usage triggers warnings
        self.critical_threshold = 0.95  # 95% memory usage triggers emergency measures
        self.initial_batch_size = config.batch_size
        
    def check_memory_status(self) -> Dict[str, float]:
        """Check current memory usage status."""
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            max_allocated = torch.cuda.max_memory_allocated()
            if max_allocated > 0:
                gpu_memory_used = torch.cuda.memory_allocated() / max_allocated
            else:
                gpu_memory_used = 0.0
        else:
            gpu_memory_used = 0.0
            
        system_memory_used = psutil.virtual_memory().percent / 100
        
        return {
            'gpu_memory': gpu_memory_used,
            'system_memory': system_memory_used
        }
        
    def apply_emergency_optimization(self) -> str:
        """Apply emergency optimizations when resources are critical."""
        gc.collect()
        torch.cuda.empty_cache()
        
        # Reduce batch size temporarily
        self.config.batch_size = max(self.config.batch_size // 2, 16)
        
        # Enable gradient checkpointing if not already enabled
        torch.backends.cudnn.benchmark = False
        
        # Reduce number of parallel environments
        self.config.num_parallel_envs = max(self.config.num_parallel_envs // 2, 1)
        
        return "Emergency optimizations applied: reduced batch size, enabled gradient checkpointing, reduced parallel environments"
        
    def restore_normal_operation(self):
        """Restore normal operation settings."""
        self.config.batch_size = self.initial_batch_size
        self.config.num_parallel_envs = min(self.config.num_parallel_envs * 2, 8)
        torch.backends.cudnn.benchmark = True
        
    def optimize_memory_usage(self) -> Optional[str]:
        """Check and optimize memory usage."""
        memory_status = self.check_memory_status()
        
        if memory_status['gpu_memory'] > self.critical_threshold or \
           memory_status['system_memory'] > self.critical_threshold:
            return self.apply_emergency_optimization()
            
        elif memory_status['gpu_memory'] > self.warning_threshold or \
             memory_status['system_memory'] > self.warning_threshold:
            # Preventive measures
            gc.collect()
            torch.cuda.empty_cache()
            return "Preventive memory cleanup performed"
            
        return None
        
    def optimize_training_parameters(self, current_fps: float) -> Dict[str, any]:
        """Dynamically adjust training parameters based on performance."""
        adjustments = {}
        
        # Adjust frame skip if FPS is too low
        if current_fps < self.config.target_fps * 0.8:
            self.config.frame_skip = min(self.config.frame_skip + 1, 4)
            adjustments['frame_skip'] = self.config.frame_skip
            
        # Adjust batch size based on GPU memory usage
        memory_status = self.check_memory_status()
        if memory_status['gpu_memory'] < 0.5:  # GPU underutilized
            self.config.batch_size = min(self.config.batch_size * 2, 256)
            adjustments['batch_size'] = self.config.batch_size
            
        return adjustments
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return {
            'memory_usage': self.check_memory_status(),
            'batch_size': self.config.batch_size,
            'frame_skip': self.config.frame_skip,
            'num_parallel_envs': self.config.num_parallel_envs
        } 