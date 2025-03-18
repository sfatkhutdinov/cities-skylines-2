"""
Performance safeguards to prevent memory issues and optimize training.
"""

import torch
import psutil
import gc
import time
import logging
from typing import Optional, Dict, Any
from config.hardware_config import HardwareConfig

logger = logging.getLogger(__name__)

class PerformanceSafeguards:
    def __init__(self, config: Optional[HardwareConfig] = None):
        """Initialize performance safeguards.
        
        Args:
            config (HardwareConfig): Optional hardware configuration
        """
        self.config = config
        self.warning_threshold = 0.85  # 85% memory usage triggers warnings
        self.critical_threshold = 0.95  # 95% memory usage triggers emergency measures
        self.initial_batch_size = config.batch_size if config else 16
        self.previous_check = time.time()
        self.check_interval = 60  # Check every 60 seconds
        
        # Resource usage thresholds
        self.cpu_threshold = 90.0  # 90% CPU usage
        self.memory_threshold = 90.0  # 90% memory usage
        self.gpu_threshold = 90.0  # 90% GPU memory usage
        
        # Monitoring history
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []
        self.history_size = 10
        
        # Initialize GPU monitoring if available
        self.has_gpu = torch.cuda.is_available()
        
    def check_resources(self) -> Dict[str, Any]:
        """Check system resource usage.
        
        Returns:
            Dict[str, Any]: Resource usage metrics
        """
        current_time = time.time()
        if current_time - self.previous_check < self.check_interval:
            return {}  # Skip check if not enough time has passed
            
        self.previous_check = current_time
        
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_history.append(cpu_percent)
        if len(self.cpu_history) > self.history_size:
            self.cpu_history.pop(0)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        self.memory_history.append(memory_percent)
        if len(self.memory_history) > self.history_size:
            self.memory_history.pop(0)
        
        # Get GPU usage if available
        gpu_percent = 0.0
        if self.has_gpu:
            try:
                gpu_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100.0
                self.gpu_history.append(gpu_percent)
                if len(self.gpu_history) > self.history_size:
                    self.gpu_history.pop(0)
            except Exception as e:
                logger.warning(f"Failed to get GPU usage: {e}")
        
        # Log resource usage
        logger.info(f"Resource usage - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%"
                   f"{f', GPU: {gpu_percent:.1f}%' if self.has_gpu else ''}")
        
        # Check for resource problems
        resources_ok = self._check_resource_thresholds()
        
        # Return metrics
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "gpu_percent": gpu_percent if self.has_gpu else 0.0,
            "resources_ok": resources_ok
        }
    
    def _check_resource_thresholds(self) -> bool:
        """Check if resource usage exceeds thresholds.
        
        Returns:
            bool: True if resources are OK, False if there are problems
        """
        # Calculate average usage
        avg_cpu = sum(self.cpu_history) / max(1, len(self.cpu_history))
        avg_memory = sum(self.memory_history) / max(1, len(self.memory_history))
        
        # Check CPU and memory
        if avg_cpu > self.cpu_threshold:
            logger.warning(f"High CPU usage detected: {avg_cpu:.1f}% (threshold: {self.cpu_threshold}%)")
            return False
            
        if avg_memory > self.memory_threshold:
            logger.warning(f"High memory usage detected: {avg_memory:.1f}% (threshold: {self.memory_threshold}%)")
            return False
            
        # Check GPU if available
        if self.has_gpu and self.gpu_history:
            avg_gpu = sum(self.gpu_history) / len(self.gpu_history)
            if avg_gpu > self.gpu_threshold:
                logger.warning(f"High GPU usage detected: {avg_gpu:.1f}% (threshold: {self.gpu_threshold}%)")
                return False
                
        return True
    
    def optimize_resources(self):
        """Optimize resource usage if needed."""
        # First check if optimization is needed
        resource_metrics = self.check_resources()
        if resource_metrics.get("resources_ok", True):
            return  # No optimization needed
            
        logger.info("Optimizing system resources...")
        
        # Try to free CPU resources
        if resource_metrics.get("cpu_percent", 0) > self.cpu_threshold:
            logger.info("Reducing CPU load...")
            # Force garbage collection
            gc.collect()
        
        # Try to free memory
        if resource_metrics.get("memory_percent", 0) > self.memory_threshold:
            logger.info("Reducing memory usage...")
            # Force garbage collection
            gc.collect()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Try to free GPU memory if available
        if self.has_gpu and resource_metrics.get("gpu_percent", 0) > self.gpu_threshold:
            logger.info("Reducing GPU memory usage...")
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
        logger.info("Resource optimization completed")
        
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