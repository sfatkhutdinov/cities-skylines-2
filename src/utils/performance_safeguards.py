"""
Performance safeguards to prevent memory issues and optimize training.
"""

import torch
import psutil
import gc
import time
import logging
import os
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from ..config.hardware_config import HardwareConfig

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
        self.current_batch_size = self.initial_batch_size
        self.previous_check = time.time()
        self.check_interval = 60  # Check every 60 seconds
        self.optimization_count = 0
        self.emergency_mode = False
        
        # Resource usage thresholds
        self.cpu_threshold = 90.0  # 90% CPU usage
        self.memory_threshold = 90.0  # 90% memory usage
        self.gpu_threshold = 90.0  # 90% GPU memory usage
        self.disk_io_threshold = 100.0  # 100 MB/s disk I/O
        
        # Monitoring history
        self.cpu_history: List[float] = []
        self.memory_history: List[float] = []
        self.gpu_history: List[float] = []
        self.disk_io_history: List[float] = []
        self.fps_history: List[float] = []
        self.history_size = 20  # Track more history for better trend analysis
        
        # Initialize GPU monitoring if available
        self.has_gpu = torch.cuda.is_available()
        
        # Training optimization state
        self.use_amp = False  # Automatic Mixed Precision
        self.use_gradient_checkpointing = False
        self.optimizations_applied: List[str] = []
        
        # Performance monitoring start time
        self.start_time = time.time()
        
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
                # Improved GPU memory calculation
                gpu_allocated = torch.cuda.memory_allocated()
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_percent = (gpu_allocated / gpu_total) * 100.0
                self.gpu_history.append(gpu_percent)
                if len(self.gpu_history) > self.history_size:
                    self.gpu_history.pop(0)
            except Exception as e:
                logger.warning(f"Failed to get GPU usage: {e}")
                
        # Get disk I/O usage
        try:
            disk_io = psutil.disk_io_counters()
            if hasattr(self, '_last_disk_io') and self._last_disk_io:
                last_time, last_read, last_write = self._last_disk_io
                read_bytes = disk_io.read_bytes - last_read
                write_bytes = disk_io.write_bytes - last_write
                time_diff = current_time - last_time
                
                if time_diff > 0:
                    # Calculate MB/s
                    io_rate = (read_bytes + write_bytes) / (1024 * 1024) / time_diff
                    self.disk_io_history.append(io_rate)
                    if len(self.disk_io_history) > self.history_size:
                        self.disk_io_history.pop(0)
            
            self._last_disk_io = (current_time, disk_io.read_bytes, disk_io.write_bytes)
        except Exception as e:
            logger.warning(f"Failed to get disk I/O stats: {e}")
        
        # Log resource usage
        logger.info(f"Resource usage - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, "
                   f"GPU: {gpu_percent:.1f}%, Uptime: {(current_time - self.start_time) / 3600:.1f}h")
        
        # Check if resources are above thresholds and apply optimizations if needed
        if self._check_resource_thresholds():
            self.optimize_resources()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_percent': gpu_percent,
            'disk_io': self.disk_io_history[-1] if self.disk_io_history else 0.0,
            'uptime': current_time - self.start_time
        }
        
    def _check_resource_thresholds(self) -> bool:
        """Check if any resource is above threshold.
        
        Returns:
            bool: True if any resource is above threshold
        """
        # Use average of recent history to avoid reacting to spikes
        cpu_avg = np.mean(self.cpu_history[-5:]) if len(self.cpu_history) >= 5 else 0
        memory_avg = np.mean(self.memory_history[-5:]) if len(self.memory_history) >= 5 else 0
        gpu_avg = np.mean(self.gpu_history[-5:]) if len(self.gpu_history) >= 5 else 0
        disk_io_avg = np.mean(self.disk_io_history[-5:]) if len(self.disk_io_history) >= 5 else 0
        
        # Check if any resource is above threshold
        cpu_above = cpu_avg > self.cpu_threshold
        memory_above = memory_avg > self.memory_threshold
        gpu_above = gpu_avg > self.gpu_threshold and self.has_gpu
        disk_io_above = disk_io_avg > self.disk_io_threshold
        
        # Log warnings for resources above threshold
        if cpu_above:
            logger.warning(f"CPU usage above threshold: {cpu_avg:.1f}% > {self.cpu_threshold}%")
        if memory_above:
            logger.warning(f"Memory usage above threshold: {memory_avg:.1f}% > {self.memory_threshold}%")
        if gpu_above:
            logger.warning(f"GPU memory usage above threshold: {gpu_avg:.1f}% > {self.gpu_threshold}%")
        if disk_io_above:
            logger.warning(f"Disk I/O above threshold: {disk_io_avg:.1f} MB/s > {self.disk_io_threshold} MB/s")
        
        return cpu_above or memory_above or gpu_above or disk_io_above
        
    def optimize_resources(self):
        """Apply optimizations to reduce resource usage."""
        # Increment optimization count
        self.optimization_count += 1
        logger.info(f"Applying optimization round {self.optimization_count}")
        
        # Force garbage collection
        gc.collect()
        if self.has_gpu:
            torch.cuda.empty_cache()
            
        # Check memory status and apply optimizations if needed
        memory_status = self.check_memory_status()
        
        # Apply different optimizations based on which resource is constrained
        if np.mean(self.gpu_history[-5:]) > self.gpu_threshold and self.has_gpu:
            # GPU is the bottleneck
            opt_message = self.optimize_gpu_usage()
            if opt_message:
                logger.info(f"Applied GPU optimization: {opt_message}")
                self.optimizations_applied.append(f"GPU: {opt_message}")
                
        elif np.mean(self.memory_history[-5:]) > self.memory_threshold:
            # System memory is the bottleneck
            opt_message = self.optimize_memory_usage()
            if opt_message:
                logger.info(f"Applied memory optimization: {opt_message}")
                self.optimizations_applied.append(f"Memory: {opt_message}")
                
        elif np.mean(self.cpu_history[-5:]) > self.cpu_threshold:
            # CPU is the bottleneck
            opt_message = self.optimize_cpu_usage()
            if opt_message:
                logger.info(f"Applied CPU optimization: {opt_message}")
                self.optimizations_applied.append(f"CPU: {opt_message}")
                
        elif np.mean(self.disk_io_history[-5:]) > self.disk_io_threshold:
            # Disk I/O is the bottleneck
            opt_message = self.optimize_io_usage()
            if opt_message:
                logger.info(f"Applied Disk I/O optimization: {opt_message}")
                self.optimizations_applied.append(f"Disk I/O: {opt_message}")
                
        # If in critical memory state, apply emergency optimizations
        if memory_status.get('gpu_critical', False) or memory_status.get('memory_critical', False):
            msg = self.apply_emergency_optimization()
            logger.warning(f"Applied emergency optimization: {msg}")
            self.emergency_mode = True
            self.optimizations_applied.append(f"Emergency: {msg}")
            
    def check_memory_status(self) -> Dict[str, float]:
        """Check memory status and determine if optimizations are needed.
        
        Returns:
            Dict[str, float]: Memory status information
        """
        memory = psutil.virtual_memory()
        
        # Calculate memory usage ratios
        memory_usage = memory.percent / 100.0
        
        # Check GPU memory if available
        gpu_usage = 0
        if self.has_gpu:
            try:
                gpu_allocated = torch.cuda.memory_allocated()
                gpu_total = torch.cuda.get_device_properties(0).total_memory
                gpu_usage = gpu_allocated / gpu_total
            except Exception as e:
                logger.warning(f"Failed to get GPU memory usage: {e}")
                
        # Determine memory status
        memory_ok = memory_usage < self.warning_threshold
        memory_warning = memory_usage >= self.warning_threshold and memory_usage < self.critical_threshold
        memory_critical = memory_usage >= self.critical_threshold
        
        gpu_ok = not self.has_gpu or gpu_usage < self.warning_threshold
        gpu_warning = self.has_gpu and gpu_usage >= self.warning_threshold and gpu_usage < self.critical_threshold
        gpu_critical = self.has_gpu and gpu_usage >= self.critical_threshold
        
        return {
            'memory_usage': memory_usage,
            'memory_warning': memory_warning,
            'memory_critical': memory_critical,
            'memory_ok': memory_ok,
            'gpu_usage': gpu_usage,
            'gpu_warning': gpu_warning,
            'gpu_critical': gpu_critical,
            'gpu_ok': gpu_ok
        }
        
    def apply_emergency_optimization(self) -> str:
        """Apply emergency optimization measures.
        
        Returns:
            str: Description of action taken
        """
        # We're in a critical memory state, take drastic measures
        gc.collect()
        if self.has_gpu:
            torch.cuda.empty_cache()
            
        # Reduce batch size significantly
        if self.current_batch_size > 1:
            self.current_batch_size = max(1, self.current_batch_size // 2)
            return f"Reduced batch size to {self.current_batch_size} (emergency)"
            
        # Enable gradient checkpointing if not already enabled
        if not self.use_gradient_checkpointing and self.has_gpu:
            self.use_gradient_checkpointing = True
            return "Enabled gradient checkpointing (emergency)"
            
        # Enable AMP if not already enabled and supported
        if not self.use_amp and self.has_gpu and torch.cuda.is_available():
            self.use_amp = True
            return "Enabled automatic mixed precision (emergency)"
            
        return "Already in emergency mode, no further optimizations available"
        
    def restore_normal_operation(self):
        """Restore normal operation if resource usage has decreased."""
        # Check if we can restore normal operation
        memory_status = self.check_memory_status()
        
        if (memory_status['memory_ok'] and memory_status['gpu_ok'] and 
                self.emergency_mode and time.time() - self.previous_check > 300):  # Wait 5 minutes before trying
            
            logger.info("Resources back to normal, restoring normal operation")
            
            # Gradually restore batch size
            if self.current_batch_size < self.initial_batch_size:
                self.current_batch_size = min(self.initial_batch_size, self.current_batch_size * 2)
                logger.info(f"Increased batch size to {self.current_batch_size}")
                
            self.emergency_mode = False
            self.previous_check = time.time()  # Reset timer to prevent frequent changes
            
            return True
        
        return False
        
    def optimize_memory_usage(self) -> Optional[str]:
        """Optimize memory usage.
        
        Returns:
            Optional[str]: Description of optimization applied
        """
        # Already applied all optimizations
        if self.use_gradient_checkpointing and self.current_batch_size < self.initial_batch_size:
            return None
            
        # First reduce batch size
        if self.current_batch_size > 1:
            new_batch_size = max(1, int(self.current_batch_size * 0.8))
            if new_batch_size < self.current_batch_size:
                self.current_batch_size = new_batch_size
                return f"Reduced batch size to {self.current_batch_size}"
                
        # Then try gradient checkpointing
        if not self.use_gradient_checkpointing and self.has_gpu:
            self.use_gradient_checkpointing = True
            return "Enabled gradient checkpointing"
            
        return None
        
    def optimize_gpu_usage(self) -> Optional[str]:
        """Optimize GPU usage.
        
        Returns:
            Optional[str]: Description of optimization applied
        """
        # Enable AMP if not already enabled
        if not self.use_amp and self.has_gpu and torch.cuda.is_available():
            self.use_amp = True
            return "Enabled automatic mixed precision"
            
        # Then try gradient checkpointing
        if not self.use_gradient_checkpointing and self.has_gpu:
            self.use_gradient_checkpointing = True
            return "Enabled gradient checkpointing"
            
        # Then reduce batch size
        if self.current_batch_size > 1:
            new_batch_size = max(1, int(self.current_batch_size * 0.8))
            if new_batch_size < self.current_batch_size:
                self.current_batch_size = new_batch_size
                return f"Reduced batch size to {self.current_batch_size}"
                
        return None
        
    def optimize_cpu_usage(self) -> Optional[str]:
        """Optimize CPU usage.
        
        Returns:
            Optional[str]: Description of optimization applied
        """
        # Reduce thread count for parallel operations
        if hasattr(torch, 'set_num_threads'):
            current_threads = torch.get_num_threads()
            if current_threads > 1:
                torch.set_num_threads(max(1, current_threads - 2))
                return f"Reduced PyTorch threads from {current_threads} to {torch.get_num_threads()}"
                
        return None
        
    def optimize_io_usage(self) -> Optional[str]:
        """Optimize disk I/O usage.
        
        Returns:
            Optional[str]: Description of optimization applied
        """
        # If config available, suggest reducing checkpoint frequency
        if self.config and hasattr(self.config, 'checkpoint_freq'):
            if self.config.checkpoint_freq < 100:  # Only if checkpointing frequently
                return f"Consider increasing checkpoint frequency (currently every {self.config.checkpoint_freq} steps)"
                
        return "Consider reducing logging verbosity and checkpoint frequency"
        
    def optimize_training_parameters(self, current_fps: float) -> Dict[str, Any]:
        """Optimize training parameters based on performance metrics.
        
        Args:
            current_fps (float): Current training FPS
            
        Returns:
            Dict[str, Any]: Updated training parameters
        """
        # Update FPS history
        self.fps_history.append(current_fps)
        if len(self.fps_history) > self.history_size:
            self.fps_history.pop(0)
            
        # Check if resources are back to normal
        self.restore_normal_operation()
        
        # Return current training parameters
        return {
            'batch_size': self.current_batch_size,
            'use_amp': self.use_amp,
            'use_gradient_checkpointing': self.use_gradient_checkpointing,
            'optimizations_applied': self.optimizations_applied.copy() if self.optimizations_applied else []
        }
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for monitoring.
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Calculate average metrics
        cpu_avg = np.mean(self.cpu_history) if self.cpu_history else 0
        memory_avg = np.mean(self.memory_history) if self.memory_history else 0
        gpu_avg = np.mean(self.gpu_history) if self.gpu_history else 0
        fps_avg = np.mean(self.fps_history) if self.fps_history else 0
        
        # Return metrics
        return {
            'cpu_usage': cpu_avg,
            'memory_usage': memory_avg,
            'gpu_usage': gpu_avg,
            'fps': fps_avg,
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'optimization_count': self.optimization_count,
            'emergency_mode': self.emergency_mode,
            'batch_size': self.current_batch_size
        }
        
    def reset(self):
        """Reset performance monitoring state between episodes."""
        self.previous_check = time.time()
        self.emergency_mode = False
        # Empty histories to track fresh data
        self.cpu_history.clear()
        self.memory_history.clear()
        self.gpu_history.clear()
        self.disk_io_history.clear()
        self.fps_history.clear()
        # Restore batch size if it was reduced
        if self.current_batch_size < self.initial_batch_size:
            self.current_batch_size = self.initial_batch_size
            logger.info(f"Reset batch size to initial value: {self.initial_batch_size}")
        # Run garbage collection to clear any lingering references
        gc.collect()
        if self.has_gpu:
            torch.cuda.empty_cache()
        logger.debug("Reset performance safeguards state")

    def check_limits(self):
        """Check if resource limits are being approached and apply optimizations if needed.
        
        Returns:
            Dict with status of any actions taken
        """
        # Use check_resources to get current resource usage
        resources = self.check_resources()
        
        # Early exit if no resource data available
        if not resources:
            return {}
        
        # Check if resources are above thresholds and apply optimizations
        if (resources.get('cpu_percent', 0) > self.cpu_threshold or
            resources.get('memory_percent', 0) > self.memory_threshold or
            resources.get('gpu_percent', 0) > self.gpu_threshold):
            
            # Apply optimizations
            self.optimize_resources()
            
            return {
                'optimizations_applied': True,
                'current_batch_size': self.current_batch_size,
                'emergency_mode': self.emergency_mode
            }
        
        # Check if we can restore normal operation
        if self.emergency_mode:
            restored = self.restore_normal_operation()
            if restored:
                return {
                    'normal_operation_restored': True,
                    'current_batch_size': self.current_batch_size
                }
        
        return {'optimizations_applied': False}

    def get_throttle_time(self) -> float:
        """Get time to throttle execution based on resource usage.
        
        Returns:
            float: Sleep time in seconds (0 means no throttling needed)
        """
        # Don't throttle if not in emergency mode
        if not self.emergency_mode:
            return 0.0
        
        # Check current resource usage
        resources = self.check_resources()
        
        # If resources are critical, add throttling
        if (resources.get('memory_percent', 0) > self.critical_threshold * 100 or
            resources.get('gpu_percent', 0) > self.critical_threshold * 100):
            # Severe throttling (100ms sleep)
            return 0.1
        elif (resources.get('memory_percent', 0) > self.warning_threshold * 100 or
              resources.get('gpu_percent', 0) > self.warning_threshold * 100):
            # Moderate throttling (20ms sleep)
            return 0.02
        
        # No throttling needed
        return 0.0
        
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update internal state with hardware metrics from the hardware monitor.
        
        Args:
            metrics (Dict[str, Any]): Hardware metrics from HardwareMonitor.get_metrics()
        """
        # Update CPU usage if available
        if 'cpu_usage' in metrics:
            self.cpu_history.append(metrics['cpu_usage'])
            if len(self.cpu_history) > self.history_size:
                self.cpu_history.pop(0)
        
        # Update memory usage if available
        if 'memory_usage' in metrics:
            self.memory_history.append(metrics['memory_usage'])
            if len(self.memory_history) > self.history_size:
                self.memory_history.pop(0)
        
        # Update GPU usage if available
        if 'gpu_usage' in metrics and self.has_gpu:
            self.gpu_history.append(metrics['gpu_usage'])
            if len(self.gpu_history) > self.history_size:
                self.gpu_history.pop(0)
                
        # Update FPS history if available
        if 'fps' in metrics:
            self.fps_history.append(metrics['fps'])
            if len(self.fps_history) > self.history_size:
                self.fps_history.pop(0)
                
        # Check if we need to apply optimizations based on updated metrics
        if (len(self.cpu_history) > 0 and np.mean(self.cpu_history[-5:]) > self.cpu_threshold) or \
           (len(self.memory_history) > 0 and np.mean(self.memory_history[-5:]) > self.memory_threshold) or \
           (len(self.gpu_history) > 0 and np.mean(self.gpu_history[-5:]) > self.gpu_threshold):
            # Apply optimizations if resources are constrained
            self.optimize_resources()
        
        # Log current status after update
        logger.debug(f"Updated performance metrics - CPU: {np.mean(self.cpu_history[-5:]) if len(self.cpu_history) >= 5 else 0:.1f}%, "
                    f"Memory: {np.mean(self.memory_history[-5:]) if len(self.memory_history) >= 5 else 0:.1f}%, "
                    f"GPU: {np.mean(self.gpu_history[-5:]) if len(self.gpu_history) >= 5 else 0:.1f}%, "
                    f"Emergency mode: {self.emergency_mode}") 