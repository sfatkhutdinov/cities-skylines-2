"""
Hardware monitoring utility to track GPU and CPU usage.
"""

import psutil
import torch
import numpy as np
import threading
import time
from typing import Dict, List, Optional
from ..config.hardware_config import HardwareConfig

class HardwareMonitor:
    def __init__(self, config: HardwareConfig):
        """Initialize hardware monitor.
        
        Args:
            config (HardwareConfig): Hardware configuration
        """
        self.config = config
        self.monitoring = False
        self.stats_history: Dict[str, List[float]] = {
            'gpu_util': [],
            'gpu_mem': [],
            'gpu_temp': [],
            'cpu_util': [],
            'ram_util': [],
            'fps': []
        }
        self.current_stats: Dict[str, float] = {}
        self.frame_times: List[float] = []
        self._lock = threading.Lock()
        
    def start_monitoring(self):
        """Start monitoring hardware in a separate thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop hardware monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            self._update_stats()
            time.sleep(1)  # Update every second
            
    def _update_stats(self):
        """Update current hardware statistics."""
        with self._lock:
            # GPU stats
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                gpu_util = torch.cuda.utilization(device)
                gpu_mem = torch.cuda.memory_allocated(device) / torch.cuda.max_memory_allocated(device)
                gpu_temp = torch.cuda.temperature(device)
                
                self.stats_history['gpu_util'].append(gpu_util)
                self.stats_history['gpu_mem'].append(gpu_mem * 100)
                self.stats_history['gpu_temp'].append(gpu_temp)
                
                self.current_stats.update({
                    'gpu_util': gpu_util,
                    'gpu_mem': gpu_mem * 100,
                    'gpu_temp': gpu_temp
                })
                
            # CPU stats
            cpu_util = psutil.cpu_percent(interval=None)
            ram_util = psutil.virtual_memory().percent
            
            self.stats_history['cpu_util'].append(cpu_util)
            self.stats_history['ram_util'].append(ram_util)
            
            self.current_stats.update({
                'cpu_util': cpu_util,
                'ram_util': ram_util
            })
            
    def log_frame_time(self, frame_time: float):
        """Log frame processing time for FPS calculation.
        
        Args:
            frame_time (float): Time taken to process one frame
        """
        with self._lock:
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)
            
            fps = 1.0 / np.mean(self.frame_times)
            self.stats_history['fps'].append(fps)
            self.current_stats['fps'] = fps
            
    def get_current_stats(self) -> Dict[str, float]:
        """Get current hardware statistics.
        
        Returns:
            Dict[str, float]: Current statistics
        """
        with self._lock:
            return self.current_stats.copy()
            
    def get_performance_warning(self) -> Optional[str]:
        """Check for performance issues.
        
        Returns:
            Optional[str]: Warning message if performance issues detected
        """
        stats = self.get_current_stats()
        warnings = []
        
        if 'gpu_util' in stats and stats['gpu_util'] < self.config.gpu_util_target * 100:
            warnings.append("GPU underutilized")
            
        if 'gpu_temp' in stats and stats['gpu_temp'] > self.config.gpu_temp_threshold:
            warnings.append("GPU temperature too high")
            
        if 'cpu_util' in stats and stats['cpu_util'] > self.config.cpu_util_target * 100:
            warnings.append("CPU utilization too high")
            
        if 'fps' in stats and stats['fps'] < self.config.target_fps:
            warnings.append(f"FPS below target ({stats['fps']:.1f} < {self.config.target_fps})")
            
        return "; ".join(warnings) if warnings else None
        
    def get_optimization_suggestion(self) -> Optional[str]:
        """Get suggestions for performance optimization.
        
        Returns:
            Optional[str]: Optimization suggestion if needed
        """
        stats = self.get_current_stats()
        
        if 'gpu_util' in stats and stats['gpu_util'] < 50:
            return "Consider increasing batch size or reducing frame skip"
            
        if 'gpu_mem' in stats and stats['gpu_mem'] > 90:
            return "Consider reducing batch size or using gradient checkpointing"
            
        if 'cpu_util' in stats and stats['cpu_util'] > 90:
            return "Consider reducing number of parallel environments"
            
        return None 