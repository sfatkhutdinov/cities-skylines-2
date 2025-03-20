"""
Hardware monitoring utility to track GPU and CPU usage.
"""

import psutil
import torch
import numpy as np
import threading
import time
import os
from typing import Dict, List, Optional, Tuple
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
            'disk_io': [],
            'fps': []
        }
        self.current_stats: Dict[str, float] = {}
        self.frame_times: List[float] = []
        self._lock = threading.Lock()
        self._last_disk_io: Optional[Tuple[float, float]] = None
        self._start_time = time.time()
        
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
            self.monitor_thread.join(timeout=2)  # Add timeout to prevent blocking
            
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._update_stats()
            except Exception as e:
                print(f"Error in hardware monitoring: {e}")
            time.sleep(1)  # Update every second
            
    def _update_stats(self):
        """Update current hardware statistics."""
        with self._lock:
            # GPU stats
            if torch.cuda.is_available():
                try:
                    device = torch.cuda.current_device()
                    gpu_util = torch.cuda.utilization(device)
                    
                    # Memory usage calculation improved
                    gpu_mem_allocated = torch.cuda.memory_allocated(device)
                    gpu_mem_total = torch.cuda.get_device_properties(device).total_memory
                    gpu_mem_percent = (gpu_mem_allocated / gpu_mem_total) * 100
                    
                    # Get temperature if available
                    try:
                        gpu_temp = torch.cuda.temperature(device)
                    except (AttributeError, RuntimeError):
                        # Fall back to -1 if temperature not available
                        gpu_temp = -1
                    
                    self.stats_history['gpu_util'].append(gpu_util)
                    self.stats_history['gpu_mem'].append(gpu_mem_percent)
                    self.stats_history['gpu_temp'].append(gpu_temp)
                    
                    self.current_stats.update({
                        'gpu_util': gpu_util,
                        'gpu_mem': gpu_mem_percent,
                        'gpu_temp': gpu_temp
                    })
                except Exception as e:
                    print(f"Error getting GPU stats: {e}")
                
            # CPU stats
            cpu_util = psutil.cpu_percent(interval=None)
            ram_util = psutil.virtual_memory().percent
            
            # Disk I/O stats
            disk_io = psutil.disk_io_counters()
            if disk_io and self._last_disk_io:
                last_read, last_write = self._last_disk_io
                read_bytes = disk_io.read_bytes - last_read
                write_bytes = disk_io.write_bytes - last_write
                
                # Calculate MB/s
                io_rate = (read_bytes + write_bytes) / (1024 * 1024)
                self.stats_history['disk_io'].append(io_rate)
                self.current_stats['disk_io'] = io_rate
            
            if disk_io:
                self._last_disk_io = (disk_io.read_bytes, disk_io.write_bytes)
            
            self.stats_history['cpu_util'].append(cpu_util)
            self.stats_history['ram_util'].append(ram_util)
            
            self.current_stats.update({
                'cpu_util': cpu_util,
                'ram_util': ram_util,
                'uptime': time.time() - self._start_time
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
            
            if self.frame_times:
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
    
    def get_stats_history(self, last_n: Optional[int] = None) -> Dict[str, List[float]]:
        """Get historical statistics.
        
        Args:
            last_n (Optional[int]): Number of last entries to return, or None for all
            
        Returns:
            Dict[str, List[float]]: Historical statistics
        """
        with self._lock:
            if last_n is None:
                return {k: v.copy() for k, v in self.stats_history.items()}
            return {k: v[-last_n:] for k, v in self.stats_history.items() if v}
            
    def get_performance_warning(self) -> Optional[str]:
        """Check for performance issues.
        
        Returns:
            Optional[str]: Warning message if performance issues detected
        """
        stats = self.get_current_stats()
        warnings = []
        
        if 'gpu_util' in stats and stats['gpu_util'] < self.config.gpu_util_target * 100:
            warnings.append(f"GPU underutilized ({stats['gpu_util']:.1f}%)")
            
        if 'gpu_temp' in stats and stats['gpu_temp'] > 0 and stats['gpu_temp'] > self.config.gpu_temp_threshold:
            warnings.append(f"GPU temperature too high ({stats['gpu_temp']}°C)")
            
        if 'cpu_util' in stats and stats['cpu_util'] > self.config.cpu_util_target * 100:
            warnings.append(f"CPU utilization too high ({stats['cpu_util']:.1f}%)")
            
        if 'ram_util' in stats and stats['ram_util'] > 90:
            warnings.append(f"RAM usage critical ({stats['ram_util']:.1f}%)")
            
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
            return "Consider increasing batch size, model complexity, or reducing frame skip"
            
        if 'gpu_mem' in stats and stats['gpu_mem'] > 90:
            return "Consider reducing batch size, using gradient checkpointing, or mixed precision training"
            
        if 'cpu_util' in stats and stats['cpu_util'] > 90:
            return "Consider reducing number of parallel environments or environment complexity"
            
        if 'disk_io' in stats and stats['disk_io'] > 100:  # More than 100 MB/s
            return "High disk I/O detected; consider reducing checkpoint frequency or log verbosity"
            
        return None 