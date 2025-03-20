"""
Performance monitoring and optimization for Cities: Skylines 2 environment.

This module handles monitoring and optimizing performance to ensure
stable operation of the agent in the Cities: Skylines 2 environment.
"""

import torch
import numpy as np
import time
import logging
import psutil
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from src.config.hardware_config import HardwareConfig

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitors and optimizes performance for the Cities: Skylines 2 environment."""
    
    def __init__(self, config: Optional[HardwareConfig] = None):
        """Initialize performance monitor.
        
        Args:
            config: Hardware configuration
        """
        self.config = config or HardwareConfig()
        
        # Performance history
        self.fps_history = deque(maxlen=100)
        self.cpu_usage_history = deque(maxlen=20)
        self.gpu_usage_history = deque(maxlen=20)
        self.ram_usage_history = deque(maxlen=20)
        
        # Timing and thresholds
        self.last_optimization_check = time.time()
        self.optimization_interval = 60  # Check every 60 seconds
        self.min_acceptable_fps = 10.0
        self.target_fps = 30.0
        
        # Resolution and frame skip settings
        self.min_frame_skip = 1
        self.max_frame_skip = 4
        self.current_frame_skip = getattr(self.config, 'frame_skip', 2)
        self.min_resolution = (64, 64)
        self.max_resolution = (256, 256)
        self.current_resolution = (84, 84)
        
        # Current optimization level (0=none, 3=max)
        self.optimization_level = 0
        
        logger.info("Performance monitor initialized")
    
    def reset(self):
        """Reset performance monitor state."""
        # Clear histories
        self.fps_history.clear()
        self.cpu_usage_history.clear()
        self.gpu_usage_history.clear()
        self.ram_usage_history.clear()
        
        # Reset timing
        self.last_optimization_check = time.time()
        
        # Reset optimization
        self.optimization_level = 0
        self.current_frame_skip = getattr(self.config, 'frame_skip', 2)
        self.current_resolution = (84, 84)
    
    def update_metrics(self, fps: float):
        """Update performance metrics.
        
        Args:
            fps: Current frames per second
        """
        # Update FPS history
        self.fps_history.append(fps)
        
        # Get CPU usage
        cpu_usage = psutil.cpu_percent()
        self.cpu_usage_history.append(cpu_usage)
        
        # Get RAM usage
        ram_usage = psutil.virtual_memory().percent
        self.ram_usage_history.append(ram_usage)
        
        # Get GPU usage if available
        try:
            if torch.cuda.is_available():
                # Not all systems support this, so use try/except
                gpu_usage = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                self.gpu_usage_history.append(gpu_usage)
        except Exception:
            # If we can't get GPU usage, estimate based on CPU usage
            self.gpu_usage_history.append(cpu_usage * 1.5)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics.
        
        Returns:
            Dict: Current performance metrics
        """
        # Calculate average metrics
        avg_fps = sum(self.fps_history) / max(1, len(self.fps_history))
        avg_cpu = sum(self.cpu_usage_history) / max(1, len(self.cpu_usage_history))
        avg_ram = sum(self.ram_usage_history) / max(1, len(self.ram_usage_history))
        avg_gpu = sum(self.gpu_usage_history) / max(1, len(self.gpu_usage_history)) if self.gpu_usage_history else 0
        
        metrics = {
            'fps': avg_fps,
            'cpu_usage': avg_cpu,
            'ram_usage': avg_ram,
            'gpu_usage': avg_gpu,
            'frame_skip': self.current_frame_skip,
            'resolution': self.current_resolution,
            'optimization_level': self.optimization_level
        }
        
        return metrics
    
    def check_and_optimize(self, environment):
        """Check performance and apply optimizations if needed.
        
        Args:
            environment: Environment instance to optimize
        """
        current_time = time.time()
        
        # Only check periodically
        if current_time - self.last_optimization_check < self.optimization_interval:
            return
            
        # Update check time
        self.last_optimization_check = current_time
        
        # Update metrics
        avg_fps = sum(self.fps_history) / max(1, len(self.fps_history)) if self.fps_history else 30.0
        
        # Log current performance
        metrics = self.get_metrics()
        logger.info(f"Performance: FPS={metrics['fps']:.1f}, CPU={metrics['cpu_usage']:.1f}%, RAM={metrics['ram_usage']:.1f}%, GPU={metrics['gpu_usage']:.1f}%")
        
        # Check if optimization needed
        if avg_fps < self.min_acceptable_fps:
            logger.warning(f"Performance issue detected: FPS={avg_fps:.1f} below threshold {self.min_acceptable_fps}")
            self._apply_optimization(environment)
        elif avg_fps > self.target_fps * 1.5 and self.optimization_level > 0:
            # We have room to reduce optimization
            logger.info(f"Good performance detected: FPS={avg_fps:.1f}, reducing optimization")
            self._reduce_optimization(environment)
    
    def _apply_optimization(self, environment):
        """Apply performance optimization.
        
        Args:
            environment: Environment instance to optimize
        """
        # Increase optimization level
        self.optimization_level = min(3, self.optimization_level + 1)
        
        # Apply optimizations based on level
        if self.optimization_level == 1:
            # Level 1: Increase frame skip
            self.current_frame_skip = min(self.max_frame_skip, self.current_frame_skip + 1)
            logger.info(f"Performance optimization level 1: Increased frame skip to {self.current_frame_skip}")
            
        elif self.optimization_level == 2:
            # Level 2: Reduce resolution
            width, height = self.current_resolution
            new_width = max(self.min_resolution[0], int(width * 0.8))
            new_height = max(self.min_resolution[1], int(height * 0.8))
            self.current_resolution = (new_width, new_height)
            logger.info(f"Performance optimization level 2: Reduced resolution to {self.current_resolution}")
            
        elif self.optimization_level == 3:
            # Level 3: Maximum optimization
            self.current_frame_skip = self.max_frame_skip
            self.current_resolution = self.min_resolution
            logger.warning("Performance optimization level 3: Applied maximum optimization")
        
        # Apply changes to environment components
        self._apply_changes_to_environment(environment)
    
    def _reduce_optimization(self, environment):
        """Reduce performance optimization.
        
        Args:
            environment: Environment instance to optimize
        """
        # Decrease optimization level
        self.optimization_level = max(0, self.optimization_level - 1)
        
        # Apply changes based on new level
        if self.optimization_level == 0:
            # Reset to defaults
            self.current_frame_skip = getattr(self.config, 'frame_skip', 2)
            self.current_resolution = (84, 84)
            logger.info("Performance optimization disabled: Restored default settings")
            
        elif self.optimization_level == 1:
            # Level 1: Adjust frame skip only
            self.current_frame_skip = 2
            self.current_resolution = (84, 84)
            logger.info(f"Performance optimization reduced to level 1: frame skip {self.current_frame_skip}")
            
        elif self.optimization_level == 2:
            # Level 2: Moderate optimization
            self.current_frame_skip = 3
            self.current_resolution = (64, 64)
            logger.info(f"Performance optimization reduced to level 2: frame skip {self.current_frame_skip}, resolution {self.current_resolution}")
        
        # Apply changes to environment components
        self._apply_changes_to_environment(environment)
    
    def _apply_changes_to_environment(self, environment):
        """Apply optimization changes to environment components.
        
        Args:
            environment: Environment instance to modify
        """
        # Update observation resolution if observation manager exists
        if hasattr(environment, 'observation_manager'):
            observation_manager = environment.observation_manager
            if hasattr(observation_manager, 'target_resolution'):
                observation_manager.target_resolution = self.current_resolution
                logger.info(f"Updated observation resolution to {self.current_resolution}")
                
        # Update frame skip if needed
        if hasattr(environment, 'current_frame_skip'):
            environment.current_frame_skip = self.current_frame_skip
            logger.info(f"Updated environment frame skip to {self.current_frame_skip}")
            
        # Update optimization level for any performance safeguards
        if hasattr(environment, 'safeguards') and hasattr(environment.safeguards, 'set_optimization_level'):
            environment.safeguards.set_optimization_level(self.optimization_level)
            logger.info(f"Updated safeguards optimization level to {self.optimization_level}") 