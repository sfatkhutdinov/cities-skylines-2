"""
Network module for the PPO agent.
This file imports OptimizedNetwork from the model directory.
"""

from src.model.optimized_network import OptimizedNetwork

# Re-export the OptimizedNetwork class
__all__ = ['OptimizedNetwork'] 