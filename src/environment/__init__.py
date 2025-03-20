"""
Environment module for Cities Skylines 2 agent.

This module provides interfaces for interacting with the game environment,
including both the real game environment and a mock environment for testing.
"""

# New modular structure
from src.environment.core import Environment
from src.environment.core import GameState
from src.environment.core import ObservationManager
from src.environment.core import ActionSpace
from src.environment.core import PerformanceMonitor

# Import components from modularized sections
from src.environment.input import InputSimulator, KeyboardController, MouseController, ActionExecutor, InputTracker
from src.environment.menu import MenuHandler, MenuDetector, MenuNavigator, MenuRecovery, MenuTemplateManager
from src.environment.rewards import AutonomousRewardSystem, VisualChangeAnalyzer, WorldModelCNN
from src.environment.mock_environment import MockEnvironment

__all__ = [
    # Core environment
    'Environment',
    'GameState',
    'ObservationManager',
    'ActionSpace',
    'PerformanceMonitor',
    
    # Input components
    'InputSimulator',
    'KeyboardController',
    'MouseController',
    'ActionExecutor',
    'InputTracker',
    
    # Menu components
    'MenuHandler',
    'MenuDetector',
    'MenuNavigator',
    'MenuRecovery',
    'MenuTemplateManager',
    
    # Reward components
    'AutonomousRewardSystem',
    'VisualChangeAnalyzer',
    'WorldModelCNN',
    
    # Mock environment
    'MockEnvironment',
] 