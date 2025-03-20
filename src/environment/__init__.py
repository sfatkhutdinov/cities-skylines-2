"""
Environment package for Cities: Skylines 2 agent.
"""

# Backward compatibility alias
from src.environment.game_env import CitiesEnvironment

# New modular structure
from src.environment.core import Environment
from src.environment.core import GameState
from src.environment.core import ObservationManager
from src.environment.core import ActionSpace
from src.environment.core import PerformanceMonitor

# Import components from modularized sections
from src.environment.input import InputSimulator, KeyboardController, MouseController
from src.environment.menu import MenuHandler, MenuDetector, MenuNavigator, MenuRecovery, MenuTemplateManager
from src.environment.rewards import AutonomousRewardSystem, VisualChangeAnalyzer, WorldModelCNN

__all__ = [
    # Legacy classes
    'CitiesEnvironment',
    
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
    
    # Menu components
    'MenuHandler',
    'MenuDetector',
    'MenuNavigator',
    'MenuRecovery',
    'MenuTemplateManager',
    
    # Reward components
    'AutonomousRewardSystem',
    'VisualChangeAnalyzer',
    'WorldModelCNN'
] 