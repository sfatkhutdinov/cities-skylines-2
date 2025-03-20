"""
Agent package for Cities: Skylines 2 agent.
"""

# New modular implementation
from src.agent.core.ppo_agent import PPOAgent
from src.agent.core.policy import Policy
from src.agent.core.value import ValueFunction
from src.agent.core.memory import Memory
from src.agent.core.updater import PPOUpdater

__all__ = [
    'PPOAgent',          # New modular implementation
    'Policy',            # Policy component
    'ValueFunction',     # Value function component
    'Memory',            # Memory component
    'PPOUpdater'         # Updater component
] 