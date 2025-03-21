"""
Memory-Augmented Training System for Cities: Skylines 2 agent.
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any

from src.training.trainer import Trainer
from src.agent.memory_agent import MemoryAugmentedAgent

logger = logging.getLogger(__name__)

class MemoryTrainer(Trainer):
    """Trainer class extended for memory-augmented agents."""
    
    def __init__(self, agent, env, config_file=None, **kwargs):
        """Initialize the memory trainer.
        
        Args:
            agent: The memory-augmented agent
            env: The environment
            config_file: Path to configuration file
            **kwargs: Additional arguments for Trainer
        """
        # Initialize base trainer
        super().__init__(agent, env, config_file, **kwargs)
        
        # Verify agent type
        if not isinstance(agent, MemoryAugmentedAgent):
            logger.warning("Agent is not a MemoryAugmentedAgent, some memory features may not work")
        
        # Memory-specific training parameters
        self.memory_config = self.config.get("memory", {})
        self.memory_warmup_episodes = self.memory_config.get("warmup_episodes", 10)
        self.memory_curriculum = self.memory_config.get("use_curriculum", True)
        self.curriculum_phases = self.memory_config.get("curriculum_phases", {
            "observation": 10,    # Just observe and store memories 
            "retrieval": 30,      # Start retrieving but limit influence
            "integration": 50,    # Fully integrate memory
            "refinement": 100     # Fine-tune memory system
        })
        
        # Memory training metrics
        self.memory_metrics = {
            "memories_stored": 0,
            "memory_retrievals": 0,
            "memory_hits": 0,
            "memory_misses": 0,
            "avg_memory_influence": 0.0
        }
        
        logger.critical(f"Memory trainer initialized with warmup of {self.memory_warmup_episodes} episodes")
    
    def train_episode(self, episode_num: int, render: bool = False, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Train for a single episode with memory augmentation.
        
        Args:
            episode_num: Current episode number
            render: Whether to render the environment
            max_steps: Maximum steps for this episode (overrides default)
            
        Returns:
            Dict with episode metrics
        """
        logger.critical(f"===== STARTING EPISODE {episode_num} WITH MEMORY =====")
        
        # Determine memory usage based on curriculum
        memory_usage_phase = self._get_memory_curriculum_phase(episode_num)
        
        # Apply memory settings based on phase
        self._configure_memory_for_phase(memory_usage_phase, episode_num)
        
        # Reset agent and environment
        if hasattr(self.agent, 'reset'):
            logger.critical("Resetting agent state for new episode")
            self.agent.reset()
        
        # Run the base training episode
        episode_metrics = super().train_episode(episode_num, render, max_steps)
        
        # Add memory-specific metrics
        memory_stats = self.agent.get_memory_stats() if hasattr(self.agent, 'get_memory_stats') else {}
        episode_metrics.update({f"memory_{k}": v for k, v in memory_stats.items()})
        episode_metrics["memory_phase"] = memory_usage_phase
        
        # Log memory metrics
        if len(memory_stats) > 0:
            logger.info(f"Memory stats: {memory_stats}")
        
        return episode_metrics
    
    def _get_memory_curriculum_phase(self, episode_num: int) -> str:
        """Determine the current memory curriculum phase.
        
        Args:
            episode_num: Current episode number
            
        Returns:
            str: Current curriculum phase
        """
        if not self.memory_curriculum:
            return "integration"  # Full memory usage
            
        if episode_num < self.memory_warmup_episodes:
            return "disabled"  # No memory during warmup
            
        for phase, threshold in sorted(self.curriculum_phases.items(), key=lambda x: x[1]):
            if episode_num < threshold:
                return phase
                
        return "refinement"  # Final phase
    
    def _configure_memory_for_phase(self, phase: str, episode_num: int) -> None:
        """Configure memory usage based on curriculum phase.
        
        Args:
            phase: Current curriculum phase
            episode_num: Current episode number
        """
        if not hasattr(self.agent, 'enable_memory') or not hasattr(self.agent, 'set_memory_use_probability'):
            logger.warning("Agent doesn't support memory configuration")
            return
            
        logger.info(f"Configuring memory for phase: {phase}, episode: {episode_num}")
        
        if phase == "disabled":
            # No memory usage
            self.agent.enable_memory(False)
            
        elif phase == "observation":
            # Just collect memories but don't use them for decisions
            self.agent.enable_memory(True)
            self.agent.set_memory_use_probability(0.0)
            
        elif phase == "retrieval":
            # Start using memories with limited influence
            self.agent.enable_memory(True)
            # Gradually increase memory usage probability
            progress = (episode_num - self.curriculum_phases.get("observation", 0)) / \
                       (self.curriculum_phases.get("retrieval", 30) - self.curriculum_phases.get("observation", 0))
            memory_prob = min(0.5, max(0.1, progress * 0.5))
            self.agent.set_memory_use_probability(memory_prob)
            
        elif phase == "integration":
            # Full memory integration
            self.agent.enable_memory(True)
            self.agent.set_memory_use_probability(0.8)
            
        elif phase == "refinement":
            # Full memory usage for refinement
            self.agent.enable_memory(True)
            self.agent.set_memory_use_probability(1.0)
            
        else:
            logger.warning(f"Unknown memory phase: {phase}")
            # Default to enabled
            self.agent.enable_memory(True)
            self.agent.set_memory_use_probability(0.5)
    
    def process_step_experience(self, state, action, reward, next_state, done, info=None):
        """Process experience to store in memory if appropriate.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            info: Additional information
            
        Returns:
            bool: Whether the experience was processed
        """
        # Only process if agent supports it
        if hasattr(self.agent, 'process_experience'):
            return self.agent.process_experience(state, action, reward, next_state, done, info)
        return False 