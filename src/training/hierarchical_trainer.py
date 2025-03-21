"""
Hierarchical Training System for Cities: Skylines 2 hierarchical agent.
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from src.training.memory_trainer import MemoryTrainer
from src.agent.hierarchical_agent import HierarchicalAgent

logger = logging.getLogger(__name__)

class HierarchicalTrainer(MemoryTrainer):
    """Trainer class extended for hierarchical agents with specialized networks."""
    
    def __init__(self, agent, env, config, **kwargs):
        """Initialize the hierarchical trainer.
        
        Args:
            agent: The hierarchical agent
            env: The environment
            config: Configuration object or dictionary
            **kwargs: Additional arguments for MemoryTrainer
        """
        # Initialize base trainer
        super().__init__(agent, env, config, **kwargs)
        
        # Verify agent type
        if not isinstance(agent, HierarchicalAgent):
            logger.warning("Agent is not a HierarchicalAgent, some hierarchical features may not work")
        
        # Hierarchical-specific training parameters
        # Check if config is a dict-like object or a HardwareConfig
        if hasattr(self.config, "get"):
            # Config is a dict-like object
            self.hierarchical_config = self.config.get("hierarchical", {})
        else:
            # Config is likely a HardwareConfig object
            # Create a default hierarchical config since HardwareConfig doesn't have hierarchical settings
            self.hierarchical_config = {
                "training_schedules": {
                    "visual_network": 10,
                    "world_model": 5,
                    "error_detection": 20
                },
                "batch_sizes": {
                    "visual_network": 32,
                    "world_model": 64,
                    "error_detection": 32
                },
                "progressive_training": True,
                "progressive_phases": {
                    "visual_network": 50,
                    "world_model": 100,
                    "error_detection": 150
                }
            }
        
        # Component training schedules (how often to train each component)
        self.training_schedules = self.hierarchical_config.get("training_schedules", {
            "visual_network": 10,      # Train visual network every 10 steps
            "world_model": 5,          # Train world model every 5 steps
            "error_detection": 20      # Train error detection every 20 steps
        })
        
        # Component training parameters
        self.component_batch_sizes = self.hierarchical_config.get("batch_sizes", {
            "visual_network": 32,
            "world_model": 64,
            "error_detection": 32
        })
        
        # Progressive training (curriculum for components)
        self.progressive_training = self.hierarchical_config.get("progressive_training", True)
        self.progressive_phases = self.hierarchical_config.get("progressive_phases", {
            "visual_network": 50,      # Start training visual network after 50 episodes
            "world_model": 100,        # Start training world model after 100 episodes
            "error_detection": 150     # Start training error detection after 150 episodes
        })
        
        # Experience buffer for component training
        self.component_buffers = {
            "visual": deque(maxlen=10000),
            "world": deque(maxlen=10000),
            "error": deque(maxlen=10000)
        }
        
        # Track component training metrics
        self.component_metrics = {
            "visual_loss": 0.0,
            "world_loss": 0.0,
            "error_loss": 0.0,
            "visual_train_count": 0,
            "world_train_count": 0,
            "error_train_count": 0
        }
        
        # Current phase tracking
        self.current_phase = "base"
        
        logger.critical(f"Hierarchical trainer initialized with progressive training: {self.progressive_training}")
    
    def train_episode(self, episode_num: int, render: bool = False, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Train for a single episode with hierarchical components.
        
        Args:
            episode_num: Current episode number
            render: Whether to render the environment
            max_steps: Maximum steps for this episode (overrides default)
            
        Returns:
            Dict with episode metrics
        """
        logger.critical(f"===== STARTING EPISODE {episode_num} WITH HIERARCHICAL AGENT =====")
        
        # Determine current training phase based on episode number
        if self.progressive_training:
            self.current_phase = self._get_progressive_phase(episode_num)
            logger.info(f"Current training phase: {self.current_phase}")
        
        # Run the base training episode from memory trainer
        episode_metrics = super().train_episode(episode_num, render, max_steps)
        
        # Add hierarchical-specific metrics
        if hasattr(self.agent, 'get_hierarchical_stats'):
            hierarchical_stats = self.agent.get_hierarchical_stats()
            episode_metrics.update({f"hierarchical_{k}": v for k, v in hierarchical_stats.items()})
        
        # Add component training metrics
        episode_metrics.update(self.component_metrics)
        episode_metrics["current_phase"] = self.current_phase
        
        # Log component metrics
        logger.info(f"Component metrics: Visual loss={self.component_metrics['visual_loss']:.4f}, "
                   f"World loss={self.component_metrics['world_loss']:.4f}, "
                   f"Error loss={self.component_metrics['error_loss']:.4f}")
        
        return episode_metrics
    
    def _get_progressive_phase(self, episode_num: int) -> str:
        """Determine the current progressive training phase.
        
        Args:
            episode_num: Current episode number
            
        Returns:
            str: Current training phase
        """
        if not self.progressive_training:
            return "all"  # Train all components
            
        # Determine phase based on episode thresholds
        if episode_num < self.progressive_phases.get("visual_network", 50):
            return "base"  # Only train base policy network
        elif episode_num < self.progressive_phases.get("world_model", 100):
            return "visual"  # Train base + visual network
        elif episode_num < self.progressive_phases.get("error_detection", 150):
            return "world"  # Train base + visual + world model
        else:
            return "all"  # Train all components
    
    def process_step_experience(self, state, action, reward, next_state, done, info=None):
        """Process experience to store in memory and component buffers.
        
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
        # First, let the agent process the experience for memory
        memory_processed = super().process_step_experience(state, action, reward, next_state, done, info)
        
        # Then, store in component-specific buffers based on the current phase
        # Only store if we're actually training that component
        
        # Always add to visual buffer if we're past the visual phase
        if self.current_phase in ["visual", "world", "all"]:
            self.component_buffers["visual"].append((state, next_state))
        
        # Add to world model buffer if we're past the world model phase
        if self.current_phase in ["world", "all"]:
            if hasattr(self.agent, 'preprocess_observation'):
                # Store processed observations for world model
                processed_state = self.agent.preprocess_observation(state)
                processed_next_state = self.agent.preprocess_observation(next_state)
                self.component_buffers["world"].append((processed_state, action, processed_next_state))
            else:
                self.component_buffers["world"].append((state, action, next_state))
        
        # Add to error detection buffer if we're in the final phase
        if self.current_phase == "all":
            # For error detection, we need additional data
            additional_info = info.get('additional_info', {}) if info else {}
            predicted_next_state = additional_info.get('predicted_next_state', None)
            
            if hasattr(self.agent, 'preprocess_observation'):
                # Store processed observations for error detection
                processed_state = self.agent.preprocess_observation(state)
                processed_next_state = self.agent.preprocess_observation(next_state)
                self.component_buffers["error"].append(
                    (processed_state, action, processed_next_state, predicted_next_state)
                )
            else:
                self.component_buffers["error"].append(
                    (state, action, next_state, predicted_next_state)
                )
        
        return memory_processed
    
    def train_components(self, step_count: int):
        """Train the hierarchical components based on their schedules.
        
        Args:
            step_count: Current step count
            
        Returns:
            Dict of component training metrics
        """
        losses = {}
        
        # Check if each component should be trained this step
        if (self.current_phase in ["visual", "world", "all"] and 
            step_count % self.training_schedules.get("visual_network", 10) == 0 and
            len(self.component_buffers["visual"]) >= self.component_batch_sizes.get("visual_network", 32)):
            
            # Train visual understanding network
            visual_losses = self._train_visual_network()
            losses.update(visual_losses)
            
            # Update metrics
            self.component_metrics["visual_loss"] = visual_losses.get("visual_loss", 0.0)
            self.component_metrics["visual_train_count"] += 1
            
        if (self.current_phase in ["world", "all"] and 
            step_count % self.training_schedules.get("world_model", 5) == 0 and
            len(self.component_buffers["world"]) >= self.component_batch_sizes.get("world_model", 64)):
            
            # Train world model
            world_losses = self._train_world_model()
            losses.update(world_losses)
            
            # Update metrics
            self.component_metrics["world_loss"] = world_losses.get("total_loss", 0.0)
            self.component_metrics["world_train_count"] += 1
            
        if (self.current_phase == "all" and 
            step_count % self.training_schedules.get("error_detection", 20) == 0 and
            len(self.component_buffers["error"]) >= self.component_batch_sizes.get("error_detection", 32)):
            
            # Train error detection network
            error_losses = self._train_error_detection()
            losses.update(error_losses)
            
            # Update metrics
            self.component_metrics["error_loss"] = error_losses.get("total_loss", 0.0)
            self.component_metrics["error_train_count"] += 1
            
        return losses
    
    def _train_visual_network(self):
        """Train the visual understanding network.
        
        Returns:
            Dict of losses
        """
        if not hasattr(self.agent, 'train_visual_network') or not self.agent.use_visual_network:
            return {"visual_loss": 0.0}
            
        # Sample batch from visual buffer
        batch_size = min(self.component_batch_sizes.get("visual_network", 32), len(self.component_buffers["visual"]))
        batch_indices = np.random.choice(len(self.component_buffers["visual"]), batch_size, replace=False)
        
        observations = []
        next_observations = []
        
        for idx in batch_indices:
            obs, next_obs = self.component_buffers["visual"][idx]
            observations.append(obs)
            next_observations.append(next_obs)
            
        # Convert to tensors
        if not isinstance(observations[0], torch.Tensor):
            observations = torch.tensor(np.array(observations), dtype=torch.float32, device=self.agent.device)
            next_observations = torch.tensor(np.array(next_observations), dtype=torch.float32, device=self.agent.device)
        else:
            observations = torch.stack(observations).to(self.agent.device)
            next_observations = torch.stack(next_observations).to(self.agent.device)
            
        # Train visual network (using next_observations as augmentation)
        losses = self.agent.train_visual_network(observations)
        
        return losses
    
    def _train_world_model(self):
        """Train the world model.
        
        Returns:
            Dict of losses
        """
        if not hasattr(self.agent, 'train_world_model') or not self.agent.use_world_model:
            return {"world_loss": 0.0}
            
        # Sample batch from world buffer
        batch_size = min(self.component_batch_sizes.get("world_model", 64), len(self.component_buffers["world"]))
        batch_indices = np.random.choice(len(self.component_buffers["world"]), batch_size, replace=False)
        
        states = []
        actions = []
        next_states = []
        
        for idx in batch_indices:
            state, action, next_state = self.component_buffers["world"][idx]
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            
        # Convert to tensors
        if not isinstance(states[0], torch.Tensor):
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.agent.device)
            actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.agent.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.agent.device)
        else:
            states = torch.stack(states).to(self.agent.device)
            if isinstance(actions[0], torch.Tensor):
                actions = torch.stack(actions).to(self.agent.device)
            else:
                actions = torch.tensor(actions, dtype=torch.long, device=self.agent.device)
            next_states = torch.stack(next_states).to(self.agent.device)
            
        # Train world model
        losses = self.agent.train_world_model(states, actions, next_states)
        
        return losses
    
    def _train_error_detection(self):
        """Train the error detection network.
        
        Returns:
            Dict of losses
        """
        if not hasattr(self.agent, 'train_error_network') or not self.agent.use_error_detection:
            return {"error_loss": 0.0}
            
        # Sample batch from error buffer
        batch_size = min(self.component_batch_sizes.get("error_detection", 32), len(self.component_buffers["error"]))
        batch_indices = np.random.choice(len(self.component_buffers["error"]), batch_size, replace=False)
        
        states = []
        actions = []
        next_states = []
        predicted_next_states = []
        
        for idx in batch_indices:
            state, action, next_state, predicted = self.component_buffers["error"][idx]
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            # predicted can be None if world model wasn't used
            if predicted is not None:
                predicted_next_states.append(predicted)
            
        # Convert to tensors
        if not isinstance(states[0], torch.Tensor):
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.agent.device)
            actions = torch.tensor(np.array(actions), dtype=torch.long, device=self.agent.device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.agent.device)
            if predicted_next_states:
                predicted_next_states = torch.tensor(np.array(predicted_next_states), 
                                                   dtype=torch.float32, device=self.agent.device)
        else:
            states = torch.stack(states).to(self.agent.device)
            if isinstance(actions[0], torch.Tensor):
                actions = torch.stack(actions).to(self.agent.device)
            else:
                actions = torch.tensor(actions, dtype=torch.long, device=self.agent.device)
            next_states = torch.stack(next_states).to(self.agent.device)
            if predicted_next_states:
                predicted_next_states = torch.stack(predicted_next_states).to(self.agent.device)
                
        # If we don't have predicted states from the buffer but the agent has world model
        if not predicted_next_states and self.agent.use_world_model:
            # Generate predictions using the world model
            with torch.no_grad():
                predicted_next_states, _ = self.agent.world_model(states, actions)
            
        # Train error detection network
        losses = self.agent.train_error_network(
            states, actions, next_states, 
            predicted_next_states if predicted_next_states else None
        )
        
        return losses
    
    def update_agent(self, step_count):
        """Update the agent including all hierarchical components.
        
        Args:
            step_count: Current step count
            
        Returns:
            Dict with update metrics
        """
        # First perform standard PPO update
        update_metrics = super().update_agent(step_count)
        
        # Then train the hierarchical components
        component_metrics = self.train_components(step_count)
        
        # Combine metrics
        update_metrics.update(component_metrics)
        
        return update_metrics
    
    def save_checkpoint(self, filename, additional_data=None):
        """Save a checkpoint of the trainer and agent state.
        
        Args:
            filename: Checkpoint filename
            additional_data: Additional data to include in the checkpoint
            
        Returns:
            str: Path to the saved checkpoint
        """
        # Create checkpoint data dictionary
        checkpoint_data = {
            'agent_state': self.agent.state_dict() if hasattr(self.agent, 'state_dict') else None,
            'optimizer_state': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'component_metrics': self.component_metrics,
            'current_phase': self.current_phase,
            'episode_num': self.episode_num,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward
        }
        
        # Add hierarchical component states
        if hasattr(self.agent, 'visual_network') and self.agent.visual_network is not None:
            checkpoint_data['visual_network_state'] = self.agent.visual_network.state_dict()
            
        if hasattr(self.agent, 'world_model') and self.agent.world_model is not None:
            checkpoint_data['world_model_state'] = self.agent.world_model.state_dict()
            
        if hasattr(self.agent, 'error_network') and self.agent.error_network is not None:
            checkpoint_data['error_network_state'] = self.agent.error_network.state_dict()
        
        # Add additional data if provided
        if additional_data:
            checkpoint_data.update(additional_data)
            
        # Save checkpoint
        torch.save(checkpoint_data, filename)
        logger.critical(f"Saved hierarchical checkpoint to {filename}")
        
        return filename
    
    def load_checkpoint(self, filename):
        """Load a checkpoint of the trainer and agent state.
        
        Args:
            filename: Checkpoint filename
            
        Returns:
            bool: Whether the load was successful
        """
        try:
            # Load checkpoint data
            checkpoint_data = torch.load(filename, map_location=self.agent.device)
            
            # Load agent state
            if 'agent_state' in checkpoint_data and checkpoint_data['agent_state'] is not None:
                if hasattr(self.agent, 'load_state_dict'):
                    self.agent.load_state_dict(checkpoint_data['agent_state'])
            
            # Load optimizer state
            if 'optimizer_state' in checkpoint_data and hasattr(self, 'optimizer'):
                self.optimizer.load_state_dict(checkpoint_data['optimizer_state'])
            
            # Load hierarchical component states
            if ('visual_network_state' in checkpoint_data and 
                hasattr(self.agent, 'visual_network') and 
                self.agent.visual_network is not None):
                self.agent.visual_network.load_state_dict(checkpoint_data['visual_network_state'])
                
            if ('world_model_state' in checkpoint_data and 
                hasattr(self.agent, 'world_model') and 
                self.agent.world_model is not None):
                self.agent.world_model.load_state_dict(checkpoint_data['world_model_state'])
                
            if ('error_network_state' in checkpoint_data and 
                hasattr(self.agent, 'error_network') and 
                self.agent.error_network is not None):
                self.agent.error_network.load_state_dict(checkpoint_data['error_network_state'])
            
            # Load metrics and counters
            if 'component_metrics' in checkpoint_data:
                self.component_metrics = checkpoint_data['component_metrics']
                
            if 'current_phase' in checkpoint_data:
                self.current_phase = checkpoint_data['current_phase']
                
            if 'episode_num' in checkpoint_data:
                self.episode_num = checkpoint_data['episode_num']
                
            if 'total_steps' in checkpoint_data:
                self.total_steps = checkpoint_data['total_steps']
                
            if 'best_reward' in checkpoint_data:
                self.best_reward = checkpoint_data['best_reward']
            
            logger.critical(f"Loaded hierarchical checkpoint from {filename}")
            logger.critical(f"Resumed at episode {self.episode_num}, total steps {self.total_steps}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False 