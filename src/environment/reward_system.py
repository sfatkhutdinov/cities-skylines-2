"""
Hierarchical reward system for learning game objectives through visual observation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from ..config.hardware_config import HardwareConfig
from .visual_metrics import VisualMetricsEstimator
import time
import logging
import math

class RewardSystem:
    """Reward system for Cities: Skylines 2 agent."""
    
    def __init__(self, config: HardwareConfig):
        """Initialize reward system.
        
        Args:
            config (HardwareConfig): Hardware and training configuration
        """
        self.config = config
        self.visual_estimator = VisualMetricsEstimator(config)
        
        # Reward weights for different objectives
        self.reward_weights = {
            'population_growth': 1.0,
            'building_progress': 0.8,
            'infrastructure': 0.7,
            'environment': 0.6,
            'exploration': 0.3
        }
        
        # Historical metrics for progress tracking
        self.history = {
            'estimated_population': [],
            'building_density': [],
            'infrastructure_score': [],
            'environment_score': [],
            'exploration_score': []
        }
        
        # Previous frame storage for temporal analysis
        self.previous_frame = None
        self.previous_metrics = None
        
        # Curiosity metrics
        self.state_visitation_count = {}
        self.novelty_threshold = 0.1
        
        # Track exploration to reward discovering new areas of the UI
        self.explored_regions = set()
        self.region_size = 50  # Size of grid cells for exploration tracking
        self.last_click_time = time.time()
        self.last_click_position = None
        self.unique_actions_taken = set()
        self.menu_discoveries = set()
        
        # Track recent rewards for normalization
        self.recent_rewards = []
        self.max_recent_rewards = 100
        
    def compute_population_reward(self, current_frame: torch.Tensor) -> Tuple[float, int]:
        """Calculate reward based on population growth using visual estimation."""
        estimated_population, metrics = self.visual_estimator.estimate_population(current_frame)
        
        if len(self.history['estimated_population']) > 0:
            previous_population = self.history['estimated_population'][-1]
            growth_rate = (estimated_population - previous_population) / max(previous_population, 1)
            reward = np.clip(growth_rate * self.reward_weights['population_growth'], -1.0, 1.0)
        else:
            reward = 0.0
            
        return reward, estimated_population
        
    def compute_building_progress(self, current_frame: torch.Tensor) -> float:
        """Calculate reward based on building development using visual metrics."""
        _, metrics = self.visual_estimator.estimate_population(current_frame)
        
        # Combine building density and residential areas metrics
        progress_score = (metrics['building_density'] * 0.6 + 
                         metrics['residential_areas'] * 0.4)
        
        if self.previous_metrics:
            previous_score = (self.previous_metrics['building_density'] * 0.6 + 
                            self.previous_metrics['residential_areas'] * 0.4)
            reward = (progress_score - previous_score) * self.reward_weights['building_progress']
        else:
            reward = 0.0
            
        return reward
        
    def compute_infrastructure_reward(self, current_frame: torch.Tensor) -> float:
        """Calculate reward based on infrastructure development using visual analysis."""
        _, metrics = self.visual_estimator.estimate_population(current_frame)
        
        # Use traffic density as a proxy for infrastructure quality
        infrastructure_score = metrics['traffic_density']
        
        if self.previous_metrics:
            previous_score = self.previous_metrics['traffic_density']
            reward = (infrastructure_score - previous_score) * self.reward_weights['infrastructure']
        else:
            reward = 0.0
            
        return reward
        
    def compute_environment_reward(self, current_frame: torch.Tensor) -> float:
        """Calculate reward based on environmental factors using visual analysis."""
        # This would need to be expanded with more sophisticated visual analysis
        # Currently using a simple proxy based on color distribution
        green_mask = ((current_frame[1] > 0.5) & 
                     (current_frame[0] < 0.4) & 
                     (current_frame[2] < 0.4))
        
        environment_score = float(green_mask.float().mean().item())
        
        if self.previous_frame is not None:
            previous_green_mask = ((self.previous_frame[1] > 0.5) & 
                                 (self.previous_frame[0] < 0.4) & 
                                 (self.previous_frame[2] < 0.4))
            previous_score = float(previous_green_mask.float().mean().item())
            reward = (environment_score - previous_score) * self.reward_weights['environment']
        else:
            reward = 0.0
            
        return reward
        
    def compute_curiosity_reward(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Calculate intrinsic reward based on novelty of state-action pairs."""
        state_action = self._hash_state_action(state, action)
        visit_count = self.state_visitation_count.get(state_action, 0)
        
        # Update visitation count
        self.state_visitation_count[state_action] = visit_count + 1
        
        # Compute novelty reward
        novelty_reward = self.reward_weights['exploration'] * (1.0 / (visit_count + 1))
        return np.clip(novelty_reward, 0, self.novelty_threshold)
        
    def _hash_state_action(self, state: torch.Tensor, action: torch.Tensor) -> str:
        """Create a hash for state-action pair."""
        # Downsample state for efficient hashing
        downsampled_state = torch.nn.functional.interpolate(
            state.unsqueeze(0),
            size=(32, 32),
            mode='bilinear'
        ).squeeze(0)
        
        # Combine state and action features
        state_features = downsampled_state.mean(dim=0).flatten()
        combined = torch.cat([state_features, action.flatten()])
        
        # Create hash
        return str(torch.argmax(combined).item())
        
    def compute_total_reward(self, 
                           current_frame: torch.Tensor,
                           action: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """Compute total reward combining all components using visual analysis."""
        population_reward, estimated_population = self.compute_population_reward(current_frame)
        
        rewards = {
            'population': population_reward,
            'building': self.compute_building_progress(current_frame),
            'infrastructure': self.compute_infrastructure_reward(current_frame),
            'environment': self.compute_environment_reward(current_frame),
            'curiosity': self.compute_curiosity_reward(current_frame, action)
        }
        
        # Update history
        _, metrics = self.visual_estimator.estimate_population(current_frame)
        self.history['estimated_population'].append(estimated_population)
        self.history['building_density'].append(metrics['building_density'])
        self.history['infrastructure_score'].append(metrics['traffic_density'])
        
        # Store current frame and metrics for next comparison
        self.previous_frame = current_frame.clone()
        self.previous_metrics = metrics
        
        total_reward = sum(rewards.values())
        
        # Update visual estimator based on reward
        self.visual_estimator.update_model(current_frame, total_reward)
        
        return total_reward, rewards
        
    def get_progress_metrics(self) -> Dict[str, List[float]]:
        """Get historical progress metrics."""
        return self.history.copy()
        
    def adjust_reward_weights(self, performance_metrics: Dict[str, float]):
        """Dynamically adjust reward weights based on performance."""
        # Increase weights for underperforming aspects
        for metric, value in performance_metrics.items():
            if metric in self.reward_weights and value < 0.5:
                self.reward_weights[metric] *= 1.1  # Increase weight by 10%
            
        # Normalize weights
        total = sum(self.reward_weights.values())
        self.reward_weights = {k: v/total for k, v in self.reward_weights.items()} 

    def compute_reward(self, current_state: torch.Tensor, action: int, 
                      action_info: Dict[str, Any]) -> float:
        """Compute reward for the current state and action.
        
        Args:
            current_state (torch.Tensor): Current observation
            action (int): Action index taken
            action_info (Dict): Information about the action
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Base reward for taking any action (encourages exploration)
        reward += 0.01
        
        # Reward for unique actions (encourage trying different things)
        if action not in self.unique_actions_taken:
            self.unique_actions_taken.add(action)
            reward += 0.1 * (1.0 / (len(self.unique_actions_taken) + 1))
        
        # Handle different action types
        action_type = action_info.get("type", "")
        
        # Reward UI exploration actions more to encourage interface discovery
        if action_type == "ui" or action_type == "ui_position":
            # Extract position information
            position = None
            if action_type == "ui_position":
                position = action_info.get("position", (0, 0))
            elif action_type == "ui" and action_info.get("action") in ["click_random", "right_click_random"]:
                # For random click actions, position is handled in the action handler
                # We'll use this as an opportunity to encourage exploration
                reward += 0.05
            
            # Reward for clicking in unexplored regions
            if position:
                region_x = position[0] // self.region_size
                region_y = position[1] // self.region_size
                region_key = (region_x, region_y)
                
                if region_key not in self.explored_regions:
                    self.explored_regions.add(region_key)
                    # Higher reward for discovering new regions
                    reward += 0.2 * (1.0 / (len(self.explored_regions) + 1))
                    
                # Track time between clicks to discourage rapid clicking in the same place
                current_time = time.time()
                if self.last_click_position:
                    # Reward for clicking in different locations (encourage spatial exploration)
                    distance = np.sqrt((position[0] - self.last_click_position[0])**2 + 
                                      (position[1] - self.last_click_position[1])**2)
                    # Normalize by screen size (approx 1920x1080)
                    normalized_distance = distance / np.sqrt(1920**2 + 1080**2)
                    reward += 0.05 * normalized_distance
                
                # Update last click info
                self.last_click_position = position
                self.last_click_time = current_time
                
        # Reward for camera control (slight encouragement for map exploration)
        elif action_type == "camera":
            reward += 0.02
            
        # Higher rewards for successful build actions
        elif action_type in ["build", "service", "tool"]:
            # Building should have higher rewards than just UI exploration
            reward += 0.1
        
        # Store reward for normalization
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.max_recent_rewards:
            self.recent_rewards.pop(0)
            
        # Normalize reward to avoid exploding values
        if len(self.recent_rewards) > 10:
            mean_reward = np.mean(self.recent_rewards)
            std_reward = np.std(self.recent_rewards) + 1e-5  # Avoid division by zero
            normalized_reward = (reward - mean_reward) / std_reward
            return normalized_reward
            
        return reward 

    def calculate_menu_penalty(self, in_menu: bool, consecutive_menu_steps: int = 0) -> float:
        """Calculate penalty for being in a menu state.
        
        Args:
            in_menu: Whether the agent is currently in a menu
            consecutive_menu_steps: Number of consecutive steps agent has been in menu
            
        Returns:
            float: Negative reward value as penalty for being in a menu
        """
        if not in_menu:
            return 0.0
            
        # Base penalty for being in a menu (as per user instruction)
        base_penalty = -1000.0
        
        # More controlled penalty growth to avoid catastrophic learning
        if consecutive_menu_steps > 0:
            # Use a more limited growth factor to prevent extreme values that would break learning
            # Square root growth provides diminishing returns
            growth_factor = min(1.0 + 0.5 * math.sqrt(consecutive_menu_steps), 3.0)
            penalty = base_penalty * growth_factor
            
            # Log the penalty for debugging
            logger = logging.getLogger(__name__)
            logger.info(f"Menu penalty: {penalty:.2f} (steps: {consecutive_menu_steps}, factor: {growth_factor:.2f})")
            
            return penalty
        
        return base_penalty 