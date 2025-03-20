"""
Advanced learning components for efficient, unbiased learning from raw observations.
These components enhance learning capabilities without introducing domain knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class ActionPatternDiscovery:
    """Discovers recurring patterns of actions that lead to positive outcomes."""
    
    def __init__(self, sequence_length=5, threshold=0.5, max_patterns=100):
        """Initialize action pattern discovery.
        
        Args:
            sequence_length: Length of action sequences to analyze
            threshold: Minimum success rate for pattern recognition
            max_patterns: Maximum number of patterns to store
        """
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.max_patterns = max_patterns
        
        # Storage for discovered patterns
        self.discovered_patterns = []  # List of (action_sequence, success_rate, avg_reward)
        self.pattern_usage_count = {}  # Count of how often each pattern has been used
        
        # For statistical analysis
        self.sequence_outcomes = {}  # Dictionary mapping sequences to outcome statistics
        
    def update(self, action_history: List[int], reward_history: List[float]) -> None:
        """Update pattern discovery with new action-reward data.
        
        Args:
            action_history: History of actions taken
            reward_history: Corresponding rewards received
        """
        if len(action_history) < self.sequence_length or len(reward_history) < self.sequence_length:
            return
            
        # Extract recent sequences
        for i in range(len(action_history) - self.sequence_length + 1):
            sequence = tuple(action_history[i:i+self.sequence_length])
            
            # Calculate average reward for this sequence
            seq_rewards = reward_history[i:i+self.sequence_length]
            avg_reward = sum(seq_rewards) / len(seq_rewards)
            
            # Track this sequence
            if sequence not in self.sequence_outcomes:
                self.sequence_outcomes[sequence] = {
                    'count': 0,
                    'total_reward': 0,
                    'positive_count': 0
                }
                
            stats = self.sequence_outcomes[sequence]
            stats['count'] += 1
            stats['total_reward'] += avg_reward
            if avg_reward > 0:
                stats['positive_count'] += 1
                
        # Periodically update discovered patterns
        self._update_discovered_patterns()
        
    def _update_discovered_patterns(self) -> None:
        """Update the list of discovered patterns based on accumulated statistics."""
        # Only analyze sequences with enough samples
        candidates = []
        
        for sequence, stats in self.sequence_outcomes.items():
            if stats['count'] >= 3:  # Require minimum samples
                success_rate = stats['positive_count'] / stats['count']
                avg_reward = stats['total_reward'] / stats['count']
                
                # Only consider sequences with sufficient success rate and positive reward
                if success_rate >= self.threshold and avg_reward > 0:
                    candidates.append((sequence, success_rate, avg_reward))
                    
        # Sort by average reward * success rate (overall utility)
        candidates.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        # Update discovered patterns, keeping the best ones
        self.discovered_patterns = candidates[:self.max_patterns]
        
        # Log newly discovered high-value patterns
        for seq, success_rate, avg_reward in self.discovered_patterns[:3]:
            logger.debug(f"Discovered valuable action pattern: {seq}, "
                         f"success rate: {success_rate:.2f}, avg reward: {avg_reward:.2f}")
                         
    def get_successful_patterns(self, min_success_rate=0.6) -> List[Tuple[Tuple[int, ...], float, float]]:
        """Get patterns that have been consistently successful.
        
        Args:
            min_success_rate: Minimum success rate to consider
            
        Returns:
            List of (action_sequence, success_rate, avg_reward) tuples
        """
        return [(seq, rate, reward) for seq, rate, reward in self.discovered_patterns 
                if rate >= min_success_rate]
                
    def recommend_continuation(self, current_actions: List[int]) -> Optional[int]:
        """Recommend next action based on current action prefix.
        
        Args:
            current_actions: Current sequence of actions
            
        Returns:
            Optional[int]: Recommended next action or None
        """
        if not current_actions or not self.discovered_patterns:
            return None
            
        # Use the most recent actions
        recent_actions = current_actions[-min(len(current_actions), self.sequence_length-1):]
        
        # Find patterns that start with this prefix
        matching_patterns = []
        
        for pattern, success_rate, avg_reward in self.discovered_patterns:
            pattern_prefix = pattern[:len(recent_actions)]
            if pattern_prefix == tuple(recent_actions):
                # This pattern matches our current action prefix
                matching_patterns.append((pattern, success_rate, avg_reward))
                
        if not matching_patterns:
            return None
            
        # Sort by success rate * reward
        matching_patterns.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        # Get best match
        best_pattern = matching_patterns[0][0]
        
        # Get the next action in the sequence
        next_action_idx = len(recent_actions)
        if next_action_idx < len(best_pattern):
            next_action = best_pattern[next_action_idx]
            
            # Track usage of this pattern
            if best_pattern not in self.pattern_usage_count:
                self.pattern_usage_count[best_pattern] = 0
            self.pattern_usage_count[best_pattern] += 1
            
            return next_action
            
        return None

class SpatialAttention(nn.Module):
    """Attention mechanism to focus on important regions of the input."""
    
    def __init__(self, input_channels):
        """Initialize spatial attention module.
        
        Args:
            input_channels: Number of input channels
        """
        super(SpatialAttention, self).__init__()
        
        # Convolutional layers to compute attention map
        self.conv1 = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply spatial attention to input.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Attended tensor and attention map
        """
        # Compute attention map
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)  # Scale to [0, 1]
        
        # Apply attention to input
        attended = x * attention.expand_as(x)
        
        return attended, attention

class AdaptiveLearningRate:
    """Adjusts learning rate based on prediction performance."""
    
    def __init__(self, base_rate=0.001, min_rate=0.0001, max_rate=0.01, 
                 history_size=100, adaptation_factor=0.1):
        """Initialize adaptive learning rate.
        
        Args:
            base_rate: Base learning rate
            min_rate: Minimum allowed learning rate
            max_rate: Maximum allowed learning rate
            history_size: Size of error history to maintain
            adaptation_factor: How quickly to adapt the rate
        """
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adaptation_factor = adaptation_factor
        
        self.current_rate = base_rate
        self.error_history = deque(maxlen=history_size)
        
    def update(self, prediction_error: float) -> float:
        """Update learning rate based on prediction error.
        
        Args:
            prediction_error: Current prediction error
            
        Returns:
            float: Updated learning rate
        """
        # Store error in history
        self.error_history.append(prediction_error)
        
        if len(self.error_history) < 10:
            # Not enough data to adapt yet
            return self.current_rate
            
        # Calculate recent error statistics
        recent_errors = list(self.error_history)[-10:]
        avg_error = sum(recent_errors) / len(recent_errors)
        
        # Look at error trend
        if len(self.error_history) >= 20:
            prev_errors = list(self.error_history)[-20:-10]
            prev_avg = sum(prev_errors) / len(prev_errors)
            error_trend = avg_error - prev_avg
        else:
            error_trend = 0.0
            
        # Adjust learning rate
        if error_trend > 0.01:
            # Errors increasing - increase learning rate to adapt faster
            adjustment = self.adaptation_factor
        elif error_trend < -0.01:
            # Errors decreasing - slightly decrease learning rate for stability
            adjustment = -self.adaptation_factor * 0.5
        else:
            # Errors stable - maintain current rate
            adjustment = 0.0
            
        # Apply adjustment
        self.current_rate = self.current_rate * (1.0 + adjustment)
        
        # Ensure rate stays within bounds
        self.current_rate = max(self.min_rate, min(self.max_rate, self.current_rate))
        
        return self.current_rate

class ContrastiveLearning:
    """Improves state representations through contrastive learning."""
    
    def __init__(self, embedding_dim, temperature=0.07, device=None):
        """Initialize contrastive learning module.
        
        Args:
            embedding_dim: Dimension of state embeddings
            temperature: Temperature parameter for contrastive loss
            device: Device for tensor operations
        """
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create projector network
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 4)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.projector.parameters(), lr=0.001)
        
        # Sample storage for mining negative pairs
        self.sample_memory = deque(maxlen=1000)
        
    def compute_loss(self, anchor: torch.Tensor, positive: torch.Tensor, 
                    negatives: List[torch.Tensor]) -> torch.Tensor:
        """Compute contrastive loss for improving state representations.
        
        Args:
            anchor: Anchor state embedding
            positive: Positive (similar) state embedding
            negatives: List of negative (different) state embeddings
            
        Returns:
            torch.Tensor: Contrastive loss
        """
        # Project embeddings to representation space
        anchor_proj = self.projector(anchor)
        positive_proj = self.projector(positive)
        negative_projs = [self.projector(neg) for neg in negatives]
        
        # Normalize projections
        anchor_proj = F.normalize(anchor_proj, dim=-1)
        positive_proj = F.normalize(positive_proj, dim=-1)
        negative_projs = [F.normalize(neg, dim=-1) for neg in negative_projs]
        
        # Compute similarity with positive pair
        pos_similarity = torch.sum(anchor_proj * positive_proj) / self.temperature
        
        # Compute similarities with negative pairs
        neg_similarities = torch.stack([torch.sum(anchor_proj * neg) / self.temperature 
                                      for neg in negative_projs])
        
        # Compute loss (similar to InfoNCE)
        logits = torch.cat([pos_similarity.unsqueeze(0), neg_similarities])
        labels = torch.zeros(logits.size(0), device=self.device, dtype=torch.long)  # Positive is index 0
        
        loss = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
        
        return loss
        
    def update(self, state: torch.Tensor, next_state: torch.Tensor, action: int, 
               reward: float) -> float:
        """Update representation learning with new experience.
        
        Args:
            state: Current state embedding
            next_state: Next state embedding
            action: Action taken
            reward: Reward received
            
        Returns:
            float: Contrastive loss
        """
        # Store sample in memory
        self.sample_memory.append({
            'state': state.detach(),
            'next_state': next_state.detach(),
            'action': action,
            'reward': reward
        })
        
        # Need sufficient samples to perform update
        if len(self.sample_memory) < 10:
            return 0.0
            
        # Create contrastive pairs
        # Use current state as anchor
        anchor = state
        
        # Next state is positive sample
        positive = next_state
        
        # Sample negative pairs (states that are not related)
        # Good negatives are from different episodes or distant in time
        negative_samples = []
        
        # Sample random states from memory
        sample_indices = np.random.choice(len(self.sample_memory), 
                                        min(5, len(self.sample_memory)), 
                                        replace=False)
        
        for idx in sample_indices:
            neg_sample = self.sample_memory[idx]
            # Don't use sequential states as negatives
            if neg_sample['state'] is not state and neg_sample['next_state'] is not next_state:
                negative_samples.append(neg_sample['state'])
                
        if not negative_samples:
            return 0.0
            
        # Compute contrastive loss
        loss = self.compute_loss(anchor, positive, negative_samples)
        
        # Update projector
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class MemoryConsolidation:
    """Periodically consolidates memory to focus on important experiences."""
    
    def __init__(self, update_frequency=1000, importance_threshold=0.3):
        """Initialize memory consolidation.
        
        Args:
            update_frequency: How often to perform consolidation (steps)
            importance_threshold: Threshold for keeping memories
        """
        self.update_frequency = update_frequency
        self.importance_threshold = importance_threshold
        self.steps = 0
        
        # Last consolidation timestamp
        self.last_consolidation = time.time()
        
        # Statistics
        self.memories_before = 0
        self.memories_after = 0
        self.consolidation_count = 0
        
    def should_consolidate(self) -> bool:
        """Check if memory consolidation should be performed now.
        
        Returns:
            bool: True if consolidation should occur
        """
        self.steps += 1
        
        # Time-based check (no more than once per minute)
        current_time = time.time()
        time_since_last = current_time - self.last_consolidation
        
        if time_since_last < 60:  # At least 60 seconds between consolidations
            return False
            
        # Frequency-based check
        return self.steps % self.update_frequency == 0
        
    def consolidate_memory(self, memory_dict: Dict[str, List], 
                          importance_fn) -> Dict[str, List]:
        """Consolidate memory by keeping only important experiences.
        
        Args:
            memory_dict: Dictionary containing memory components
            importance_fn: Function that calculates importance of memories
            
        Returns:
            Dict[str, List]: Consolidated memory dictionary
        """
        if not self.should_consolidate():
            return memory_dict
            
        logger.info("Performing memory consolidation...")
        consolidated = {}
        self.memories_before = 0
        
        # For each memory component
        for key, items in memory_dict.items():
            if not items:
                consolidated[key] = items
                continue
                
            self.memories_before += len(items)
            
            # Calculate importance of each memory item
            importances = importance_fn(key, items)
            
            # Keep only important memories
            keep_indices = [i for i, imp in enumerate(importances) 
                           if imp >= self.importance_threshold]
            
            consolidated[key] = [items[i] for i in keep_indices]
            
        # Update statistics
        self.memories_after = sum(len(items) for items in consolidated.values())
        self.consolidation_count += 1
        self.last_consolidation = time.time()
        
        # Report consolidation results
        logger.info(f"Memory consolidation complete: "
                   f"{self.memories_before} → {self.memories_after} items "
                   f"({(self.memories_after/max(1, self.memories_before))*100:.1f}% kept)")
                   
        return consolidated
        
    @lru_cache(maxsize=1000)
    def compute_memory_importance(self, memory_key: str, item_index: int, 
                                reward: float, recency: float) -> float:
        """Compute importance score for a memory item (with caching).
        
        Args:
            memory_key: Type of memory
            item_index: Index in memory list
            reward: Associated reward
            recency: How recent the memory is (0-1)
            
        Returns:
            float: Importance score (0-1)
        """
        # More important if:
        # 1. Associated with high absolute reward (positive or negative)
        # 2. Recent experiences
        # 3. Part of successful action sequences
        
        # Reward importance (higher weight for high absolute rewards)
        reward_importance = min(1.0, abs(reward) / 2.0)
        
        # Recency importance (higher weight for recent experiences)
        recency_importance = recency  # Already 0-1
        
        # Combine factors (equal weighting)
        return 0.6 * reward_importance + 0.4 * recency_importance 