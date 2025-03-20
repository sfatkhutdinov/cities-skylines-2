"""
Causal learning components to enhance agent's understanding of action-outcome relationships.
These components help the agent identify and leverage cause-effect relationships in the game.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

class ActionSequenceMemory:
    """Tracks which sequences of actions lead to meaningful outcomes."""
    
    def __init__(self, sequence_length=5, memory_size=1000, device=None):
        """Initialize action sequence memory.
        
        Args:
            sequence_length (int): Length of action sequences to track
            memory_size (int): Maximum number of memory entries
            device: Device to use for tensor operations
        """
        self.sequence_length = sequence_length
        self.memory_size = memory_size
        self.device = device
        
        # Memory storage
        self.action_sequences = []  # List of action sequences
        self.context_features = []  # Corresponding context features 
        self.outcome_values = []    # Outcome values for each sequence
        
        # For efficient retrieval
        self.kdtree = None
        self.last_update_size = 0
        
    def store(self, action_sequence: List[int], context_feature: torch.Tensor, outcome: float) -> None:
        """Store action sequence with its context and outcome.
        
        Args:
            action_sequence: Sequence of action indices
            context_feature: Feature representing the context when actions were taken
            outcome: The observed outcome/reward
        """
        # Ensure action sequence has the right length
        if len(action_sequence) < self.sequence_length:
            # Pad shorter sequences
            action_sequence = action_sequence + [0] * (self.sequence_length - len(action_sequence))
        elif len(action_sequence) > self.sequence_length:
            # Truncate longer sequences
            action_sequence = action_sequence[-self.sequence_length:]
            
        # Convert action sequence to tuple for hashability
        action_tuple = tuple(action_sequence)
        
        # Store feature, action sequence and outcome
        context_cpu = context_feature.detach().cpu() if isinstance(context_feature, torch.Tensor) else context_feature
        self.action_sequences.append(action_tuple)
        self.context_features.append(context_cpu)
        self.outcome_values.append(outcome)
        
        # Keep memory within size limit
        if len(self.action_sequences) > self.memory_size:
            self.action_sequences.pop(0)
            self.context_features.pop(0)
            self.outcome_values.pop(0)
            
        # Flag that we need to rebuild kd-tree for efficient retrieval
        self.kdtree = None
        
    def query(self, context_feature: torch.Tensor, action_sequence: List[int], k: int = 5) -> Tuple[float, float]:
        """Query memory for expected outcome given context and action sequence.
        
        Args:
            context_feature: Feature representing the current context
            action_sequence: The action sequence being considered
            k: Number of nearest neighbors to consider
            
        Returns:
            Tuple[float, float]: Expected outcome and confidence score
        """
        if not self.action_sequences:
            return 0.0, 0.0
        
        # Ensure action sequence has the right format
        if len(action_sequence) < self.sequence_length:
            action_sequence = action_sequence + [0] * (self.sequence_length - len(action_sequence))
        elif len(action_sequence) > self.sequence_length:
            action_sequence = action_sequence[-self.sequence_length:]
            
        action_tuple = tuple(action_sequence)
        
        # First, look for exact action sequence matches
        exact_matches = [i for i, seq in enumerate(self.action_sequences) if seq == action_tuple]
        
        if exact_matches:
            # For exact action sequence matches, find context similarity
            context_np = context_feature.detach().cpu().numpy().flatten() if isinstance(context_feature, torch.Tensor) else context_feature.flatten()
            
            # Compute distances to contexts with matching action sequences
            match_distances = []
            for idx in exact_matches:
                match_context = self.context_features[idx]
                match_context_np = match_context.numpy().flatten() if isinstance(match_context, torch.Tensor) else match_context.flatten()
                
                # Ensure dimensions match
                min_dim = min(len(context_np), len(match_context_np))
                distance = np.linalg.norm(context_np[:min_dim] - match_context_np[:min_dim])
                match_distances.append((idx, distance))
                
            # Sort by distance
            match_distances.sort(key=lambda x: x[1])
            
            # Take top-k matches
            k_matches = match_distances[:min(k, len(match_distances))]
            
            if k_matches:
                # Calculate weights based on inverse distance
                weights = [1.0 / (dist + 1e-6) for _, dist in k_matches]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    # Weighted average of outcomes
                    weighted_sum = sum(weights[i] * self.outcome_values[idx] for i, (idx, _) in enumerate(k_matches))
                    predicted_outcome = weighted_sum / total_weight
                    
                    # Confidence based on number of matches and their similarity
                    confidence = min(1.0, len(k_matches) / k) * (1.0 - min(k_matches[0][1], 1.0))
                    
                    return predicted_outcome, confidence
        
        # If no exact matches or no valid prediction, fall back to context-based retrieval
        try:
            context_np = context_feature.detach().cpu().numpy().flatten() if isinstance(context_feature, torch.Tensor) else context_feature.flatten()
            
            # Check if we need to rebuild KD-tree
            if self.kdtree is None or len(self.context_features) != self.last_update_size:
                # Convert stored features to numpy array
                feature_array = np.vstack([
                    f.numpy().flatten() if isinstance(f, torch.Tensor) else f.flatten() 
                    for f in self.context_features
                ])
                self.kdtree = KDTree(feature_array)
                self.last_update_size = len(self.context_features)
            
            # Find k nearest neighbors
            distances, indices = self.kdtree.query(context_np.reshape(1, -1), k=min(k, len(self.context_features)))
            distances = distances[0]
            indices = indices[0]
            
            # Calculate weights based on inverse distance
            weights = 1.0 / (distances + 1e-6)
            total_weight = np.sum(weights)
            
            if total_weight == 0:
                return 0.0, 0.0
                
            # Calculate weighted average of outcomes
            weighted_sum = sum(weights[i] * self.outcome_values[indices[i]] for i in range(len(indices)))
            predicted_outcome = weighted_sum / total_weight
            
            # Lower confidence for context-only matches
            confidence = 0.5 * (1.0 - min(distances[0], 1.0))
            
            return predicted_outcome, confidence
            
        except Exception as e:
            logger.error(f"Error in action sequence query: {e}")
            return 0.0, 0.0

    def get_best_action_for_context(self, context_feature: torch.Tensor, available_actions: List[int]) -> Optional[int]:
        """Find the best action for the current context based on past experiences.
        
        Args:
            context_feature: Feature representing the current context
            available_actions: List of available action indices
            
        Returns:
            Optional[int]: Best action index or None if no data
        """
        if not self.action_sequences or not available_actions:
            return None
            
        best_outcome = float('-inf')
        best_action = None
        best_confidence = 0.0
        
        # Find sequences that start with each available action
        for action in available_actions:
            # Find sequences that start with this action
            action_sequences = [seq for seq in self.action_sequences if seq[0] == action]
            
            if not action_sequences:
                continue
                
            # For each sequence starting with this action, query expected outcome
            outcomes = []
            confidences = []
            
            for seq_idx, seq in enumerate(action_sequences):
                idx = self.action_sequences.index(seq)
                outcome, confidence = self.query(context_feature, list(seq), k=3)
                outcomes.append(outcome)
                confidences.append(confidence)
                
            if outcomes:
                # Weighted average by confidence
                total_confidence = sum(confidences)
                if total_confidence > 0:
                    avg_outcome = sum(o * c for o, c in zip(outcomes, confidences)) / total_confidence
                    avg_confidence = total_confidence / len(outcomes)
                    
                    if avg_outcome > best_outcome:
                        best_outcome = avg_outcome
                        best_action = action
                        best_confidence = avg_confidence
        
        logger.debug(f"Best action: {best_action}, predicted outcome: {best_outcome:.4f}, confidence: {best_confidence:.4f}")
        return best_action

class ContextualStateRepresentation(nn.Module):
    """Neural network that extracts context-aware representations of game state."""
    
    def __init__(self, input_dim=512, hidden_dim=256, context_dim=128, sequence_length=5):
        """Initialize contextual state representation network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            context_dim: Dimension of context embedding
            sequence_length: Number of timesteps to consider for context
        """
        super(ContextualStateRepresentation, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.sequence_length = sequence_length
        
        # GRU for temporal context processing
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Context embedding
        self.context_embedding = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.ReLU()
        )
        
        # Action embedding for incorporating past actions
        self.action_embedding = nn.Embedding(1000, 64)  # Up to 1000 action types
        
        # Projection to combine action and state features
        self.feature_projection = nn.Sequential(
            nn.Linear(context_dim + 64, context_dim),
            nn.ReLU()
        )
        
    def forward(self, features_sequence: List[torch.Tensor], actions_sequence: List[int]) -> torch.Tensor:
        """Extract contextual features from a sequence of observations and actions.
        
        Args:
            features_sequence: Sequence of feature tensors [seq_len, input_dim]
            actions_sequence: Sequence of action indices
            
        Returns:
            torch.Tensor: Contextual feature embedding
        """
        # Ensure features are properly batched
        features = torch.stack(features_sequence).unsqueeze(0)  # [1, seq_len, input_dim]
        
        # Process sequence through GRU
        output, hidden = self.gru(features)
        
        # Extract final hidden state
        context_features = self.context_embedding(hidden.squeeze(0))
        
        # Embed most recent action if available
        if actions_sequence:
            action_idx = actions_sequence[-1]
            action_tensor = torch.tensor([action_idx], device=features.device).long()
            action_embedding = self.action_embedding(action_tensor)
            
            # Combine context features with action embedding
            combined = torch.cat([context_features, action_embedding.squeeze(0)], dim=-1)
            return self.feature_projection(combined)
        else:
            # If no actions, just return context features
            return context_features
            
    def extract_context(self, features_sequence: List[torch.Tensor], actions_sequence: List[int]) -> torch.Tensor:
        """Extract context features from history (convenience wrapper).
        
        Args:
            features_sequence: List of feature vectors from past observations
            actions_sequence: List of past action indices
            
        Returns:
            torch.Tensor: Context feature embedding
        """
        with torch.no_grad():
            return self.forward(features_sequence, actions_sequence)

class MetaLearner:
    """Learns to identify which actions are useful in which contexts."""
    
    def __init__(self, context_dim=128, num_actions=1000, learning_rate=0.01):
        """Initialize meta-learner.
        
        Args:
            context_dim: Dimension of context features
            num_actions: Number of possible actions
            learning_rate: Learning rate for value updates
        """
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        
        # Create a mapping table: context -> action -> value
        # We'll use a sparse representation for efficiency
        self.context_action_values = {}
        
        # Track context prototypes for efficient lookup
        self.context_prototypes = []
        self.prototype_keys = []
        
        # Limit growth with a max size
        self.max_prototypes = 1000
        
    def update(self, context: torch.Tensor, action: int, outcome: float) -> None:
        """Update the value of taking an action in a context.
        
        Args:
            context: Context feature tensor
            action: Action index
            outcome: Observed outcome/reward
        """
        # Find the closest context prototype
        context_key = self._get_context_key(context)
        
        # Get current action values for this context
        if context_key not in self.context_action_values:
            self.context_action_values[context_key] = {}
            
        # Get current value for this action
        current_value = self.context_action_values[context_key].get(action, 0.0)
        
        # Update value with learning rate
        updated_value = current_value + self.learning_rate * (outcome - current_value)
        
        # Store updated value
        self.context_action_values[context_key][action] = updated_value
        
    def predict(self, context: torch.Tensor, action: int) -> float:
        """Predict the value of taking an action in a context.
        
        Args:
            context: Context feature tensor
            action: Action index
            
        Returns:
            float: Predicted value
        """
        # Find the closest context prototype
        context_key = self._get_context_key(context)
        
        # Return value if available, otherwise default to zero
        if context_key in self.context_action_values:
            return self.context_action_values[context_key].get(action, 0.0)
        else:
            return 0.0
            
    def get_best_action(self, context: torch.Tensor, available_actions: List[int]) -> int:
        """Get the best action for a given context.
        
        Args:
            context: Context feature tensor
            available_actions: List of available action indices
            
        Returns:
            int: Best action index
        """
        if not available_actions:
            return 0  # Default action
            
        # Get values for all available actions
        values = [self.predict(context, action) for action in available_actions]
        
        # Find best action (highest value)
        best_idx = np.argmax(values)
        return available_actions[best_idx]
        
    def _get_context_key(self, context: torch.Tensor) -> str:
        """Get a key for the context by finding the closest prototype.
        
        Args:
            context: Context feature tensor
            
        Returns:
            str: Context key
        """
        # Convert to numpy for distance calculation
        context_np = context.detach().cpu().numpy().flatten()
        
        # If we don't have any prototypes yet, create the first one
        if not self.context_prototypes:
            key = f"context_{len(self.context_prototypes)}"
            self.context_prototypes.append(context_np)
            self.prototype_keys.append(key)
            return key
            
        # Calculate distances to all prototypes
        distances = [np.linalg.norm(context_np - proto) for proto in self.context_prototypes]
        
        # Find the closest prototype
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # If close enough to an existing prototype, use it
        if min_distance < 0.1:  # Threshold for similarity
            return self.prototype_keys[min_distance_idx]
            
        # Otherwise, create a new prototype (if we have space)
        if len(self.context_prototypes) < self.max_prototypes:
            key = f"context_{len(self.context_prototypes)}"
            self.context_prototypes.append(context_np)
            self.prototype_keys.append(key)
            return key
        else:
            # If at capacity, use the closest prototype
            return self.prototype_keys[min_distance_idx]

class CausalUnderstandingModule:
    """Central module for enhancing the agent's causal understanding."""
    
    def __init__(self, config, feature_dim=512, device=None):
        """Initialize the causal understanding module.
        
        Args:
            config: Configuration object
            feature_dim: Dimension of feature embeddings
            device: Device for tensor operations
        """
        self.config = config
        self.feature_dim = feature_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # History tracking
        self.frame_feature_history = deque(maxlen=10)
        self.action_history = deque(maxlen=10)
        self.reward_history = deque(maxlen=10)
        
        # Initialize components
        self.action_sequence_memory = ActionSequenceMemory(
            sequence_length=5,
            memory_size=2000,
            device=self.device
        )
        
        self.contextual_representation = ContextualStateRepresentation(
            input_dim=feature_dim,
            hidden_dim=256,
            context_dim=128,
            sequence_length=5
        ).to(self.device)
        
        self.meta_learner = MetaLearner(
            context_dim=128,
            num_actions=1000,
            learning_rate=0.01
        )
        
    def update(self, frame_feature: torch.Tensor, action_idx: int, reward: float) -> None:
        """Update the causal understanding with new experience.
        
        Args:
            frame_feature: Feature embedding of the current frame
            action_idx: Index of the action taken
            reward: Observed reward
        """
        # Add to history
        self.frame_feature_history.append(frame_feature.detach())
        self.action_history.append(action_idx)
        self.reward_history.append(reward)
        
        # Skip further processing if we don't have enough history
        if len(self.frame_feature_history) < 3:
            return
            
        # Extract contextual features
        context_features = self.contextual_representation.extract_context(
            list(self.frame_feature_history)[:-1],  # Previous frames
            list(self.action_history)[:-1]          # Previous actions
        )
        
        # Update action sequence memory
        action_sequence = list(self.action_history)
        self.action_sequence_memory.store(
            action_sequence=action_sequence,
            context_feature=context_features,
            outcome=reward
        )
        
        # Update meta-learner
        self.meta_learner.update(
            context=context_features,
            action=action_idx,
            outcome=reward
        )
        
    def predict_action_outcome(self, frame_feature: torch.Tensor, action_idx: int) -> Tuple[float, float]:
        """Predict the outcome of taking an action in the current state.
        
        Args:
            frame_feature: Feature embedding of the current frame
            action_idx: Index of the action to take
            
        Returns:
            Tuple[float, float]: Predicted outcome and confidence
        """
        # Skip prediction if we don't have enough history
        if len(self.frame_feature_history) < 2:
            return 0.0, 0.0
            
        # Extract contextual features
        context_features = self.contextual_representation.extract_context(
            list(self.frame_feature_history),
            list(self.action_history)
        )
        
        # Create hypothetical action sequence
        action_sequence = list(self.action_history)[-4:] + [action_idx]
        
        # Query action sequence memory
        outcome, confidence = self.action_sequence_memory.query(
            context_feature=context_features,
            action_sequence=action_sequence,
            k=5
        )
        
        # Get meta-learner prediction
        meta_outcome = self.meta_learner.predict(context_features, action_idx)
        
        # Combine predictions based on confidence
        if confidence > 0.3:
            # Higher weight to memory-based prediction when confidence is high
            final_outcome = 0.7 * outcome + 0.3 * meta_outcome
        else:
            # Otherwise rely more on meta-learner
            final_outcome = 0.3 * outcome + 0.7 * meta_outcome
            
        return final_outcome, confidence
        
    def get_best_action(self, frame_feature: torch.Tensor, available_actions: List[int]) -> int:
        """Get the best action to take in the current state.
        
        Args:
            frame_feature: Feature embedding of the current frame
            available_actions: List of available action indices
            
        Returns:
            int: Index of the best action to take
        """
        # Skip prediction if we don't have enough history
        if len(self.frame_feature_history) < 2 or not available_actions:
            return available_actions[0] if available_actions else 0
            
        # Extract contextual features
        context_features = self.contextual_representation.extract_context(
            list(self.frame_feature_history),
            list(self.action_history)
        )
        
        # Query action sequence memory
        sequence_action = self.action_sequence_memory.get_best_action_for_context(
            context_feature=context_features,
            available_actions=available_actions
        )
        
        # Get meta-learner prediction
        meta_action = self.meta_learner.get_best_action(
            context=context_features,
            available_actions=available_actions
        )
        
        # Use sequence-based action if available, otherwise fall back to meta-learner
        return sequence_action if sequence_action is not None else meta_action

    def enhance_temporal_causality(self, prev_frames: List[torch.Tensor], 
                                 actions: List[int], 
                                 outcomes: List[float], 
                                 current_state: torch.Tensor) -> Dict[str, Any]:
        """Explicitly analyze cause-effect relationships across multiple timesteps.
        
        Args:
            prev_frames: List of previous frame features
            actions: List of actions taken
            outcomes: List of observed outcomes/rewards
            current_state: Current state feature
            
        Returns:
            Dict[str, Any]: Analysis of causal relationships
        """
        if len(prev_frames) < 2 or len(actions) < 2 or len(outcomes) < 2:
            return {"causal_strength": 0.0, "significant_actions": []}
            
        # 1. Identify action sequences with significant outcomes
        significant_idx = []
        for i, outcome in enumerate(outcomes):
            if abs(outcome) > 0.2:  # Threshold for "significant" outcome
                significant_idx.append(i)
                
        # 2. Extract sequences leading to significant outcomes
        significant_sequences = []
        for idx in significant_idx:
            if idx < 2:  # Need at least 2 previous steps
                continue
                
            # Get the sequence of 3 actions leading to the significant outcome
            action_seq = actions[max(0, idx-2):idx+1]
            frame_seq = prev_frames[max(0, idx-2):idx+1]
            
            significant_sequences.append({
                "actions": action_seq,
                "frames": frame_seq,
                "outcome": outcomes[idx]
            })
            
        # 3. Look for common patterns in significant sequences
        action_patterns = {}
        for seq in significant_sequences:
            pattern = tuple(seq["actions"])
            if pattern not in action_patterns:
                action_patterns[pattern] = []
            action_patterns[pattern].append(seq["outcome"])
            
        # 4. Calculate causal strength for each pattern
        causal_scores = {}
        for pattern, pattern_outcomes in action_patterns.items():
            # Higher score for consistent outcomes
            avg_outcome = sum(pattern_outcomes) / len(pattern_outcomes)
            consistency = 1.0 - min(1.0, np.std(pattern_outcomes))
            frequency = len(pattern_outcomes) / max(1, len(significant_sequences))
            
            causal_scores[pattern] = {
                "avg_outcome": avg_outcome,
                "consistency": consistency,
                "frequency": frequency,
                "causal_strength": consistency * frequency * abs(avg_outcome)
            }
            
        # 5. Identify strongest causal relationships
        sorted_patterns = sorted(
            causal_scores.items(), 
            key=lambda x: x[1]["causal_strength"], 
            reverse=True
        )
        
        # Return analysis
        return {
            "causal_strength": sorted_patterns[0][1]["causal_strength"] if sorted_patterns else 0.0,
            "significant_actions": [list(pattern) for pattern, _ in sorted_patterns[:3]],
            "detailed_analysis": sorted_patterns[:3]
        } 