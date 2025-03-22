"""
Episodic Memory Module for Cities: Skylines 2 agent.
Implements a Memory-Augmented Neural Network (MANN) component for long-term memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

class EpisodicMemory:
    """Implements a memory system for storing and retrieving episodic experiences."""
    
    def __init__(self, 
                 memory_size: int = 1000,
                 embedding_size: int = 256,
                 key_size: int = 128,
                 value_size: int = 256,
                 retrieval_threshold: float = 0.5,
                 device: Optional[torch.device] = None):
        """Initialize the episodic memory.
        
        Args:
            memory_size: Maximum number of memories to store
            embedding_size: Size of state/observation embeddings
            key_size: Size of memory keys
            value_size: Size of memory values
            retrieval_threshold: Minimum similarity threshold for memory retrieval
            device: Computation device
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing episodic memory on device {self.device}")
        
        self.memory_size = memory_size
        self.embedding_size = embedding_size
        self.key_size = key_size
        self.value_size = value_size
        self.retrieval_threshold = retrieval_threshold
        
        # Memory matrices
        self.keys = torch.zeros((memory_size, key_size), device=self.device)
        self.values = torch.zeros((memory_size, value_size), device=self.device)
        self.usage = torch.zeros(memory_size, device=self.device)  # Usage frequency
        self.age = torch.zeros(memory_size, device=self.device)  # Age of memory
        self.memory_count = 0
        self.current_time = 0
        
        # Memory metrics
        self.write_count = 0
        self.read_count = 0
        self.hit_count = 0
        self.miss_count = 0
        
        # Memory encoder/decoder networks
        self.key_encoder = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, key_size)
        ).to(self.device)
        
        self.value_encoder = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, value_size)
        ).to(self.device)
        
        logger.info(f"Episodic memory initialized with size {memory_size}")
        
    def write(self, state_embedding: torch.Tensor, 
              memory_data: Dict[str, Any],
              importance: float = 1.0) -> bool:
        """Write a new memory to the memory store.
        
        Args:
            state_embedding: Embedding of the current state
            memory_data: Additional data to store with the memory
            importance: Importance score of memory (influences retention)
            
        Returns:
            bool: Success of write operation
        """
        try:
            # Get key and value embeddings
            key = self.key_encoder(state_embedding.to(self.device))
            
            # Create value embedding with additional data
            memory_tensor = self._prepare_memory_data(memory_data, state_embedding)
            value = self.value_encoder(memory_tensor)
            
            # Find slot to write memory (either empty or least important)
            if self.memory_count < self.memory_size:
                # Empty slot available
                index = self.memory_count
                self.memory_count += 1
            else:
                # Find least important memory to replace
                importance_score = self.usage * (1.0 / (self.age + 1))  # Balance usage and age
                index = torch.argmin(importance_score).item()
            
            # Write memory
            self.keys[index] = key
            self.values[index] = value
            self.usage[index] = importance
            self.age[index] = 0  # Reset age
            
            # Update counter and current time
            self.write_count += 1
            self.current_time += 1
            
            logger.debug(f"Memory written at index {index}, total memories: {self.memory_count}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing memory: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def read(self, state_embedding: torch.Tensor, top_k: int = 3) -> Tuple[List[torch.Tensor], List[float]]:
        """Read memories similar to the current state embedding.
        
        Args:
            state_embedding: Embedding of the current state
            top_k: Number of top memories to retrieve
            
        Returns:
            Tuple of (retrieved_values, similarity_scores)
        """
        try:
            # Early return if memory is empty
            if self.memory_count == 0:
                self.miss_count += 1
                return [], []
            
            # Get query key
            query = self.key_encoder(state_embedding.to(self.device))
            
            # Calculate similarity with all stored keys
            # Only consider valid memory entries
            valid_keys = self.keys[:self.memory_count]
            similarities = F.cosine_similarity(
                query.unsqueeze(0), valid_keys, dim=1
            )
            
            # Get top-k memories above threshold
            similar_indices = []
            similar_scores = []
            
            # Find memories above threshold
            for i in range(min(top_k, self.memory_count)):
                max_idx = torch.argmax(similarities).item()
                score = similarities[max_idx].item()
                
                if score >= self.retrieval_threshold:
                    similar_indices.append(max_idx)
                    similar_scores.append(score)
                    
                    # Increase usage count for retrieved memory
                    self.usage[max_idx] += 1.0
                    
                    # Set similarity of selected entry to -1 to exclude it from next selection
                    similarities[max_idx] = -1.0
                else:
                    break
            
            # Get corresponding values
            retrieved_values = [self.values[idx] for idx in similar_indices]
            
            # Update counters and ages
            self.read_count += 1
            self.current_time += 1
            self.age += 1  # Age all memories
            
            if len(retrieved_values) > 0:
                self.hit_count += 1
                logger.debug(f"Retrieved {len(retrieved_values)} memories with scores: {similar_scores}")
            else:
                self.miss_count += 1
                logger.debug("No similar memories found")
            
            return retrieved_values, similar_scores
            
        except Exception as e:
            logger.error(f"Error reading memory: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return [], []
    
    def _prepare_memory_data(self, memory_data: Dict[str, Any], 
                            state_embedding: torch.Tensor) -> torch.Tensor:
        """Prepare memory data for encoding.
        
        Args:
            memory_data: Additional data to store with the memory
            state_embedding: Embedding of the current state
            
        Returns:
            torch.Tensor: Prepared memory tensor for encoding
        """
        # For now, we'll just use the state embedding as the memory data
        # In a more advanced implementation, we would encode the additional data as well
        return state_embedding
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Dict of memory statistics
        """
        return {
            "memory_count": self.memory_count,
            "memory_size": self.memory_size,
            "write_count": self.write_count,
            "read_count": self.read_count,
            "hit_rate": self.hit_count / max(1, self.read_count),
            "miss_rate": self.miss_count / max(1, self.read_count),
            "utilization": self.memory_count / self.memory_size
        }
    
    def clear(self) -> None:
        """Clear all memories."""
        self.keys = torch.zeros((self.memory_size, self.key_size), device=self.device)
        self.values = torch.zeros((self.memory_size, self.value_size), device=self.device)
        self.usage = torch.zeros(self.memory_size, device=self.device)
        self.age = torch.zeros(self.memory_size, device=self.device)
        self.memory_count = 0
        logger.info("Episodic memory cleared")


class MANNController(nn.Module):
    """Controller for integrating episodic memory into agent's decision making."""
    
    def __init__(self, 
                 embedding_size: int = 256,
                 output_size: int = 256,
                 memory_size: int = 1000,
                 key_size: int = 128,
                 value_size: int = 256,
                 device: Optional[torch.device] = None):
        """Initialize the MANN controller.
        
        Args:
            embedding_size: Size of state/observation embeddings
            output_size: Size of the controller output
            memory_size: Maximum number of memories to store
            key_size: Size of memory keys
            value_size: Size of memory values
            device: Computation device
        """
        super(MANNController, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing MANN controller on device {self.device}")
        
        self.embedding_size = embedding_size
        self.output_size = output_size
        
        # Create episodic memory
        self.memory = EpisodicMemory(
            memory_size=memory_size,
            embedding_size=embedding_size,
            key_size=key_size,
            value_size=value_size,
            device=self.device
        )
        
        # Attention layer for memory integration
        self.attention = nn.Sequential(
            nn.Linear(embedding_size + value_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        ).to(self.device)
        
        # Memory integration layer
        self.integration = nn.Sequential(
            nn.Linear(embedding_size + value_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        ).to(self.device)
        
        # Memory projection when no memories are retrieved
        # Ensure the input dimension matches the embedding_size
        self.fallback_projection = nn.Sequential(
            nn.Linear(embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        ).to(self.device)
        
        # Create statistic counters
        self.memory_usage_count = 0
        self.fallback_count = 0
        
        self.to(self.device)
        logger.info("MANN controller initialized")
    
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MANN controller.
        
        Args:
            state_embedding: Embedding of the current state
            
        Returns:
            torch.Tensor: Controller output incorporating memory
        """
        # Ensure input is on correct device
        state_embedding = state_embedding.to(self.device)
        
        # Get input shape for debugging
        original_shape = state_embedding.shape
        logger.debug(f"MANNController input shape: {original_shape}")
        
        # Reshape if needed to match expected dimensions
        # For batched input, we need to process each embedding separately
        if len(original_shape) > 1 and original_shape[0] > 1:
            batch_size = original_shape[0]
            # Process each embedding in the batch separately and then stack results
            outputs = []
            for i in range(batch_size):
                single_embedding = state_embedding[i].unsqueeze(0)  # Add batch dim back
                outputs.append(self._process_single_embedding(single_embedding))
            return torch.stack(outputs)
        else:
            # Handle single embedding case
            return self._process_single_embedding(state_embedding)
    
    def _process_single_embedding(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """Process a single state embedding through memory system.
        
        Args:
            state_embedding: Single state embedding (with batch dim)
            
        Returns:
            torch.Tensor: Processed output
        """
        # Retrieve relevant memories
        memories, scores = self.memory.read(state_embedding, top_k=5)
        
        if len(memories) > 0:
            # We have retrieved memories
            self.memory_usage_count += 1
            
            # Prepare for attention calculation
            repeated_state = state_embedding.repeat(len(memories), 1)
            mem_tensor = torch.stack(memories)
            
            # Concatenate state with each memory
            concat_tensor = torch.cat([repeated_state, mem_tensor], dim=1)
            
            # Calculate attention weights
            attn_weights = self.attention(concat_tensor)
            attn_weights = F.softmax(attn_weights, dim=0)
            
            # Weight memories by attention
            weighted_memories = attn_weights * mem_tensor
            
            # Sum weighted memories
            combined_memory = torch.sum(weighted_memories, dim=0)
            
            # Combine with current state
            combined = torch.cat([state_embedding.squeeze(0), combined_memory], dim=0)
            
            # Final integration
            output = self.integration(combined.unsqueeze(0))
            
        else:
            # No memories retrieved, use fallback
            self.fallback_count += 1
            output = self.fallback_projection(state_embedding)
        
        return output
    
    def write_memory(self, state_embedding: torch.Tensor, 
                    memory_data: Dict[str, Any],
                    importance: float = 1.0) -> bool:
        """Write a new memory to the memory store.
        
        Args:
            state_embedding: Embedding of the current state
            memory_data: Additional data to store with the memory
            importance: Importance score of memory (influences retention)
            
        Returns:
            bool: Success of write operation
        """
        return self.memory.write(state_embedding, memory_data, importance)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get controller statistics.
        
        Returns:
            Dict of controller statistics
        """
        memory_stats = self.memory.get_stats()
        
        # Add controller-specific stats
        total_forwards = self.memory_usage_count + self.fallback_count
        controller_stats = {
            "memory_usage_rate": self.memory_usage_count / max(1, total_forwards),
            "fallback_rate": self.fallback_count / max(1, total_forwards),
            "total_forwards": total_forwards
        }
        
        # Combine both stats
        return {**memory_stats, **controller_stats}
    
    def clear_memory(self) -> None:
        """Clear all memories."""
        self.memory.clear() 