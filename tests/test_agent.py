"""
Tests for the agent module.
"""

import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from src.agent.core import PPOAgent, Policy, ValueFunction


class TestAgent(unittest.TestCase):
    """Tests for the Agent class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the networks
        self.mock_policy = MagicMock(spec=Policy)
        self.mock_value = MagicMock(spec=ValueFunction)
        
        # Create agent with mocked networks
        with patch("src.agent.core.ppo_agent.Policy", return_value=self.mock_policy):
            with patch("src.agent.core.ppo_agent.ValueFunction", return_value=self.mock_value):
                self.agent = PPOAgent(
                    state_dim=(3, 84, 84),
                    action_dim=10,
                    device="cpu"
                )
    
    def test_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertEqual(self.agent.state_dim, (3, 84, 84))
        self.assertEqual(self.agent.action_dim, 10)
        self.assertEqual(self.agent.device, "cpu")
    
    def test_select_action(self):
        """Test that select_action returns a valid action."""
        # Mock the policy's select_action method
        self.mock_policy.select_action.return_value = (2, torch.tensor([0.1, 0.2, 0.5, 0.1, 0.05, 0.05]))
        
        # Create a dummy state
        state = np.zeros((3, 84, 84))
        
        # Select action
        action, action_log_prob = self.agent.select_action(state)
        
        # Verify that the action is within the expected range
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 10)
        
        # Verify that the policy's select_action method was called
        self.mock_policy.select_action.assert_called_once()
    
    def test_evaluate_actions(self):
        """Test that evaluate_actions returns valid values."""
        # Mock the policy and value function methods
        self.mock_policy.evaluate_actions.return_value = (
            torch.tensor([0.1, 0.2, 0.1, 0.1]),
            torch.tensor(0.5)
        )
        self.mock_value.forward.return_value = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Create dummy states and actions
        states = torch.zeros((4, 3, 84, 84))
        actions = torch.tensor([0, 1, 2, 3])
        
        # Evaluate actions
        action_log_probs, entropy, state_values = self.agent.evaluate_actions(states, actions)
        
        # Verify that the outputs have the expected shapes
        self.assertEqual(action_log_probs.shape, (4,))
        self.assertEqual(entropy.shape, ())
        self.assertEqual(state_values.shape, (4,))
        
        # Verify that the policy and value function methods were called
        self.mock_policy.evaluate_actions.assert_called_once()
        self.mock_value.forward.assert_called_once()
    
    # Add more tests as needed


if __name__ == "__main__":
    unittest.main() 