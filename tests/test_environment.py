"""
Tests for the environment module.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from src.environment.core import Environment, ObservationManager, ActionSpace


class TestEnvironment(unittest.TestCase):
    """Tests for the Environment class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config
        self.mock_config = MagicMock()
        self.mock_config.get_device.return_value = "cpu"
        self.mock_config.get_dtype.return_value = "float32"
        
        # Create environment with mock mode
        with patch("src.environment.core.environment.ObservationManager"):
            with patch("src.environment.core.environment.ActionSpace"):
                self.env = Environment(config=self.mock_config, mock_mode=True)
    
    def test_initialization(self):
        """Test that the environment initializes correctly."""
        self.assertTrue(self.env.mock_mode)
        self.assertEqual(self.env.device, "cpu")
        self.assertEqual(self.env.dtype, "float32")
    
    def test_reset(self):
        """Test that reset returns an observation."""
        # Mock the observation manager
        self.env.observation_manager.get_observation.return_value = np.zeros((3, 84, 84))
        
        # Reset the environment
        obs = self.env.reset()
        
        # Verify that the observation has the right shape
        self.assertEqual(obs.shape, (3, 84, 84))
    
    # Add more tests as needed


if __name__ == "__main__":
    unittest.main() 