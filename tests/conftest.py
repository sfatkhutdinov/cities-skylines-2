"""
Pytest fixtures for testing.
"""

import pytest
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch

# Make sure src is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.hardware_config import HardwareConfig
from src.environment.core import Environment
from src.agent.core import PPOAgent


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--mock-env",
        action="store_true",
        default=False,
        help="Run tests with mock environment"
    )


@pytest.fixture
def mock_config():
    """Fixture for a mock hardware configuration."""
    config = MagicMock(spec=HardwareConfig)
    config.get_device.return_value = "cpu"
    config.get_dtype.return_value = "float32"
    return config


@pytest.fixture
def mock_environment(mock_config):
    """Fixture for a mock environment."""
    with patch("src.environment.core.environment.ObservationManager") as mock_obs:
        with patch("src.environment.core.environment.ActionSpace") as mock_action:
            # Configure mocks
            mock_obs.return_value.get_observation.return_value = np.zeros((3, 84, 84))
            mock_action.return_value.sample.return_value = 0
            mock_action.return_value.n = 10
            
            # Create environment with mock mode
            env = Environment(config=mock_config, mock_mode=True)
            
            # Add needed properties
            env.observation_shape = (3, 84, 84)
            env.action_space.n = 10
            
            yield env


@pytest.fixture
def mock_agent(mock_environment, mock_config):
    """Fixture for a mock agent."""
    with patch("src.model.optimized_network.OptimizedNetwork"):
        agent = PPOAgent(
            state_dim=mock_environment.observation_shape,
            action_dim=mock_environment.action_space.n,
            device=mock_config.get_device(),
            dtype=mock_config.get_dtype()
        )
        
        yield agent 