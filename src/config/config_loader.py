"""
Configuration loader for Cities: Skylines 2 agent.

This module provides a class for loading and managing different sections
of the configuration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Type, TypeVar, cast
from . import load_config, save_config
from .hardware_config import HardwareConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ConfigLoader:
    """Configuration loader for Cities: Skylines 2 agent."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration loader.
        
        Args:
            config_path (Optional[str]): Path to the configuration file
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        self._hardware_config: Optional[HardwareConfig] = None
        
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a section of the configuration.
        
        Args:
            section (str): Section name
            
        Returns:
            Dict[str, Any]: Configuration section
        """
        if section not in self.config:
            logger.warning(f"Configuration section '{section}' not found")
            return {}
            
        return self.config[section]
    
    def save(self, output_path: Optional[str] = None) -> bool:
        """Save the configuration.
        
        Args:
            output_path (Optional[str]): Path to save the configuration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        path = output_path or self.config_path
        if not path:
            logger.error("No path provided for saving configuration")
            return False
            
        return save_config(self.config, path)
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """Update a section of the configuration.
        
        Args:
            section (str): Section name
            updates (Dict[str, Any]): Updated configuration values
        """
        if section not in self.config:
            self.config[section] = {}
            
        self.config[section].update(updates)
        
        # If updating hardware config, refresh the cached instance
        if section == 'hardware':
            self._hardware_config = None
    
    def validate_config(self) -> List[str]:
        """Validate the configuration.
        
        Returns:
            List[str]: List of validation errors
        """
        errors: List[str] = []
        
        # Check required sections
        required_sections = ['hardware', 'training', 'environment', 'model']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        # Validate hardware configuration if present
        if 'hardware' in self.config:
            try:
                _ = self.get_hardware_config()
            except Exception as e:
                errors.append(f"Invalid hardware configuration: {e}")
        
        return errors
    
    def get_hardware_config(self) -> HardwareConfig:
        """Get hardware configuration.
        
        Returns:
            HardwareConfig: Hardware configuration
        """
        if self._hardware_config is None:
            hardware_section = self.get_section('hardware')
            self._hardware_config = HardwareConfig.from_dict(hardware_section)
            
        return self._hardware_config
    
    def get_config_as_args(self, section: str) -> Dict[str, Any]:
        """Get configuration section as arguments.
        
        Args:
            section (str): Section name
            
        Returns:
            Dict[str, Any]: Configuration as arguments
        """
        # Flattens nested configurations to be compatible with argparse
        section_config = self.get_section(section)
        args = {}
        
        for key, value in section_config.items():
            if not isinstance(value, dict):
                args[key] = value
        
        return args
    
    def merge_with_args(self, args: Dict[str, Any]) -> None:
        """Merge configuration with command-line arguments.
        
        Args:
            args (Dict[str, Any]): Command-line arguments
        """
        # Update each section with relevant arguments
        for section in self.config.keys():
            section_args = {k: v for k, v in args.items() 
                          if k in self.config[section] and v is not None}
            if section_args:
                self.update_section(section, section_args)
                
    @classmethod
    def load_typed_config(cls, config: Dict[str, Any], config_class: Type[T]) -> T:
        """Load configuration into a typed class.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
            config_class (Type[T]): Configuration class type
            
        Returns:
            T: Configuration instance
        """
        try:
            if hasattr(config_class, 'from_dict'):
                return config_class.from_dict(config)
            return cast(T, config_class(**config))
        except Exception as e:
            logger.error(f"Failed to load configuration into {config_class.__name__}: {e}")
            raise 