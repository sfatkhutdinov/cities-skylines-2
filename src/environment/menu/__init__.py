"""
Cities: Skylines 2 menu management package.

This package handles the detection, navigation, and recovery of in-game menus.
"""

from .menu_handler import MenuHandler
from .detector import MenuDetector
from .navigator import MenuNavigator
from .recovery import MenuRecovery
from .templates import MenuTemplateManager

__all__ = [
    "MenuHandler",
    "MenuDetector",
    "MenuNavigator",
    "MenuRecovery",
    "MenuTemplateManager",
] 