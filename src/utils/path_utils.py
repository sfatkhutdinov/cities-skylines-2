from pathlib import Path
import os

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT

def get_path(relative_path: str) -> Path:
    """
    Convert a relative path (from project root) to an absolute path.
    
    Args:
        relative_path: A path relative to the project root
        
    Returns:
        An absolute path starting from the project root
    """
    return PROJECT_ROOT / relative_path

def ensure_dir_exists(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: The directory path to ensure exists
        
    Returns:
        The path to the directory
    """
    path.mkdir(exist_ok=True, parents=True)
    return path

def get_logs_dir() -> Path:
    """Return the absolute path to the logs directory."""
    logs_dir = get_path("logs")
    return ensure_dir_exists(logs_dir)

def get_output_dir() -> Path:
    """Return the absolute path to the output directory."""
    output_dir = get_path("output")
    return ensure_dir_exists(output_dir)

def get_checkpoints_dir() -> Path:
    """Return the absolute path to the checkpoints directory."""
    checkpoints_dir = get_path("checkpoints")
    return ensure_dir_exists(checkpoints_dir) 