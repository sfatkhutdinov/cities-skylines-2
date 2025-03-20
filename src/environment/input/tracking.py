"""
Input tracking for Cities: Skylines 2.

This module tracks and logs input events for debugging and analysis.
"""

import time
import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class InputEvent:
    """Represents a single input event."""
    
    event_type: str  # 'key_press', 'mouse_click', 'mouse_drag', etc.
    timestamp: float = field(default_factory=time.time)  # Unix timestamp
    details: Dict[str, Any] = field(default_factory=dict)  # Event-specific details
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation of event."""
        time_str = datetime.fromtimestamp(self.timestamp).strftime('%H:%M:%S.%f')[:-3]
        return f"[{time_str}] {self.event_type}: {self.details}"


class InputTracker:
    """Tracks all input events for the game environment."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize the input tracker.
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        self.events: deque = deque(maxlen=max_history)
        self.start_time: float = time.time()
        self.log_to_file: bool = False
        self.log_file: Optional[str] = None
        
        # Statistics for each event type
        self.event_counts: Dict[str, int] = {}
        
        logger.info(f"Input tracker initialized (max_history={max_history})")
    
    def track_event(self, event_type: str, **details) -> InputEvent:
        """Track a new input event.
        
        Args:
            event_type: Type of input event
            **details: Additional details about the event
            
        Returns:
            The created InputEvent
        """
        event = InputEvent(event_type=event_type, details=details)
        self.events.append(event)
        
        # Update statistics
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
        
        # Log to file if enabled
        if self.log_to_file and self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(f"{str(event)}\n")
            except Exception as e:
                logger.error(f"Error writing to input log file: {e}")
        
        return event
    
    def track_key_press(self, key: str, duration: float) -> InputEvent:
        """Track a key press event.
        
        Args:
            key: Key that was pressed
            duration: Duration of press in seconds
            
        Returns:
            The created InputEvent
        """
        return self.track_event('key_press', key=key, duration=duration)
    
    def track_mouse_click(
        self, 
        x: int, 
        y: int, 
        button: str = 'left', 
        double: bool = False
    ) -> InputEvent:
        """Track a mouse click event.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button used
            double: Whether it was a double click
            
        Returns:
            The created InputEvent
        """
        return self.track_event(
            'mouse_click', 
            x=x, 
            y=y, 
            button=button, 
            double=double
        )
    
    def track_mouse_drag(
        self, 
        start_x: int, 
        start_y: int, 
        end_x: int, 
        end_y: int, 
        button: str = 'left', 
        duration: float = 0.0
    ) -> InputEvent:
        """Track a mouse drag event.
        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            button: Mouse button used
            duration: Duration of drag in seconds
            
        Returns:
            The created InputEvent
        """
        return self.track_event(
            'mouse_drag', 
            start_x=start_x, 
            start_y=start_y, 
            end_x=end_x, 
            end_y=end_y, 
            button=button, 
            duration=duration
        )
    
    def track_game_action(self, action_name: str, success: bool = True, **details) -> InputEvent:
        """Track a high-level game action.
        
        Args:
            action_name: Name of the action
            success: Whether the action was successful
            **details: Additional details about the action
            
        Returns:
            The created InputEvent
        """
        return self.track_event(
            'game_action', 
            action=action_name, 
            success=success, 
            **details
        )
    
    def get_events(self, event_type: Optional[str] = None, 
                 limit: Optional[int] = None) -> List[InputEvent]:
        """Get tracked events, optionally filtered by type.
        
        Args:
            event_type: Filter by event type, or None for all events
            limit: Maximum number of events to return, or None for all
            
        Returns:
            List of InputEvent objects
        """
        if event_type:
            events = [e for e in self.events if e.event_type == event_type]
        else:
            events = list(self.events)
            
        # Apply limit if specified
        if limit is not None:
            events = events[-limit:]
            
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked events.
        
        Returns:
            Dictionary with event statistics
        """
        stats = {
            'total_events': len(self.events),
            'event_counts': self.event_counts,
            'tracking_duration': time.time() - self.start_time,
        }
        
        if stats['tracking_duration'] > 0:
            stats['events_per_second'] = stats['total_events'] / stats['tracking_duration']
        else:
            stats['events_per_second'] = 0
            
        return stats
    
    def enable_file_logging(self, log_file: str) -> bool:
        """Enable logging events to a file.
        
        Args:
            log_file: Path to log file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Test that we can write to the file
            with open(log_file, 'a') as f:
                f.write(f"# Input tracking started at {datetime.now().isoformat()}\n")
                
            self.log_file = log_file
            self.log_to_file = True
            logger.info(f"Input tracking to file enabled: {log_file}")
            return True
        except Exception as e:
            logger.error(f"Error enabling input tracking to file: {e}")
            return False
    
    def disable_file_logging(self) -> None:
        """Disable logging events to a file."""
        self.log_to_file = False
        logger.info("Input tracking to file disabled")
    
    def export_events(self, output_file: str) -> bool:
        """Export all tracked events to a JSON file.
        
        Args:
            output_file: Path to save the JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert events to dictionaries
            events_data = [e.to_dict() for e in self.events]
            
            # Write to file
            with open(output_file, 'w') as f:
                json.dump(events_data, f, indent=2)
                
            logger.info(f"Exported {len(events_data)} events to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting events to {output_file}: {e}")
            return False
    
    def clear_history(self) -> None:
        """Clear all tracked events."""
        self.events.clear()
        self.event_counts.clear()
        self.start_time = time.time()
        logger.info("Input tracking history cleared") 