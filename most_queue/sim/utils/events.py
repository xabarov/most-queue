"""
Event system for queueing simulation
"""


class SimulationEvent:
    """
    Base class for simulation events.
    Represents an event that can occur during simulation.
    """

    def __init__(self, event_time: float, event_type: str, handler_method):
        """
        Initialize a simulation event.

        Args:
            event_time: Time when the event should occur
            event_type: Type/name of the event (e.g., 'arrival', 'serving', 'warm_up_end')
            handler_method: Callable that handles this event
        """
        self.event_time = event_time
        self.event_type = event_type
        self.handler_method = handler_method

    def execute(self, *args, **kwargs):
        """
        Execute the event handler.

        Args:
            *args: Positional arguments to pass to handler
            **kwargs: Keyword arguments to pass to handler
        """
        return self.handler_method(*args, **kwargs)

    def __repr__(self):
        return f"SimulationEvent(type={self.event_type}, time={self.event_time:.3f})"


class EventScheduler:
    """
    Manages and schedules simulation events.
    Provides methods to register, query, and retrieve events.
    """

    def __init__(self):
        """Initialize an empty event scheduler."""
        self.events: dict[str, SimulationEvent] = {}

    def register_event(self, event_type: str, event_time: float, handler_method):
        """
        Register a new event or update an existing one.

        Args:
            event_type: Type/name of the event
            event_time: Time when the event should occur
            handler_method: Callable that handles this event
        """
        self.events[event_type] = SimulationEvent(event_time, event_type, handler_method)

    def unregister_event(self, event_type: str):
        """
        Remove an event from the scheduler.

        Args:
            event_type: Type of event to remove
        """
        if event_type in self.events:
            del self.events[event_type]

    def get_next_event(self):
        """
        Get the next event to occur (earliest time).

        Returns:
            tuple: (event_type, SimulationEvent) or (None, None) if no events
        """
        if not self.events:
            return None, None

        event_type = min(self.events.keys(), key=lambda k: self.events[k].event_time)
        return event_type, self.events[event_type]

    def get_next_event_time(self):
        """
        Get the time of the next event to occur.

        Returns:
            float: Time of next event, or float('inf') if no events
        """
        if not self.events:
            return float("inf")
        return min(event.event_time for event in self.events.values())

    def get_all_times(self):
        """
        Get a dictionary mapping event types to their times.

        Returns:
            dict: Dictionary with event_type -> event_time mappings
        """
        return {k: v.event_time for k, v in self.events.items()}

    def get_all_events(self):
        """
        Get all registered events.

        Returns:
            dict: Dictionary with event_type -> SimulationEvent mappings
        """
        return self.events.copy()

    def clear(self):
        """Clear all registered events."""
        self.events.clear()

    def has_events(self):
        """
        Check if there are any registered events.

        Returns:
            bool: True if there are events, False otherwise
        """
        return len(self.events) > 0
