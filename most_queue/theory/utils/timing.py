"""
Utilities for timing and performance measurement.
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def timed_method(func: F) -> F:
    """
    Decorator to measure execution time of a method and store it in a duration attribute.

    The decorated method should return a result object with a `duration` attribute.
    If the result doesn't have a duration attribute, it will be added.

    Example:
        @timed_method
        def run(self) -> QueueResults:
            # ... calculations ...
            return QueueResults(...)
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.process_time()
        result = func(self, *args, **kwargs)
        duration = time.process_time() - start

        # Set duration on result object if it has the attribute
        if hasattr(result, "duration"):
            result.duration = duration
        elif hasattr(result, "__dict__"):
            result.__dict__["duration"] = duration

        return result

    return wrapper  # type: ignore[return-value]


@contextmanager
def measure_time() -> Any:
    """
    Context manager for measuring execution time.

    Example:
        with measure_time() as timer:
            # ... code to measure ...
        elapsed = timer.elapsed
    """
    start = time.process_time()

    class Timer:
        def __init__(self, start_time: float) -> None:
            self.start_time = start_time
            self.elapsed: float = 0.0

        def finish(self) -> None:
            self.elapsed = time.process_time() - self.start_time

    timer = Timer(start)
    try:
        yield timer
    finally:
        timer.finish()


def get_elapsed_time(start_time: float) -> float:
    """
    Calculate elapsed time from a start time.

    Args:
        start_time: Start time from time.process_time()

    Returns:
        Elapsed time in seconds.
    """
    return time.process_time() - start_time
