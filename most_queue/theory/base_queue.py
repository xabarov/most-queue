"""
Base class for queueing systems.
"""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Protocol

from most_queue.theory.calc_params import CalcParams
from most_queue.theory.utils.timing import get_elapsed_time


class ResultWithDuration(Protocol):
    """Protocol for result objects that have a duration attribute."""

    duration: float


class BaseQueue(ABC):
    """
    Base class for queueing systems.

    This class provides common functionality for all queueing system implementations,
    including validation, timing utilities, and abstract methods that must be
    implemented by subclasses.
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None, buffer: int | None = None) -> None:
        """
        Initialize the base queue class.

        Args:
            n: Number of channels (servers).
            calc_params: Calculation parameters. If None, default CalcParams will be used.
            buffer: Buffer size (maximum queue length). None means unlimited buffer.

        Raises:
            TypeError: If n is not an integer.
            ValueError: If n is not positive.
        """
        if not isinstance(n, int):
            raise TypeError(f"n must be an integer, got {type(n).__name__}")
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        self.n: int = n
        self.calc_params: CalcParams = calc_params if calc_params else CalcParams()
        self.buffer: int | None = buffer

        # Results storage
        self.ro: float | None = None  # Utilization factor
        self.p: list[float] | None = None  # Probability distribution
        self.w: list[float] | None = None  # Waiting time raw moments
        self.v: list[float] | None = None  # Sojourn time raw moments

        # Additional metrics
        self.mean_jobs_on_queue: float | None = None
        self.mean_jobs_in_system: float | None = None

        # State flags
        self.is_servers_set: bool = False
        self.is_sources_set: bool = False

    @abstractmethod
    def set_sources(self, *args: Any, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """
        Set sources (arrival process) for the queueing system.

        This method should be implemented by subclasses to configure the arrival process.
        After calling this method, self.is_sources_set should be set to True.

        Args:
            *args: Arguments for setting sources (subclass-specific).
            **kwargs: Keyword arguments for setting sources (subclass-specific).

        Raises:
            ValueError: If provided parameters are invalid.
        """

    @abstractmethod
    def set_servers(self, *args: Any, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """
        Set servers (service process) for the queueing system.

        This method should be implemented by subclasses to configure the service process.
        After calling this method, self.is_servers_set should be set to True.

        Args:
            *args: Arguments for setting servers (subclass-specific).
            **kwargs: Keyword arguments for setting servers (subclass-specific).

        Raises:
            ValueError: If provided parameters are invalid.
        """

    def _check_if_servers_and_sources_set(self) -> None:
        """
        Check if both servers and sources are set.

        This method should be called before performing calculations that require
        both sources and servers to be configured.

        Raises:
            ValueError: If servers or sources are not set.
        """
        if not self.is_servers_set or not self.is_sources_set:
            error_msg = "Both servers and sources must be set before calling this method. "
            error_msg += "Use set_servers() and set_sources() methods to configure them."
            raise ValueError(error_msg)

    @contextmanager
    def _validate_state(self) -> Any:
        """
        Context manager to validate that servers and sources are set.

        Usage:
            with self._validate_state():
                # perform calculations
                pass

        Raises:
            ValueError: If servers or sources are not set.
        """
        self._check_if_servers_and_sources_set()
        yield

    def _measure_time(self) -> float:
        """
        Get current process time for timing measurements.

        Returns:
            Current process time from time.process_time().

        Example:
            start = self._measure_time()
            # ... calculations ...
            duration = get_elapsed_time(start)
        """
        return time.process_time()

    def _set_duration(self, result: Any, start_time: float) -> None:
        """
        Set duration attribute on result object.

        Args:
            result: Result object (should have duration attribute or __dict__).
            start_time: Start time from _measure_time().
        """
        duration = get_elapsed_time(start_time)
        if hasattr(result, "duration"):
            result.duration = duration
        elif hasattr(result, "__dict__"):
            result.__dict__["duration"] = duration

    def get_p(self) -> list[float]:
        """
        Get the probability distribution of the number of customers in the system.

        Returns:
            List of probabilities where p[i] is the probability of having i customers
            in the system.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement get_p()")

    def get_w(self) -> list[float]:
        """
        Get the waiting time raw moments.

        Returns:
            List of raw moments of waiting time distribution.
            w[0] is the first moment (mean), w[1] is the second moment, etc.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement get_w()")

    def get_v(self) -> list[float]:
        """
        Get the sojourn time raw moments.

        Returns:
            List of raw moments of sojourn time distribution.
            v[0] is the first moment (mean), v[1] is the second moment, etc.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError("Subclasses must implement get_v()")
