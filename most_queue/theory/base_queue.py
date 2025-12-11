"""
Base class for queueing systems.
"""

from abc import ABC, abstractmethod
from typing import Any

from most_queue.theory.calc_params import CalcParams


class BaseQueue(ABC):
    """
    Base class for queueing systems.
    """

    def __init__(self, n: int, calc_params: CalcParams | None = None, buffer: int | None = None):
        """
        Initializes the base queue class
        :param n: number of channels
        :param calc_params: calculation parameters
        :param buffer: buffer size
        """

        self.n = n
        self.calc_params = calc_params if calc_params else CalcParams()
        self.buffer = buffer

        self.ro = None
        self.p = None
        self.w = None
        self.v = None

        self.mean_jobs_on_queue = None
        self.mean_jobs_in_system = None

        self.is_servers_set = False
        self.is_sources_set = False

    @abstractmethod
    def set_sources(self, *args: Any, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """
        Set sources for the queueing system. This method should be implemented by subclasses.
        :param args: arguments for setting sources
        :param kwargs: keyword arguments for setting sources

        After setting the sources, self.is_sources_set should be True.
        """

    @abstractmethod
    def set_servers(self, *args: Any, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """
        Set servers for the queueing system. This method should be implemented by subclasses.
        :param args: arguments for setting servers
        :param kwargs: keyword arguments for setting servers

        After setting the servers, self.is_servers_set should be True.
        """

    def _check_if_servers_and_sources_set(self) -> None:
        """
        Check if both servers and sources are set.
        Raises:
            ValueError: If servers or sources are not set.
        """
        if not self.is_servers_set or not self.is_sources_set:
            error_msg = "Both servers and sources must be set before calling this method."
            error_msg += "For setting servers and sources, use set_servers() and set_sources() methods."
            raise ValueError(error_msg)

    def get_p(self) -> list[float]:
        """
        Returns the probability distribution of the number of customers in the system.

        Returns:
            list[float]: List of probabilities where p[i] is the probability of having i customers in the system.
        """
        raise NotImplementedError("Subclasses must implement get_p()")

    def get_w(self) -> list[float]:
        """
        Returns the waiting time raw moments.

        Returns:
            list[float]: List of raw moments of waiting time distribution.
        """
        raise NotImplementedError("Subclasses must implement get_w()")

    def get_v(self) -> list[float]:
        """
        Returns the sojourn time raw moments.

        Returns:
            list[float]: List of raw moments of sojourn time distribution.
        """
        raise NotImplementedError("Subclasses must implement get_v()")
