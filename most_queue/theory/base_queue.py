"""
Base class for queueing systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from most_queue.theory.calc_params import CalcParams


@dataclass
class QueueResults:
    """
    Result of calculation for queueing system.
    """

    v: list[float] | None = None  # sojourn time initial moments
    w: list[float] | None = None  # waiting time initial moments
    p: list[float] | None = None  # probabilities of states
    pi: list[float] | None = None  # probabilities of states before arrival
    utilization: float | None = None  # utilization factor


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
    def set_sources(self):  # pylint: disable=arguments-differ
        """
        Set sources for the queueing system. This method should be implemented by subclasses.
        :param args: arguments for setting sources

        After setting the sources, self.is_sources_set should be True.
        """

    @abstractmethod
    def set_servers(self):  # pylint: disable=arguments-differ
        """
        Set servers for the queueing system. This method should be implemented by subclasses.
        :param args: arguments for setting servers

        After setting the servers, self.is_servers_set should be True.
        """

    def _check_if_servers_and_sources_set(self):
        if not self.is_servers_set or not self.is_sources_set:
            error_msg = "Both servers and sources must be set before calling this method."
            error_msg += "For setting servers and sources, use set_servers() and set_sources() methods."
            raise ValueError(error_msg)

    def get_p(self) -> list[float]:
        """
        Returns the probability distribution of the number of customers in the system.
        """

    def get_w(self) -> list[float]:
        """
        Returns the waiting time initial moments
        """

    def get_v(self) -> list[float]:
        """
        Returns the sojourn time initial moments
        """
