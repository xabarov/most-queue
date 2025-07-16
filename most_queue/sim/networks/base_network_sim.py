"""
Base class for queueing networks simulation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NetworkSimResults:
    """
    Data class to store network results.
    """

    v: list[float]  # initial moments of sojourn time distribution
    served: int
    arrived: int


@dataclass
class NetworkSimResultsPriority:
    """
    Data class to store results for network with priority discipline in nodes.
    """

    v: list[list[float]]  # initial moments of sojourn time distribution for each class
    served: list[int]
    arrived: list[int]


class BaseSimNetwork(ABC):
    """
    Base class for queueing network simulation.
    """

    def __init__(self):
        """
        Initializes the base queue network
        """

        self.is_nodes_set = False
        self.is_sources_set = False
        self.results: NetworkSimResults | NetworkSimResultsPriority | None = None

    @abstractmethod
    def set_sources(self):  # pylint: disable=arguments-differ
        """
        Set sources for the queueing network. This method should be implemented by subclasses.
        :param args: arguments for setting sources

        After setting the sources, self.is_sources_set should be True.
        """

    @abstractmethod
    def set_nodes(self):  # pylint: disable=arguments-differ
        """
        Set servers for the queueing network. This method should be implemented by subclasses.
        :param args: arguments for setting servers

        After setting the servers, self.is_servers_set should be True.
        """

    def _check_sources_and_nodes_is_set(self):
        """
        Check if sources and nodes are set.
        Raises:
            ValueError: If sources or nodes are not set.
        """
        if not self.is_sources_set:
            error_msg = "Sources (arrival rate and routing matrix) are not set."
            error_msg += "\nPlease use set_sources() method."
            raise ValueError(error_msg)
        if not self.is_nodes_set:
            error_msg = "Nodes (service time distribution parameters and number of channels) are not set."
            error_msg += "\nPlease use set_nodes() method."
            raise ValueError(error_msg)

    def _check_if_results_calculated(self):
        if self.results is None:
            error_msg = "Results have not been calculated yet."
            error_msg += "For calculating results, use run() method."
            raise ValueError(error_msg)

    @abstractmethod
    def run(self, job_served: int) -> NetworkSimResults:
        """
        Run simulation for the queueing network. This method should be implemented by subclasses.
        Parameters:
        job_served (int): Number of jobs to serve.

        :return: NetworkSimResults object containing results of simulation.
        """

    def get_v(self) -> list[float]:
        """
        Returns the initial moments of sojourn time distribution
        """
        self._check_if_results_calculated()
        return self.results.v


class BaseSimNetworkPriority(BaseSimNetwork):
    """
    Base class for queueing networks with priorities.
    """

    @abstractmethod
    def run(self, job_served: int) -> NetworkSimResultsPriority:
        """
        Run simulations for the queueing network.

        Parameters:
            job_served (int): Number of jobs to serve.

        This method should be implemented by subclasses.
        :return: NetworkSimResults object containing results of simulation.
        """

    def get_v(self) -> list[list[float]]:
        """
        Returns the sojourn time initial moments
        """
        self._check_if_results_calculated()
        return self.results.v
