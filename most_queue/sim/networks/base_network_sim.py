"""
Base class for queueing networks simulation
"""

from abc import ABC, abstractmethod
from typing import Any

from most_queue.structs import NetworkResults, NetworkResultsPriority


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
        self.results: NetworkResults | NetworkResultsPriority | None = None

    @abstractmethod
    def set_sources(self, *args: Any, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """
        Set sources for the queueing network. This method should be implemented by subclasses.
        :param args: arguments for setting sources
        :param kwargs: keyword arguments for setting sources

        After setting the sources, self.is_sources_set should be True.
        """

    @abstractmethod
    def set_nodes(self, *args: Any, **kwargs: Any) -> None:  # pylint: disable=arguments-differ
        """
        Set nodes for the queueing network. This method should be implemented by subclasses.
        :param args: arguments for setting nodes
        :param kwargs: keyword arguments for setting nodes

        After setting the nodes, self.is_nodes_set should be True.
        """

    def _check_sources_and_nodes_is_set(self) -> None:
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

    def _check_if_results_calculated(self) -> None:
        """
        Check if results have been calculated.
        Raises:
            ValueError: If results have not been calculated yet.
        """
        if self.results is None:
            error_msg = "Results have not been calculated yet."
            error_msg += "For calculating results, use run() method."
            raise ValueError(error_msg)

    @abstractmethod
    def run(self, job_served: int) -> NetworkResults:
        """
        Run simulation for the queueing network. This method should be implemented by subclasses.
        Parameters:
        job_served (int): Number of jobs to serve.

        :return: NetworkResults object containing results of simulation.
        """

    def get_v(self) -> list[float]:
        """
        Returns the raw moments of sojourn time distribution
        """
        self._check_if_results_calculated()
        return self.results.v


class BaseSimNetworkPriority(BaseSimNetwork):
    """
    Base class for queueing networks with priorities.
    """

    @abstractmethod
    def run(self, job_served: int) -> NetworkResultsPriority:
        """
        Run simulations for the queueing network.

        Parameters:
            job_served (int): Number of jobs to serve.

        This method should be implemented by subclasses.
        :return: NetworkResults object containing results of simulation.
        """

    def get_v(self) -> list[list[float]]:
        """
        Returns the sojourn time raw moments
        """
        self._check_if_results_calculated()
        return self.results.v
