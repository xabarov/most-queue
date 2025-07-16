"""
Base class for queueing networks calculation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NetworkCalcResults:
    """
    Data class to store network results.
    """

    v: list[float]  # initial moments of sojourn time distribution
    intensities: list[float]  # intensities of arrivals into nodes
    loads: list[float]  # nodes utilizations


@dataclass
class NetworkCalcResultsPriority:
    """
    Data class to store results for network with priority discipline in nodes.
    """

    v: list[list[float]]  # initial moments of sojourn time distribution for each class
    intensities: list[list[float]]  # intensities of arrivals into nodes  for each class
    loads: list[float]  # nodes utilizations


class BaseNetwork(ABC):
    """
    Base class for queueing networks.
    """

    def __init__(self):
        """
        Initializes the base queue network
        """

        self.is_nodes_set = False
        self.is_sources_set = False
        self.results: NetworkCalcResults | NetworkCalcResultsPriority | None = None

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
    def run(self) -> NetworkCalcResults:
        """
        Run calculations for the queueing network. This method should be implemented by subclasses.
        :return: NetworkCalcResults object containing results of calculations.
        """

    def get_v(self) -> list[float]:
        """
        Returns the initial moments of sojourn time distribution
        """
        self._check_if_results_calculated()
        return self.results.v

    def get_loads(self) -> list[float]:
        """
        Returns the loads of nodes in the network.
        """
        self._check_if_results_calculated()
        return self.results.loads

    def get_intensities(self) -> list[float]:
        """
        Returns the intensities of arrivals into nodes.
        """
        self._check_if_results_calculated()
        return self.results.intensities


class BaseNetworkPriority(BaseNetwork):
    """
    Base class for queueing networks with priorities.
    """

    @abstractmethod
    def run(self) -> NetworkCalcResultsPriority:
        """
        Run calculations for the queueing network. This method should be implemented by subclasses.
        :return: NetworkCalcResults object containing results of calculations.
        """

    def get_v(self) -> list[list[float]]:
        """
        Returns the sojourn time initial moments
        """
        self._check_if_results_calculated()
        return self.results.v

    def get_loads(self) -> list[float]:
        """
        Returns the loads (utilization factors) for each node
        """
        self._check_if_results_calculated()
        return self.results.loads

    def get_intensities(self) -> list[list[float]]:
        """
        Returns arrival intensities (arrival rates) for each node
        """
        self._check_if_results_calculated()
        return self.results.intensities
