"""
Base class for queueing networks simulation
"""

from abc import ABC, abstractmethod
from typing import Any

from most_queue.sim.base_core import BaseSimulationCore
from most_queue.sim.utils.events import EventScheduler
from most_queue.structs import NetworkResults, NetworkResultsPriority


class BaseSimNetwork(BaseSimulationCore, ABC):
    """
    Base class for queueing network simulation.
    """

    def __init__(self):
        """
        Initializes the base queue network
        """
        BaseSimulationCore.__init__(self)

        self.is_nodes_set = False
        self.is_sources_set = False
        self.results: NetworkResults | NetworkResultsPriority | None = None

        # Event scheduler for managing network events
        self.event_scheduler = EventScheduler()

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

    # Hook methods for network events
    def _before_network_arrival(self, k: int = None):
        """
        Hook called before network arrival event is processed.
        Override this to add custom logic before arrival.

        Args:
            k: Optional class number (for priority networks)
        """
        # Override in subclasses if needed

    def _after_network_arrival(self, k: int = None):
        """
        Hook called after network arrival event is processed.
        Override this to add custom logic after arrival.

        Args:
            k: Optional class number (for priority networks)
        """
        # Override in subclasses if needed

    def _before_node_serving(self, node: int, channel: int):
        """
        Hook called before node serving event is processed.
        Override this to add custom logic before serving.

        Args:
            node: Node number
            channel: Channel number
        """
        # Override in subclasses if needed

    def _after_node_serving(self, node: int, channel: int, task=None):
        """
        Hook called after node serving event is processed.
        Override this to add custom logic after serving.

        Args:
            node: Node number
            channel: Channel number
            task: Task that completed service
        """
        # Override in subclasses if needed

    def _get_custom_network_events(self):
        """
        Get custom events for this network.
        Override this method to register custom event types.

        Returns:
            dict: Dictionary mapping event_type -> event_time
        """
        return {}

    def _handle_custom_network_event(self, event_type: str):
        """
        Handle a custom network event.
        Override this method to process custom events.

        Args:
            event_type: Type of the custom event
        """
        raise NotImplementedError(f"Custom network event '{event_type}' not handled")

    def _get_node_serving_events(self):
        """
        Collect serving events from all nodes in the network.

        Returns:
            dict: Dictionary mapping event_type -> event_time
        """
        events = {}
        if not hasattr(self, "qs") or not hasattr(self, "n_num"):
            return events

        n_num = getattr(self, "n_num", 0)
        nodes = getattr(self, "nodes", [])
        qs = getattr(self, "qs", [])

        for node in range(n_num):
            if node >= len(nodes) or node >= len(qs):
                continue
            for channel in range(nodes[node]):
                if channel < len(qs[node].servers):
                    server = qs[node].servers[channel]
                    if not server.is_free and server.time_to_end_service < float("inf"):
                        event_type = f"node_serving_{node}_{channel}"
                        events[event_type] = server.time_to_end_service
        return events

    def _get_network_events(self):
        """
        Collect all available network events (arrivals, serving, custom).

        Returns:
            dict: Dictionary mapping event_type -> event_time
        """
        events = {}

        # External arrivals
        arrival_time = getattr(self, "arrival_time", None)
        if arrival_time is not None:
            if isinstance(arrival_time, list):
                # For PriorityNetwork - multiple classes
                for k, at in enumerate(arrival_time):
                    if at < float("inf"):
                        events[f"network_arrival_class_{k}"] = at
            else:
                # For NetworkSimulator - single arrival
                if arrival_time < float("inf"):
                    events["network_arrival"] = arrival_time

        # Serving events from all nodes
        events.update(self._get_node_serving_events())

        # Custom events from subclass
        custom_events = self._get_custom_network_events()
        events.update(custom_events)

        return events

    def _select_next_event(self):
        """
        Select the next event to process based on minimum time.

        Returns:
            tuple: (event_type, event_time) or (None, None) if no events
        """
        events = self._get_network_events()
        if not events:
            return None, None

        event_type = min(events.keys(), key=lambda k: events[k])
        return event_type, events[event_type]

    def _execute_event(self, event_type: str):
        """
        Execute the specified network event.
        This is a base implementation that should be overridden by subclasses
        to handle their specific event types.

        Args:
            event_type: Type of event to execute
        """
        arrival_time = getattr(self, "arrival_time", None)

        if event_type == "network_arrival":
            self._before_network_arrival()
            self.on_arrival(arrival_time)
            self._after_network_arrival()
        elif event_type.startswith("network_arrival_class_"):
            k = int(event_type.split("_")[-1])
            arrival_time_list = getattr(self, "arrival_time", [])
            if isinstance(arrival_time_list, list) and k < len(arrival_time_list):
                self._before_network_arrival(k)
                self.on_arrival(arrival_time_list[k], k)
                self._after_network_arrival(k)
        elif event_type.startswith("node_serving_"):
            parts = event_type.split("_")
            node = int(parts[2])
            channel = int(parts[3])
            self._before_node_serving(node, channel)
            result = self.on_serving(node, channel)
            self._after_node_serving(node, channel, result)
        else:
            # Custom event
            self._handle_custom_network_event(event_type)

    def run_one_step(self):
        """
        Execute one step of the simulation using event-based approach.
        Automatically handles all network events.
        """
        event_type, _event_time = self._select_next_event()

        if event_type is None:
            raise RuntimeError("No events available to process")

        self._execute_event(event_type)

    @abstractmethod
    def on_arrival(self, *args, **kwargs):
        """
        Handle arrival event. Should be implemented by subclasses.
        """
        # Override in subclasses

    @abstractmethod
    def on_serving(self, *args, **kwargs):
        """
        Handle serving event. Should be implemented by subclasses.
        """
        # Override in subclasses


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
