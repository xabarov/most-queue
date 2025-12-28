"""
Simulation of a network with negative jobs at each node
"""

import math
import time

import numpy as np
from colorama import Fore, init
from tqdm import tqdm

from most_queue.random.distributions import ExpDistribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.sim.networks.base_network_sim import BaseSimNetwork
from most_queue.sim.utils.tasks import Task
from most_queue.structs import NetworkResults

init()


class NegativeNetwork(BaseSimNetwork):
    """
    Simulation of network with negative jobs at each node.
    Each node uses QsSimNegatives simulator.
    """

    def __init__(self, negative_arrival_type: str = "global"):
        """
        Initialize the negative network simulator.

        Args:
            negative_arrival_type: Type of negative arrivals.
                "global" - negative arrivals affect all nodes simultaneously.
                    Use negative_arrival_rate in set_sources().
                "per_node" - each node has its own negative arrival rate.
                    Use negative_arrival_rates list in set_sources().
                    Note: set_nodes() must be called before set_sources() for per_node type.
        """
        super().__init__()

        # Positive arrivals
        self.positive_arrival_rate = None
        self.positive_arrival_time = None
        self.positive_source = None
        self.R = None  # routing matrix for positive jobs

        # Negative arrivals (can be global or per-node)
        self.negative_arrival_rate = None  # For global type
        self.negative_arrival_time = None  # For global type
        self.negative_source = None  # For global type
        self.negative_arrival_type = negative_arrival_type

        # Per-node negative arrivals (for per_node type)
        self.negative_arrival_rates = None  # List of rates for each node
        self.negative_arrival_times = None  # List of arrival times for each node
        self.negative_sources = None  # List of sources for each node

        self.n_num = None  # number of nodes
        self.nodes = None  # n[i] - number of channels in node i

        self.serv_params = None
        self.negative_types = None  # type of negative service for each node
        self.buffers = None  # buffer size for each node (None for infinite)

        self.qs = []  # list of QsSimNegatives nodes

        self.v_network = [0.0] * 3
        self.w_network = [0.0] * 3

        self.served = 0
        self.arrived = 0
        self.in_sys = 0

    def set_sources(
        self,
        positive_arrival_rate: float,
        R: np.matrix,
        negative_arrival_rate: float = None,
        negative_arrival_rates: list[float] = None,
    ):  # pylint: disable=arguments-differ
        """
        Set the arrival rates and routing matrix.

        Parameters:
            positive_arrival_rate: arrival rate of positive customers.
            R: routing matrix, dim (m + 1 x m + 1), where m is number of nodes.
                For example:
                R[0, 0] is transition from source to first node.
                R[0, m] is transition from source to out of system.
            negative_arrival_rate: arrival rate of negative customers for global type (optional).
                If None, negative arrivals are disabled for global type.
                Used only if negative_arrival_type is "global".
            negative_arrival_rates: list of arrival rates for negative customers per node (optional).
                List should have length equal to number of nodes.
                Used only if negative_arrival_type is "per_node".
        """
        self.positive_arrival_rate = positive_arrival_rate
        self.positive_source = ExpDistribution(positive_arrival_rate)
        self.positive_arrival_time = self.positive_source.generate()
        self.R = R

        if self.negative_arrival_type == "global":
            if negative_arrival_rate is not None:
                self.negative_arrival_rate = negative_arrival_rate
                # Handle zero rate specially to avoid division by zero
                if negative_arrival_rate == 0.0 or abs(negative_arrival_rate) < 1e-10:
                    self.negative_arrival_time = float("inf")
                    self.negative_source = None
                else:
                    self.negative_source = ExpDistribution(negative_arrival_rate)
                    self.negative_arrival_time = self.negative_source.generate()
            else:
                self.negative_arrival_time = float("inf")
                self.negative_source = None
        elif self.negative_arrival_type == "per_node":
            if negative_arrival_rates is not None:
                # Check if nodes are set first (needed to know number of nodes)
                if not hasattr(self, "n_num") or self.n_num is None:
                    raise ValueError(
                        "set_nodes() must be called before set_sources() when using per_node type. "
                        "This is required to know the number of nodes."
                    )

                if len(negative_arrival_rates) != self.n_num:
                    raise ValueError(
                        f"negative_arrival_rates length ({len(negative_arrival_rates)}) "
                        f"must match number of nodes ({self.n_num})"
                    )

                self.negative_arrival_rates = negative_arrival_rates
                self.negative_arrival_times = []
                self.negative_sources = []
                for rate in negative_arrival_rates:
                    if rate <= 0:
                        source = None
                        arrival_time = float("inf")
                    else:
                        source = ExpDistribution(rate)
                        arrival_time = source.generate()
                    self.negative_sources.append(source)
                    self.negative_arrival_times.append(arrival_time)
            else:
                # Disable per-node negative arrivals
                if hasattr(self, "n_num") and self.n_num is not None:
                    self.negative_arrival_times = [float("inf")] * self.n_num
                    self.negative_arrival_rates = [0.0] * self.n_num
                    self.negative_sources = [None] * self.n_num
                else:
                    # Will be initialized when set_nodes is called
                    self.negative_arrival_times = None
                    self.negative_arrival_rates = None
                    self.negative_sources = None
        else:
            raise ValueError(f"Unknown negative_arrival_type: {self.negative_arrival_type}")

        self.is_sources_set = True

    def set_nodes(
        self,
        serv_params: list[dict],
        n: list[int],
        negative_types: list[NegativeServiceType] = None,
        buffers: list[int | None] = None,
    ):  # pylint: disable=arguments-differ
        """
        Set the service time distribution parameters and number of channels for each node.

        Parameters:
            serv_params: list of dictionaries with service parameters for each node
                [m][dict(type, params)]
                where m - node number,
                type - distribution type, params - distribution parameters.
            n: list of number of channels in each node.
            negative_types: list of NegativeServiceType for each node.
                If None, defaults to DISASTER for all nodes.
            buffers: list of buffer sizes for each node (None for infinite).
                If None, all nodes have infinite buffers.
        """
        self.serv_params = serv_params
        self.nodes = n
        self.n_num = len(n)

        # Set default negative types if not provided
        if negative_types is None:
            self.negative_types = [NegativeServiceType.DISASTER] * self.n_num
        else:
            self.negative_types = negative_types

        # Set default buffers if not provided
        if buffers is None:
            self.buffers = [None] * self.n_num
        else:
            self.buffers = buffers

        # Create nodes with negative job support
        for m in range(self.n_num):
            node = QsSimNegatives(
                num_of_channels=n[m],
                type_of_negatives=self.negative_types[m],
                buffer=self.buffers[m],
                verbose=False,  # Disable verbose for network nodes
            )
            node.set_servers(serv_params[m]["params"], kendall_notation=serv_params[m]["type"])
            self.qs.append(node)

        self.is_nodes_set = True

        # Initialize per-node negative arrival times if per_node type and rates not set yet
        if self.negative_arrival_type == "per_node" and self.negative_arrival_times is None:
            # Initialize with infinite times (disabled) until set_sources is called with negative_arrival_rates
            self.negative_arrival_times = [float("inf")] * self.n_num
            self.negative_arrival_rates = [0.0] * self.n_num
            self.negative_sources = [None] * self.n_num

    def set_node_sources(
        self,
        node_index: int,
        positive_params,
        positive_kendall: str = "M",
        negative_params=None,
        negative_kendall: str = "M",
    ):
        """
        Set positive and negative sources for a specific node.

        Parameters:
            node_index: Index of the node to configure.
            positive_params: Parameters for positive source distribution.
            positive_kendall: Kendall notation for positive source.
            negative_params: Parameters for negative source distribution (optional).
            negative_kendall: Kendall notation for negative source.
        """
        if node_index >= len(self.qs):
            raise ValueError(f"Node index {node_index} out of range")

        self.qs[node_index].set_positive_sources(positive_params, positive_kendall)
        if negative_params is not None:
            self.qs[node_index].set_negative_sources(negative_params, negative_kendall)

    def choose_next_node(self, current_node):
        """
        Choose the next node for a task based on the current node.

        Args:
            current_node: The current node of the task.

        Returns:
            The index of the next node.
        """
        sum_p = 0
        p = np.random.rand()
        for i in range(self.R.shape[0]):
            sum_p += self.R[current_node + 1, i]
            if sum_p > p:
                return i
        return 0

    def refresh_v_stat(self, new_a):
        """
        Refresh sojourn time statistics.

        Args:
            new_a: The new sojourn time value.
        """
        factor = 1.0 - (1.0 / self.served)
        a_pow = [math.pow(new_a, i + 1) for i in range(3)]
        for i in range(3):
            self.v_network[i] = self.v_network[i] * factor + a_pow[i] / self.served

    def refresh_w_stat(self, new_a):
        """
        Refresh waiting time statistics.

        Args:
            new_a: The new waiting time value.
        """
        factor = 1.0 - (1.0 / self.served)
        a_pow = [math.pow(new_a, i + 1) for i in range(3)]
        for i in range(3):
            self.w_network[i] = self.w_network[i] * factor + a_pow[i] / self.served

    def _get_network_events(self):
        """
        Collect all available network events including negative arrivals.
        """
        events = {}

        # Positive network arrivals
        if self.positive_arrival_time is not None and self.positive_arrival_time < float("inf"):
            events["network_arrival_positive"] = self.positive_arrival_time

        # Negative network arrivals
        if self.negative_arrival_type == "global":
            # Global negative arrivals
            if self.negative_arrival_time is not None and self.negative_arrival_time < float("inf"):
                events["network_arrival_negative"] = self.negative_arrival_time
        elif self.negative_arrival_type == "per_node":
            # Per-node negative arrivals
            if self.negative_arrival_times is not None:
                for node_idx, arrival_time in enumerate(self.negative_arrival_times):
                    if arrival_time < float("inf"):
                        events[f"network_arrival_negative_node_{node_idx}"] = arrival_time

        # Serving events from all nodes
        events.update(self._get_node_serving_events())

        # Custom events
        custom_events = self._get_custom_network_events()
        events.update(custom_events)

        return events

    def _execute_event(self, event_type: str):
        """
        Execute network event. Handle positive/negative arrivals and node serving.
        """
        if event_type == "network_arrival_positive":
            self._before_network_arrival()
            self.on_positive_arrival(self.positive_arrival_time)
            self._after_network_arrival()
        elif event_type == "network_arrival_negative":
            # Global negative arrival
            self._before_network_arrival()
            self.on_negative_arrival(self.negative_arrival_time, node_idx=None)
            self._after_network_arrival()
        elif event_type.startswith("network_arrival_negative_node_"):
            # Per-node negative arrival
            node_idx = int(event_type.split("_")[-1])
            self._before_network_arrival()
            self.on_negative_arrival(self.negative_arrival_times[node_idx], node_idx=node_idx)
            self._after_network_arrival()
        elif event_type.startswith("node_serving_"):
            parts = event_type.split("_")
            node = int(parts[2])
            channel = int(parts[3])
            self._before_node_serving(node, channel)
            result = self.on_serving(node, channel)
            self._after_node_serving(node, channel, result)
        else:
            super()._execute_event(event_type)

    def on_positive_arrival(self, arrival_time):
        """
        Handle positive arrival event.

        Args:
            arrival_time: Time of arrival.
        """
        self.ttek = arrival_time
        self.arrived += 1
        self.in_sys += 1

        self.positive_arrival_time = self.ttek + self.positive_source.generate()

        next_node = self.choose_next_node(-1)

        # Check if job should exit the network (next_node == n_num means exit)
        if next_node == self.n_num:
            # Job exits immediately without service
            self.served += 1
            self.in_sys -= 1
            return

        ts = Task(self.ttek, is_network=True)

        # Send to node - update node time and use arrival with task
        node = self.qs[next_node]
        # QsSimNegatives inherits from QsSim, so it has arrival() method
        # We pass the task directly with moment - this will handle it correctly
        node.arrival(moment=self.ttek, ts=ts)

    def _process_node_negative_arrival(self, node: QsSimNegatives):
        """
        Process negative arrival for a node without using node's negative source.
        This is used for network-level negative arrivals.

        Args:
            node: The node to process negative arrival for.
        """
        node.negative_arrived += 1

        # Update state probabilities
        node._update_state_probs(node.ttek, node.ttek, node.in_sys)

        if node.in_sys == 0:
            # If no jobs in system, negatives has no effect
            return

        if node.type_of_negatives == NegativeServiceType.DISASTER:
            not_free_servers = [c for c in range(node.n) if not node.servers[c].is_free]
            for c in not_free_servers:
                end_ts = node.servers[c].end_service()
                node._free_servers.add(c)
                node.broken += 1
                node.total += 1
                sojourn_time = node.ttek - end_ts.arr_time
                node.refresh_v_stat(sojourn_time)
                node.refresh_v_stat_broken(sojourn_time)

            node.in_sys = 0
            node.free_channels = node.n
            node._free_servers = set(range(node.n))
            node._mark_servers_time_changed()

            while node.queue.size() > 0:
                ts = node.queue.pop()
                wait_increment = node.ttek - ts.start_waiting_time
                ts.wait_time += wait_increment
                # Update wait_network if this is a network task
                if hasattr(ts, "wait_network") and ts.start_waiting_time >= 0:
                    ts.wait_network += wait_increment
                node.taked += 1
                node.total += 1
                node.broken += 1
                node.refresh_w_stat(ts.wait_time)
                sojourn_time = node.ttek - ts.arr_time
                node.refresh_v_stat(sojourn_time)
                node.refresh_v_stat_broken(sojourn_time)

        elif node.type_of_negatives == NegativeServiceType.RCE:
            if node.queue.size() > 0:
                node.in_sys -= 1
                ts = node.queue.tail()
                wait_increment = node.ttek - ts.start_waiting_time
                ts.wait_time += wait_increment
                # Update wait_network if this is a network task
                if hasattr(ts, "wait_network") and ts.start_waiting_time >= 0:
                    ts.wait_network += wait_increment
                node.taked += 1
                node.total += 1
                node.broken += 1
                node.refresh_w_stat(ts.wait_time)
                sojourn_time = node.ttek - ts.arr_time
                node.refresh_v_stat(sojourn_time)
                node.refresh_v_stat_broken(sojourn_time)
            # If queue is empty but in_sys > 0, tasks are in servers - RCE has no effect

        elif node.type_of_negatives == NegativeServiceType.RCH:
            if node.queue.size() > 0:
                node.in_sys -= 1
                ts = node.queue.pop()
                wait_increment = node.ttek - ts.start_waiting_time
                ts.wait_time += wait_increment
                # Update wait_network if this is a network task
                if hasattr(ts, "wait_network") and ts.start_waiting_time >= 0:
                    ts.wait_network += wait_increment
                node.taked += 1
                node.total += 1
                node.broken += 1
                node.refresh_w_stat(ts.wait_time)
                sojourn_time = node.ttek - ts.arr_time
                node.refresh_v_stat(sojourn_time)
                node.refresh_v_stat_broken(sojourn_time)
            # If queue is empty but in_sys > 0, tasks are in servers - RCH has no effect

        elif node.type_of_negatives == NegativeServiceType.RCS:
            not_free_servers = [c for c in range(node.n) if not node.servers[c].is_free]
            if not_free_servers:
                c = np.random.choice(not_free_servers)
                end_ts = node.servers[c].end_service()
                node._free_servers.add(c)
                node._mark_servers_time_changed()
                node.total += 1
                node.broken += 1
                node.free_channels += 1
                sojourn_time = node.ttek - end_ts.arr_time
                node.refresh_v_stat(sojourn_time)
                node.refresh_v_stat_broken(sojourn_time)
                node.in_sys -= 1

                if node.queue.size() != 0:
                    node.send_head_of_queue_to_channel(c)

    def on_negative_arrival(self, arrival_time, node_idx=None):
        """
        Handle negative arrival event.

        Args:
            arrival_time: Time of arrival.
            node_idx: Index of specific node (for per_node type).
                     If None, affects all nodes (for global type).
        """
        self.ttek = arrival_time

        if self.negative_arrival_type == "global":
            # Global negative arrival - affects all nodes
            if self.negative_source is not None:
                self.negative_arrival_time = self.ttek + self.negative_source.generate()
            else:
                # Zero rate - negative arrivals never occur
                self.negative_arrival_time = float("inf")

            # Send negative arrival to all nodes
            # For network-level negative arrivals, nodes need to process them
            # but they don't have their own negative sources set up
            # We'll temporarily set negative_source to None to avoid generation, or handle it differently
            for node in self.qs:
                if node.in_sys > 0:  # Only if there are jobs in the node
                    # Update node's current time to match network time
                    node.ttek = self.ttek
                    # Process negative arrival at network level (we'll create a helper)
                    self._process_node_negative_arrival(node)
        elif self.negative_arrival_type == "per_node":
            # Per-node negative arrival - affects specific node
            if node_idx is None:
                raise ValueError("node_idx must be provided for per_node negative arrival type")

            if node_idx < 0 or node_idx >= len(self.qs):
                raise ValueError(f"Invalid node_idx: {node_idx}")

            # Update arrival time for this specific node
            if self.negative_sources[node_idx] is not None:
                self.negative_arrival_times[node_idx] = self.ttek + self.negative_sources[node_idx].generate()

            # Send negative arrival to the specific node
            if self.qs[node_idx].in_sys > 0:
                node = self.qs[node_idx]
                node.ttek = self.ttek
                self._process_node_negative_arrival(node)

    def on_serving(self, node, channel):
        """
        Handle serving event in a node.

        Args:
            node: Node index.
            channel: Channel index.

        Returns:
            Task that completed service, or None if task was broken by negative job.
        """
        serving_time = self.qs[node].servers[channel].time_to_end_service
        self.ttek = serving_time

        # Check if server is still busy (not interrupted by negative job)
        if not self.qs[node].servers[channel].is_free:
            ts = self.qs[node].serving(channel, True)

            next_node = self.choose_next_node(node)

            if next_node == self.n_num:
                # Job leaves the network
                self.served += 1
                self.in_sys -= 1

                self.refresh_v_stat(self.ttek - ts.arr_network)
                self.refresh_w_stat(ts.wait_network)
            else:
                # Route to next node
                if next_node < 0 or next_node >= len(self.qs):
                    # Invalid routing - job is lost or there's an error
                    # Update statistics even for lost jobs
                    self.served += 1
                    self.in_sys -= 1
                    arr_network = getattr(ts, "arr_network", None)
                    if arr_network is not None:
                        self.refresh_v_stat(self.ttek - arr_network)
                    wait_network = getattr(ts, "wait_network", None)
                    if wait_network is not None:
                        self.refresh_w_stat(wait_network)
                else:
                    # Update the task's arrival time for the next node
                    ts.arr_time = self.ttek
                    # Reset wait_time for the next node (but keep wait_network accumulated)
                    ts.wait_time = 0
                    ts.start_waiting_time = -1
                    # arr_network сохраняется (не перезаписывается)
                    self.qs[next_node].arrival(moment=self.ttek, ts=ts)

            return ts
        else:
            # Server was freed by negative job, nothing to route
            return None

    def on_arrival(self, *args, **kwargs):
        """
        Placeholder to satisfy abstract method requirement.
        Actual arrivals are handled by on_positive_arrival and on_negative_arrival.
        """
        # This method is not used - we use on_positive_arrival and on_negative_arrival instead

    def run(self, job_served: int) -> NetworkResults:
        """
        Run simulation.

        Parameters:
            job_served: Number of jobs to serve.

        Returns:
            NetworkResults object containing results of simulation.
        """
        start = time.process_time()

        self._check_sources_and_nodes_is_set()

        last_percent = 0

        with tqdm(total=100) as pbar:
            while self.served < job_served:
                self.run_one_step()
                percent = int(100 * (self.served / job_served))
                if last_percent != percent:
                    last_percent = percent
                    pbar.update(1)
                    pbar.set_description(
                        Fore.MAGENTA
                        + "\rJob served: "
                        + Fore.YELLOW
                        + f"{self.served}/{job_served}"
                        + Fore.LIGHTGREEN_EX
                    )

        self.results = NetworkResults(
            v=self.v_network,
            served=self.served,
            arrived=self.arrived,
            duration=time.process_time() - start,
        )
        # Store waiting time statistics in results if needed
        if hasattr(self, "w_network"):
            self.results.w = self.w_network

        return self.results
