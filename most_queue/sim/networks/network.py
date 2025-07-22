"""
Simulation of a priority network with priorities and multiple channels.
"""

import math
import time

import numpy as np
from colorama import Fore, init
from tqdm import tqdm

from most_queue.random.distributions import ExpDistribution
from most_queue.sim.base import QsSim
from most_queue.sim.networks.base_network_sim import BaseSimNetwork, NetworkResults
from most_queue.sim.utils.tasks import Task

init()


class NetworkSimulator(BaseSimNetwork):
    """
    Simulation of network with multiple channels in each node.
    """

    def __init__(self):
        """
        Initialize the network simulator.
        """

        super().__init__()

        self.arrival_time = None
        self.R = None
        self.n_num = None  # number of nodes
        self.nodes = None  # n[i] - number of channels in node i

        self.serv_params = None
        self.arrival_rate = None
        self.arrival_time = None
        self.source = None

        self.qs = []

        self.v_network = [0.0] * 3
        self.w_network = [0.0] * 3

        self.ttek = 0
        self.served = 0
        self.in_sys = 0
        self.arrived = 0

    def set_sources(self, arrival_rate: float, R: np.matrix):  # pylint: disable=arguments-differ
        """
        Set the arrival rate and routing matrix.
        Parameters:
            arrival_rate: arrival rate of customers.
            R: routing matrix, dim (m + 1 x m + 1), where m is number of nodes.

            For example:
            R[0, 0] is transition frome source to first node.
            R[0, m] is transition from source to out of system.

        """
        self.arrival_rate = arrival_rate

        self.source = ExpDistribution(self.arrival_rate)
        self.arrival_time = self.source.generate()

        self.R = R
        self.is_sources_set = True

    def set_nodes(self, serv_params: list[dict], n: list[int]):  # pylint: disable=arguments-differ
        """
        Set the service time distribution parameters and number of channels for each node.
        Parameters:
            serv_params: list of dictionaries with service parameters for each node
                [m][dict(type, params)]
                where m - node number,
                type - distribution type, params - distribution parameters.
            n: list of number of channels in each node.
        """
        self.serv_params = serv_params
        self.nodes = n
        self.n_num = len(n)  # number of nodes

        for m in range(self.n_num):
            self.qs.append(QsSim(n[m]))
            self.qs[m].set_servers(serv_params[m]["params"], kendall_notation=serv_params[m]["type"])

        self.is_nodes_set = True

    def choose_next_node(self, current_node):
        """
        Choose the next node for a task based on the current node
        :param current_node: The current node of the task.
        :return: The index of the next node.
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
        Refresh sojourn time statistics
        :param new_a: The new arrival time.
        """
        factor = 1.0 - (1.0 / self.served)
        a_pow = [math.pow(new_a, i + 1) for i in range(3)]
        for i in range(3):
            self.v_network[i] = self.v_network[i] * factor + a_pow[i] / self.served

    def refresh_w_stat(self, new_a):
        """
        Refresh waiting time statistics
        :param new_a: The new arrival time.

        """
        factor = 1.0 - (1.0 / self.served)
        a_pow = [math.pow(new_a, i + 1) for i in range(3)]
        for i in range(3):
            self.w_network[i] = self.w_network[i] * factor + a_pow[i] / self.served

    def run_one_step(self):
        """
        Run one step of the simulation.
        """
        num_of_serv_ch_earlier = -1  # номер канала узла, мин время до окончания обслуживания
        num_of_node_earlier = -1  # номер узла, в котором раньше всех закончится обслуживание
        arrival_earlier = 1e10  # момент прибытия ближайшего
        serving_earlier = 1e10  # момент ближайшего обслуживания

        arrival_earlier = min(arrival_earlier, self.arrival_time)

        for node in range(self.n_num):
            for c in range(self.nodes[node]):
                if self.qs[node].servers[c].time_to_end_service < serving_earlier:
                    serving_earlier = self.qs[node].servers[c].time_to_end_service
                    num_of_serv_ch_earlier = c
                    num_of_node_earlier = node

        if arrival_earlier < serving_earlier:
            self.on_arrival(arrival_earlier)
        else:
            self.on_serving(serving_earlier, num_of_serv_ch_earlier, num_of_node_earlier)

    def on_arrival(self, arrival_earlier):
        """
        Handle arrival event
        """
        self.ttek = arrival_earlier
        self.arrived += 1
        self.in_sys += 1

        self.arrival_time = self.ttek + self.source.generate()

        next_node = self.choose_next_node(-1)

        ts = Task(self.ttek, is_network=True)

        self.qs[next_node].arrival(self.ttek, ts)

    def on_serving(self, serving_earlier, num_of_serv_ch_earlier, num_of_node_earlier):
        """
        Handle serving event
        """
        self.ttek = serving_earlier
        ts = self.qs[num_of_node_earlier].serving(num_of_serv_ch_earlier, True)

        next_node = self.choose_next_node(num_of_node_earlier)

        if next_node == self.n_num:
            self.served += 1
            self.in_sys -= 1

            self.refresh_v_stat(self.ttek - ts.arr_network)
            self.refresh_w_stat(ts.wait_network)

        else:
            self.qs[next_node].arrival(self.ttek, ts)

    def run(self, job_served: int) -> NetworkResults:
        """
        Run simulation
        Parameters:
           job_served (int): Number of jobs to serve.

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
            v=self.v_network, served=self.served, arrived=self.arrived, duration=time.process_time() - start
        )

        return self.results
