"""
Simulation of a priority network with priorities and multiple channels.
"""

import math
import time

import numpy as np
from colorama import Fore, init
from tqdm import tqdm

from most_queue.random.distributions import ExpDistribution
from most_queue.sim.networks.base_network_sim import (
    BaseSimNetworkPriority,
    NetworkResultsPriority,
)
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.sim.utils.tasks import TaskPriority

init()


class PriorityNetwork(BaseSimNetworkPriority):
    """
    Simulation of a priority network with priorities and multiple channels.
    """

    def __init__(self, k_num: int):
        """
        Initializes the network simulator.

        k_num: number of classes.
        """

        super().__init__()

        self.k_num = k_num  # number of classes
        self.L = None  # list of arrival intensities for each class
        self.R = None  # list of routing matrices for each class
        self.n_num = None  # number of nodes
        self.nodes = None  # n[i] - number of channels in node i

        self.prty = None  # prty[n] - priority type of node n

        self.serv_params = None
        self.nodes_prty = None

        self.qs = []

        self.arrival_time = []
        self.sources = []
        self.v_network = []
        self.w_network = []

        self.ttek = 0
        self.served = [0] * self.k_num
        self.in_sys = [0] * self.k_num
        self.arrived = [0] * self.k_num

    def set_sources(self, L: list[float], R: list[np.matrix]):  # pylint: disable=arguments-differ
        """
        Set the arrival rates and routing matrix for each class.
        Parameters:
            L: list of arrival intensities for each class.
            R: list of routing matrices for each class.

            each routing matrix, dim (m + 1 x m + 1), where m is number of nodes.
            For example:
            R[0][0, 0] is transition frome source to first node for the first class.
            R[0][0, m] is transition from source to out of system for the first class.

        """
        self.L = L
        self.R = R

        for k in range(self.k_num):
            self.sources.append(ExpDistribution(L[k]))
            self.arrival_time.append(self.sources[k].generate())
            self.v_network.append([0.0] * 3)
            self.w_network.append([0.0] * 3)

        self.is_sources_set = True

    def set_nodes(
        self,
        serv_params: list[list[dict]],
        n: list[int],
        prty: list[str],
        nodes_prty: list[list[int]],
    ):  # pylint: disable=arguments-differ
        """
        Set the service time distribution parameters and number of channels for each node.
        Parameters:
            serv_params: list of list of dictionaries with service parameters for each node and class.
                [m][k][dict(type, params)]
                where m - node number, k - class number,
                type - distribution type, params - distribution parameters.

            n: list of number of channels in each node.

            prty: list of priority types for each node.
                No  - no priorities, FIFO
                PR  - preemptive resume, with resuming interrupted request
                RS  - preemptive repeat with resampling, re-sampling duration for new service
                RW  - preemptive repeat without resampling, repeating service with previous duration
                NP  - non preemptive, relative priority

            nodes_prty: Priority distribution among requests for each
            node in the network [m][x1, x2 .. x_k],
                m - node number, xi - priority for i-th class, k - number of classes
                For example:
                    [0][0,1,2] - for the first node, a direct order of priorities is set,
                    [2][0,2,1] - for the third node, such an order of
                    priorities is set: for the first class - the oldest (0),
                    for the second - the youngest (2), for the third - intermediate (1)
        """
        self.serv_params = serv_params

        self.n_num = len(n)  # number of nodes
        self.nodes = n  # n[i] - number of channels in node i

        self.prty = prty  # prty[n] - priority type of node n

        self.nodes_prty = nodes_prty

        for m in range(self.n_num):
            self.qs.append(PriorityQueueSimulator(n[m], self.k_num, prty[m]))
            param_serv_reset = []

            for k in range(self.k_num):
                param_serv_reset.append(serv_params[m][nodes_prty[m][k]])

            self.qs[m].set_servers(param_serv_reset)

        self.is_nodes_set = True

    def choose_next_node(self, real_class, current_node):
        """
        Choose the next node for a task based on the current node and real class.
        :param real_class: The real class of the task.
        :param current_node: The current node of the task.
        :return: The index of the next node.
        """
        sum_p = 0
        p = np.random.rand()
        for i in range(self.R[real_class].shape[0]):
            sum_p += self.R[real_class][current_node + 1, i]
            if sum_p > p:
                return i
        return 0

    def refresh_v_stat(self, k, new_a):
        """
        Refresh sojourn time statistics for a given class and new arrival.
        :param k: The class of the task.
        :param new_a: The new arrival time.
        """
        factor = 1.0 - (1.0 / self.served[k])
        a_pow = [math.pow(new_a, i + 1) for i in range(3)]
        for i in range(3):
            self.v_network[k][i] = self.v_network[k][i] * factor + a_pow[i] / self.served[k]

    def refresh_w_stat(self, k, new_a):
        """
        Refresh waiting time statistics for a given class and new arrival.
        :param k: The class of the task.
        :param new_a: The new arrival time.

        """
        factor = 1.0 - (1.0 / self.served[k])
        a_pow = [math.pow(new_a, i + 1) for i in range(3)]
        for i in range(3):
            self.w_network[k][i] = self.w_network[k][i] * factor + a_pow[i] / self.served[k]

    def run_one_step(self):
        """
        Run one step of the simulation.
        """
        num_of_serv_ch_earlier = -1  # номер канала узла, мин время до окончания обслуживания
        num_of_k_earlier = -1  # номер класса, прибывающего через мин время
        num_of_node_earlier = -1  # номер узла, в котором раньше всех закончится обслуживание
        arrival_earlier = 1e10  # момент прибытия ближайшего
        serving_earlier = 1e10  # момент ближайшего обслуживания

        for k in range(self.k_num):
            if self.arrival_time[k] < arrival_earlier:
                num_of_k_earlier = k
                arrival_earlier = self.arrival_time[k]

        for node in range(self.n_num):
            for c in range(self.nodes[node]):
                if self.qs[node].servers[c].time_to_end_service < serving_earlier:
                    serving_earlier = self.qs[node].servers[c].time_to_end_service
                    num_of_serv_ch_earlier = c
                    num_of_node_earlier = node

        if arrival_earlier < serving_earlier:
            self.on_arrival(arrival_earlier, num_of_k_earlier)
        else:
            self.on_serving(serving_earlier, num_of_serv_ch_earlier, num_of_node_earlier)

    def on_arrival(self, arrival_earlier, num_of_k_earlier):
        """
        Handle arrival event
        """
        self.ttek = arrival_earlier
        self.arrived[num_of_k_earlier] += 1
        self.in_sys[num_of_k_earlier] += 1

        self.arrival_time[num_of_k_earlier] = self.ttek + self.sources[num_of_k_earlier].generate()

        next_node = self.choose_next_node(num_of_k_earlier, -1)

        ts = TaskPriority(num_of_k_earlier, self.ttek, True)

        next_node_class = self.nodes_prty[next_node][num_of_k_earlier]

        ts.in_node_class_num = next_node_class

        self.qs[next_node].arrival(next_node_class, self.ttek, ts)

    def on_serving(self, serving_earlier, num_of_serv_ch_earlier, num_of_node_earlier):
        """
        Handle serving event
        """
        self.ttek = serving_earlier
        ts = self.qs[num_of_node_earlier].serving(num_of_serv_ch_earlier, True)

        real_class = ts.k
        next_node = self.choose_next_node(real_class, num_of_node_earlier)

        if next_node == self.n_num:
            self.served[real_class] += 1
            self.in_sys[real_class] -= 1

            self.refresh_v_stat(real_class, self.ttek - ts.arr_network)
            self.refresh_w_stat(real_class, ts.wait_network)

        else:
            next_node_class = self.nodes_prty[next_node][real_class]

            self.qs[next_node].arrival(next_node_class, self.ttek, ts)

    def run(self, job_served: int) -> NetworkResultsPriority:
        """
        Run simulation

        Parameters:
           job_served: int - number of jobs to serve.
        Returns:
            NetworkResultsPriority - simulation results.
        """

        start = time.process_time()

        self._check_sources_and_nodes_is_set()

        last_percent = 0

        with tqdm(total=100) as pbar:
            while sum(self.served) < job_served:
                self.run_one_step()
                percent = int(100 * (sum(self.served) / job_served))
                if last_percent != percent:
                    last_percent = percent
                    pbar.update(1)
                    pbar.set_description(
                        Fore.MAGENTA
                        + "\rJob served: "
                        + Fore.YELLOW
                        + f"{sum(self.served)}/{job_served}"
                        + Fore.LIGHTGREEN_EX
                    )
        self.results = NetworkResultsPriority(
            v=self.v_network, arrived=self.arrived, served=self.served, duration=time.process_time() - start
        )

        return self.results
