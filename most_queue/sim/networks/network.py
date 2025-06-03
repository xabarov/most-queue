"""
Simulation of a priority network with priorities and multiple channels.
"""
import math

import numpy as np
from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.rand_distribution import ExpDistribution
from most_queue.sim.base import QsSim
from most_queue.sim.utils.tasks import Task

init()


class NetworkSimulator:
    """
    Simulation of network with multiple channels in each node.
    """

    def __init__(self, arrival_rate: float, R: np.matrix, n: list[int],
                 serv_params):
        """
        arrival_rate: arrival rate of tasks.

        R: routing matrix
        n: list of number of channels in each node.

        serv_params: list of dictionaries with service parameters for each node 
            [m][dict(type, params)]
            where m - node number, 
            type - distribution type, params - distribution parameters.
        """
        self.arrival_time = arrival_rate
        self.R = R  
        self.n_num = len(n)  # number of nodes
        self.nodes = n  # n[i] - number of channels in node i

        self.serv_params = serv_params

        self.qs = []

        for m in range(self.n_num):
            self.qs.append(QsSim(n[m]))
            self.qs[m].set_servers(serv_params[m]['params'], kendall_notation=serv_params[m]['type'])

        self.source = ExpDistribution(self.arrival_time)
        self.arrival_time = self.source.generate()
        self.v_network = [0.0] * 3
        self.w_network = [0.0] * 3

        self.ttek = 0
        self.total = 0
        self.served = 0
        self.in_sys = 0
        self.arrived = 0

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
        factor = (1.0 - (1.0 / self.served))
        a_pow = [math.pow(new_a, i + 1) for i in range(3)]
        for i in range(3):
            self.v_network[i] = self.v_network[i] * \
                factor + a_pow[i] / self.served

    def refresh_w_stat(self, new_a):
        """
        Refresh waiting time statistics 
        :param new_a: The new arrival time.

        """
        factor = (1.0 - (1.0 / self.served))
        a_pow = [math.pow(new_a, i + 1) for i in range(3)]
        for i in range(3):
            self.w_network[i] = self.w_network[i] * \
                factor + a_pow[i] / self.served

    def run_one_step(self):
        """
        Run one step of the simulation.
        """
        num_of_serv_ch_earlier = -1  # номер канала узла, мин время до окончания обслуживания
        num_of_node_earlier = -1  # номер узла, в котором раньше всех закончится обслуживание
        arrival_earlier = 1e10  # момент прибытия ближайшего
        serving_earlier = 1e10  # момент ближайшего обслуживания

        if self.arrival_time < arrival_earlier:
            arrival_earlier = self.arrival_time

        for node in range(self.n_num):
            for c in range(self.nodes[node]):
                if self.qs[node].servers[c].time_to_end_service < serving_earlier:
                    serving_earlier = self.qs[node].servers[c].time_to_end_service
                    num_of_serv_ch_earlier = c
                    num_of_node_earlier = node

        if arrival_earlier < serving_earlier:
            self.on_arrival(arrival_earlier)
        else:
            self.on_serving(serving_earlier,
                            num_of_serv_ch_earlier, num_of_node_earlier)

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
        ts = self.qs[num_of_node_earlier].serving(
            num_of_serv_ch_earlier, True)

        next_node = self.choose_next_node(num_of_node_earlier)

        if next_node == self.n_num:
            self.served += 1
            self.in_sys -= 1

            self.refresh_v_stat(self.ttek - ts.arr_network)
            self.refresh_w_stat(ts.wait_network)

        else:
            self.qs[next_node].arrival(self.ttek, ts)

    def run(self, job_served, is_real_served=True):
        """
        Run simulation
        """
        if is_real_served:
            last_percent = 0

            with tqdm(total=100) as pbar:
                while self.served < job_served:
                    self.run_one_step()
                    percent = int(100*(self.served/job_served))
                    if last_percent != percent:
                        last_percent = percent
                        pbar.update(1)
                        pbar.set_description(Fore.MAGENTA + '\rJob served: ' +
                                             Fore.YELLOW + f'{self.served}/{job_served}' + Fore.LIGHTGREEN_EX)
        else:
            print(Fore.GREEN + '\rStart simulation')
            print(Style.RESET_ALL)

            for _ in tqdm(range(job_served)):
                self.run_one_step()

            print(Fore.GREEN + '\rSimulation is finished')
            print(Style.RESET_ALL)
