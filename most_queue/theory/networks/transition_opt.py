"""
Class for optimizing the transition matrix for an open network.
@article{рыжиков2019численные,
  title={Численные методы теории очередей},
  author={Рыжиков, ЮИ},
  journal={Учебное пособие--М.: Лань},
  year={2019}
}
"""
from dataclasses import dataclass

import numpy as np
from colorama import Fore, Style, init

from most_queue.theory.networks.open_network import OpenNetworkCalc

init()


@dataclass
class MaxLoadNodeResults:
    """
    Results of finding the maximum load node in an open network.
    """
    node: int
    lam_r_max: float
    optimized: bool
    parent: int


@dataclass
class ChildLoadBalanceResults:
    """
    Results of a child node load balancing
    """
    child: int
    z: float


@dataclass
class LoadBalanceResults:
    """
    Results of load balancing
    """
    z: float
    children: list[ChildLoadBalanceResults]


@dataclass
class OptimizerDynamic:
    """
    Dynamic data for optimization process
    """
    loads: list[float]
    v1: float


class NetworkOptimizer:
    """
    Class for optimizing the transition matrix for an open network.
    """

    def __init__(self, transition_matrix: np.ndarray,
                 arrival_rate: float,
                 b: list[list[float]],
                 num_channels: list[int],
                 maximum_rates_to_end: list[float] | None,
                 is_service_markovian=False,
                 verbose: bool = False):
        """
        :param transition_matrix: Transition matrix of the network.
        :param arrival_rate: Arrival rate of customers
        :param b: E[x^k], k=1,2,3 r serving time moments for each node, [n][k], n = number of node
        :param num_channels: Number of channels for each node, [n], n = number of node
        :param maximum_rates_to_end: Maximum rate to end for each node, [n], n = number of node
        :param is_markovian: If service time distribution of the network is markovian or not.

        """
        self.R = transition_matrix.copy()
        self.rows = self.R.shape[0]
        if maximum_rates_to_end is None:
            maximum_rates_to_end = [self.R[i, -1] for i in range(self.rows)]

        self.maximum_rates_to_end = maximum_rates_to_end
        self.b = b
        self.num_channels = num_channels
        self.arrival_rate = arrival_rate
        self.is_markovian = is_service_markovian
        self.nodes_optimized = [False]*(self.rows-1)
        self.dynamics = []
        self.verbose = verbose

    def _maximize_outs(self):
        """
        Maximize the out rates of the network.
        """

        # Step 1. Recalculate intensities for maximum_rates_to_end

        for i in range(self.rows):
            if self.R[i, -1] < self.maximum_rates_to_end[i]:
                old_value = self.R[i, -1]
                self.R[i, -1] = self.maximum_rates_to_end[i]
                for j in range(self.rows):
                    if j != self.rows-1:
                        self.R[i, j] = self.R[i, j] * \
                            (1 - self.maximum_rates_to_end[i]) / (1-old_value)

    def _optimize_loops(self):
        """
        Eliminate loops from the network
        """
        for i in range(self.rows):
            for j in range(i):
                if self.R[i, j] > 0:
                    backward_rate = self.R[i, j]
                    self.R[i, j] = 0
                    for k in range(i, self.rows):
                        self.R[i, k] = self.R[i, k]/(1.0-backward_rate)

    def _get_network_calc(self):
        """
        Calculate the network using OpenNetworkCalc
        """
        net_calc = OpenNetworkCalc(
            self.R, self.b, self.num_channels, self.arrival_rate)
        return net_calc.run(is_markovian=self.is_markovian)

    def print_last_state(self, header='Load balance dynamics'):
        """
        Print the load balance dynamics
        """
        self._print_header(header=header)
        self._print_state()
        self._print_line()

    def _print_state(self):
        state = self.dynamics[-1]
        print(Fore.CYAN + '|' + Fore.YELLOW +
              f'{len(self.dynamics):^8}'+Fore.CYAN + '|', end='')
        for load in state.loads:
            print(Fore.YELLOW+f'{load:^8.3f}'+Fore.CYAN + '|', end='')

        print(Fore.YELLOW+f'{state.v1:^12.3f}'+Fore.CYAN + '|'+Style.RESET_ALL)

    def _print_header(self, header='Load balance dynamics'):
        v1_header = 'v1'
        first_col = 'Iter'

        line_long = 9*(self.rows) + 14
        print(Fore.CYAN + '-'*line_long)
        print(Fore.LIGHTGREEN_EX + f'{header:^{line_long}}'+Fore.CYAN)
        print('-'*line_long)

        print(f'|{first_col:^8}|', end='')
        for i in range(self.rows - 1):
            print(f'{i+1:^8}|', end='')

        print(f'{v1_header:^12}|')

        print('-'*line_long+Style.RESET_ALL)

    def _print_line(self):
        line_long = 9*(self.rows) + 14
        print(Fore.CYAN + '-'*line_long+Style.RESET_ALL)

    def _check_if_optimized(self):
        """
        If all nodes are optimized, return True. Otherwise, return False.
        """
        if all(self.nodes_optimized):
            return True

    def _find_max_load_node(self, loads: list[float],
                            intensities: list[float]) -> MaxLoadNodeResults:
        """
        Find the node with the maximum load
        """

        if self._check_if_optimized():
            return MaxLoadNodeResults(optimized=True, lam_r_max=None,
                                      parent=None, node=None)

        max_load_node = -1
        max_load = -1
        for i, load in enumerate(loads):
            if self.nodes_optimized[i]:
                continue
            if load > max_load:
                max_load = load
                max_load_node = i

        # Find parent with max intesity*P[parent, max_load_node]
        parent = -1
        lam_r_max = -1
        for i in range(max_load_node+1):
            if i == 0:  # source
                lam = self.arrival_rate
            else:
                lam = intensities[i-1]
            lam_r = lam * self.R[i, max_load_node]
            if lam_r > lam_r_max:
                # check if parent has >=2 childrens
                childrens = np.sum(
                    [1 if self.R[i, k] > 0 else 0 for k in range(self.rows-1)])
                if childrens >= 2:
                    parent = i
                    lam_r_max = lam_r

        if parent == -1:
            self.nodes_optimized[max_load_node] = True
            return self._find_max_load_node(loads, intensities)

        return MaxLoadNodeResults(optimized=False, lam_r_max=lam_r_max,
                                  parent=parent, node=max_load_node)

    def _find_balance(self, loads: list[float],
                      max_node_res: MaxLoadNodeResults) -> LoadBalanceResults:
        """
        Load balancing algorithm. Moves load from max load node to its children nodes.
        :param loads: list of loads on nodes
        :param max_node_res: MaxLoadNodeResults
        :return: LoadBalanceResults 
        """

        parent = max_node_res.parent
        max_load_node = max_node_res.node
        lam_r_max = max_node_res.lam_r_max

        childrens = [k for k in range(
            self.rows-1) if self.R[parent, k] > 0 and k != max_load_node]

        sum1 = 0
        sum2 = 0
        for child in childrens:
            nb = self.num_channels[child]/self.b[child][0]
            sum1 += nb
            sum2 += nb*loads[child]

        numerator = loads[max_load_node]*sum1 - sum2
        bn_j = self.b[max_load_node][0]/self.num_channels[max_load_node]
        denominator = lam_r_max*(1 + bn_j*sum1)

        z = numerator / denominator

        children_res = []

        for child in childrens:
            nb_child = self.num_channels[child]/self.b[child][0]
            delta_x = (loads[max_load_node]-loads[child])/lam_r_max
            z_child = nb_child*(delta_x - bn_j*z)
            children_res.append(
                ChildLoadBalanceResults(child=child, z=z_child))

        return LoadBalanceResults(z=z, children=children_res)

    def _balance(self, parent: int, max_load_node: int,
                 balance_results: LoadBalanceResults):
        """
        Balance the network loads
        """

        old_r = self.R[parent, max_load_node]

        z = balance_results.z

        self.R[parent, max_load_node] = (1-z)*old_r
        for child_res in balance_results.children:
            self.R[parent, child_res.child] += child_res.z*old_r

    def run(self, tolerance=1e-8, max_steps=100):
        """
        Run the optimization algorithm.
        """

        self._maximize_outs()
        self._optimize_loops()

        if self.verbose:
            self._print_header()

        net_res = self._get_network_calc()

        loads = net_res['loads']
        intencities = net_res['intensities']
        current_v1 = net_res['v'][0]
        self.dynamics.append(OptimizerDynamic(v1=current_v1, loads=loads))

        if self.verbose:
            self._print_state()

        old_v1 = 1e10
        step_num = 0

        while np.fabs(old_v1 - current_v1) > tolerance:

            if step_num > max_steps:
                break

            max_node_res = self._find_max_load_node(
                loads, intencities)

            if max_node_res.optimized:
                break

            z_res = self._find_balance(loads, max_node_res)

            parent = max_node_res.parent
            max_load_node = max_node_res.node

            self._balance(parent, max_load_node, z_res)

            old_v1 = current_v1

            net_res = self._get_network_calc()

            loads = net_res['loads']
            intencities = net_res['intensities']
            current_v1 = net_res['v'][0]

            self.dynamics.append(OptimizerDynamic(v1=current_v1, loads=loads))

            if self.verbose:
                self._print_state()
            step_num += 1
            
        if self.verbose:
            self._print_line()

        return self.R, current_v1
