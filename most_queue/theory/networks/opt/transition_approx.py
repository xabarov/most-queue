"""
Class for optimizing the transition matrix for an open network.
Version with additional features and optimizations.
}
"""
import numpy as np

from most_queue.theory.networks.opt.transition import (
    ChildLoadBalanceResults,
    LoadBalanceResults,
    MaxLoadNodeResults,
    NetworkOptimizer,
    OptimizerDynamic,
)
from most_queue.theory.utils.utilization_approx import (
    find_delta_utilization,
    v1_on_utilization_approx,
)


class NetworkOptimizerWithApprox(NetworkOptimizer):
    """
    Class for optimizing the transition matrix for an open network.
    Version with additional features and optimizations.
    """

    def _find_balance_approx(self, loads: list[float],
                             max_node_res: MaxLoadNodeResults,
                             intencities: list[float],
                             deg=4) -> LoadBalanceResults:
        """
        Load balancing algorithm. Moves load from max load node to its children nodes.
        :param loads: list of loads on nodes
        :param max_node_res: MaxLoadNodeResults
        :return: LoadBalanceResults 
        """

        parent = max_node_res.parent
        max_load_node = max_node_res.node

        childrens = [k for k in range(
            self.rows-1) if self.R[parent, k] > 0 and k != max_load_node]

        min_load_child = -1
        min_loads = 1e10
        for child in childrens:
            if loads[child] < min_loads:
                min_load_child = child
                min_loads = loads[child]

        max_node_poly = v1_on_utilization_approx(channels=self.num_channels[max_load_node],
                                                 arrival_rate=intencities[max_load_node], deg=deg)
        min_node_poly = v1_on_utilization_approx(channels=self.num_channels[min_load_child],
                                                 arrival_rate=intencities[min_load_child], deg=deg)

        delta_ro = find_delta_utilization(poly1=max_node_poly, poly2=min_node_poly,
                                          load1=loads[max_load_node], load2=loads[min_load_child])

        lam = 0
        if parent == 0:
            lam = self.arrival_rate
        else:
            lam = intencities[parent-1]

        r_d = self.R[parent, max_load_node]
        z = delta_ro * \
            self.num_channels[max_load_node]/(r_d*lam*self.b[max_load_node][0])
        return LoadBalanceResults(z=z,
                                  children=[ChildLoadBalanceResults(child=min_load_child,
                                                                    z=z)])

    def run(self, tolerance=1e-8, max_steps=100, approx_deg: int = 4):
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

            z_res = self._find_balance_approx(loads, max_node_res,
                                              intencities, deg=approx_deg)

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
