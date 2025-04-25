"""
Test the priority queueing network simulation (IM SEMO) with priority queues at nodes.
Compare results with numerical calculations using decomposition method.
"""
import numpy as np

from most_queue.general.tables import times_print_with_classes
from most_queue.rand_distribution import H2Distribution
from most_queue.sim.networks.priority_network import PriorityNetwork
from most_queue.theory.networks.open_network import OpenNetworkCalc


def test_network():
    """
    Test the priority queueing network simulation (IM SEMO) with priority queues at nodes.
    Compare results with numerical calculations using decomposition method.
    """
    k_num = 3  # Number of job classes
    n_num = 5  # Number of network nodes

    n = [3, 2, 3, 4, 3]  # Distribution of channels (servers) at each node
    R = []  # List of transition probability matrices for each class
    b = []  # List of service time moments for each class and node
    for i in range(k_num):
        R.append(np.matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 0.4, 0.6, 0, 0, 0],
            [0, 0, 0, 0.6, 0.4, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ]))
    L = [0.1, 0.3, 0.4]  # Arrival rates for each class
    nodes_prty = []  # Priority distribution at each node: [node][class]

    jobs_num = 200000  # Number of jobs simulated

    serv_params = []

    h2_params = []
    for m in range(n_num):
        nodes_prty.append([])
        for j in range(k_num):
            if m % 2 == 0:
                nodes_prty[m].append(j)
            else:
                nodes_prty[m].append(k_num - j - 1)

        b1 = 0.9 * n[m] / sum(L)
        coev = 1.2
        h2_params.append(H2Distribution.get_params_by_mean_and_coev(b1, coev))

        serv_params.append([])
        for i in range(k_num):
            serv_params[m].append({'type': 'H', 'params': h2_params[m]})

    for k in range(k_num):
        b.append([])
        for m in range(n_num):
            b[k].append(H2Distribution.calc_theory_moments(h2_params[m], 4))

    prty = ['NP'] * n_num

    # Create simulation of priority network:
    qn = PriorityNetwork(k_num, L, R, n, prty, serv_params, nodes_prty)

    #  Run simulation of priority network:
    qn.run(jobs_num)

    # Get initial moments of soujorney time from simulation:
    v_im = qn.v_network

    #  Get initial moments of soujorney time from calculation:
    net_calc = OpenNetworkCalc(R, b, n, L, prty, nodes_prty)
    semo_calc = net_calc.run()
    v_ch = semo_calc['v']

    # Get utilization factor of each node
    loads = semo_calc['loads']

    #  Print results

    print("\n")
    print("-" * 60)
    print("{0:^60s}\n{1:^60s}".format("Comparison of simulation and calculation Results",
                                      "for Priority Queuing Network with Multiple Channels"))
    print("-" * 60)
    print(f"Channels at nodes: {n}")
    print(
        f"Node utilization coefficients: {[float(round(load, 3)) for load in loads]}")

    assert abs(v_im[0][0] - v_ch[0][0]) / \
        max(v_im[0][0], v_ch[0][0]) * 100 < 10

    print("-" * 60)
    print("Relative Priority ('NP')")
    times_print_with_classes(v_im, v_ch, False)

    prty = ['PR'] * n_num  # Absolute priority at each node
    qn = PriorityNetwork(k_num, L, R, n, prty, serv_params, nodes_prty)
    qn.run(jobs_num)
    v_im = qn.v_network

    net_calc = OpenNetworkCalc(R, b, n, L, prty, nodes_prty)
    semo_calc = net_calc.run()
    v_ch = semo_calc['v']

    print("-" * 60)
    print("Absolute Priority ('PR')")
    times_print_with_classes(v_im, v_ch, False)

    assert abs(v_im[0][0] - v_ch[0][0]) / \
        max(v_im[0][0], v_ch[0][0]) * 100 < 10


if __name__ == "__main__":

    test_network()
