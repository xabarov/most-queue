"""
Test fork-join stations embedded in an open network: an isolated fork-join
node must reduce to the standalone fork-join model, and a mixed tandem is
validated against the network simulator.
"""

import numpy as np

from most_queue.sim.networks.fork_join_network import ForkJoinNetworkSim
from most_queue.theory.fork_join.m_m_n import ForkJoinMarkovianCalc
from most_queue.theory.networks.fork_join_network import OpenNetworkCalcForkJoin

ARRIVAL_RATE = 0.5
MU_FJ = 1.0
K = 3

# Single node: source -> node 0 -> out
SINGLE = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
)

# Tandem: source -> M/M/2 -> fork-join(k=3) -> out
TANDEM = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def test_isolated_fj_node_reduces_to_standalone():
    """A network of one fork-join node must equal ForkJoinMarkovianCalc."""
    net = OpenNetworkCalcForkJoin()
    net.set_sources(arrival_rate=ARRIVAL_RATE, R=SINGLE)
    net.set_nodes([{"kind": "fork_join", "mu": MU_FJ, "k": K}])
    net_res = net.run()

    fj = ForkJoinMarkovianCalc(n=K)
    fj.set_sources(l=ARRIVAL_RATE)
    fj.set_servers(mu=MU_FJ)
    standalone = fj.run(approx="varma")

    assert np.isclose(net_res.v[0], standalone.v[0], rtol=1e-12)


def test_isolated_fj_node_vs_simulation():
    """Network simulator on a single fork-join node vs the Varma approximation."""
    net = OpenNetworkCalcForkJoin()
    net.set_sources(arrival_rate=ARRIVAL_RATE, R=SINGLE)
    net.set_nodes([{"kind": "fork_join", "mu": MU_FJ, "k": K}])
    net_res = net.run()

    sim = ForkJoinNetworkSim(seed=42)
    sim.set_sources(arrival_rate=ARRIVAL_RATE, R=SINGLE)
    sim.set_nodes([{"kind": "fork_join", "serv": {"type": "M", "params": MU_FJ}, "k": K}])
    sim_res = sim.run(200_000)

    assert np.isclose(net_res.v[0], sim_res.v[0], rtol=0.05)


def test_tandem_with_fj_vs_simulation():
    """M/M/2 -> fork-join(k=3) tandem: decomposition vs simulation."""
    nodes_calc = [
        {"kind": "queue", "mu": 0.4, "n": 2},
        {"kind": "fork_join", "mu": MU_FJ, "k": K},
    ]
    net = OpenNetworkCalcForkJoin()
    net.set_sources(arrival_rate=ARRIVAL_RATE, R=TANDEM)
    net.set_nodes(nodes_calc)
    net_res = net.run()

    sim = ForkJoinNetworkSim(seed=42)
    sim.set_sources(arrival_rate=ARRIVAL_RATE, R=TANDEM)
    sim.set_nodes(
        [
            {"kind": "queue", "serv": {"type": "M", "params": 0.4}, "n": 2},
            {"kind": "fork_join", "serv": {"type": "M", "params": MU_FJ}, "k": K},
        ]
    )
    sim_res = sim.run(200_000)

    assert np.isclose(net_res.v[0], sim_res.v[0], rtol=0.08), f"{net_res.v[0]} vs sim {sim_res.v[0]}"


if __name__ == "__main__":
    test_isolated_fj_node_reduces_to_standalone()
    test_isolated_fj_node_vs_simulation()
    test_tandem_with_fj_vs_simulation()
