"""
Test BCMP networks: single-class reductions to Jackson / single-chain MVA
(machine precision) and a 2-class closed PS network against the exact CTMC.
"""

import numpy as np

from most_queue.theory.networks.bcmp_network import BCMPClosedNetworkCalc, BCMPOpenNetworkCalc
from most_queue.theory.networks.closed_network import ClosedNetworkCalc
from most_queue.theory.networks.jackson_network import JacksonNetworkCalc

OPEN_ROUTING = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.4, 0.6],
        [0.5, 0.0, 0.5],
    ]
)


def test_open_single_class_reduces_to_jackson():
    """Single-class open BCMP with PS stations equals the Jackson network."""
    mu = [2.0, 3.0]
    bcmp = BCMPOpenNetworkCalc()
    bcmp.set_sources(arrival_rates=[1.0], R=[OPEN_ROUTING])
    bcmp.set_nodes(s=[[1.0 / mu[0]], [1.0 / mu[1]]], station_types=["ps", "ps"])
    bcmp_res = bcmp.run()

    jackson = JacksonNetworkCalc()
    jackson.set_sources(arrival_rate=1.0, R=OPEN_ROUTING)
    jackson.set_nodes(mu=mu, n=[1, 1])
    jackson_res = jackson.run()

    assert np.allclose(bcmp_res.loads, jackson_res.loads, rtol=1e-12)
    assert np.allclose(bcmp_res.mean_jobs[0], jackson_res.mean_jobs, rtol=1e-12)
    assert np.isclose(bcmp_res.v[0][0], jackson_res.v[0], rtol=1e-12)


def test_open_two_identical_classes_aggregate():
    """Two identical classes with half rates must aggregate to one class."""
    s = [[0.4, 0.4], [0.25, 0.25]]
    two = BCMPOpenNetworkCalc()
    two.set_sources(arrival_rates=[0.5, 0.5], R=[OPEN_ROUTING, OPEN_ROUTING])
    two.set_nodes(s=s, station_types=["ps", "fcfs"])
    two_res = two.run()

    one = BCMPOpenNetworkCalc()
    one.set_sources(arrival_rates=[1.0], R=[OPEN_ROUTING])
    one.set_nodes(s=[[0.4], [0.25]], station_types=["ps", "fcfs"])
    one_res = one.run()

    assert np.allclose(two_res.loads, one_res.loads, rtol=1e-12)
    assert np.isclose(two_res.v[0][0], one_res.v[0][0], rtol=1e-12)
    assert np.isclose(two_res.v[1][0], one_res.v[0][0], rtol=1e-12)


def test_closed_single_chain_reduces_to_mva():
    """Single-chain BCMP MVA equals the single-class exact MVA (with delay)."""
    routing = np.array(
        [
            [0.1, 0.5, 0.4],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    b = [0.02, 0.06, 0.08]
    n_pop = 6

    bcmp = BCMPClosedNetworkCalc()
    bcmp.set_sources(R=[routing], N=[n_pop])
    bcmp.set_nodes(s=[[b[0]], [b[1]], [b[2]]], station_types=["fcfs", "ps", "lcfs_pr"])
    bcmp_res = bcmp.run()

    mva = ClosedNetworkCalc(method="mva")
    mva.set_sources(R=routing, N=n_pop)
    mva.set_nodes(b=b, n=[1, 1, 1])
    mva_res = mva.run()

    assert np.isclose(bcmp_res.throughput[0], mva_res.throughput, rtol=1e-12)
    assert np.allclose(bcmp_res.mean_jobs[0], mva_res.mean_jobs, rtol=1e-10)


def test_closed_two_class_ps_vs_ctmc():
    """2-node, 2-class closed PS network against the exact CTMC."""
    # Both classes cycle 0 <-> 1; class-dependent PS service means
    cycle = np.array([[0.0, 1.0], [1.0, 0.0]])
    s = [[0.5, 0.8], [0.3, 0.4]]  # s[node][class]
    populations = [2, 2]

    bcmp = BCMPClosedNetworkCalc()
    bcmp.set_sources(R=[cycle, cycle], N=populations)
    bcmp.set_nodes(s=s, station_types=["ps", "ps"])
    res = bcmp.run()

    # CTMC over states (k0a, k0b) — class a/b jobs at node 0
    n_a, n_b = populations
    states = [(ka, kb) for ka in range(n_a + 1) for kb in range(n_b + 1)]
    index = {st: i for i, st in enumerate(states)}
    n_states = len(states)
    Q = np.zeros((n_states, n_states))

    mu = [[1.0 / s[i][r] for r in range(2)] for i in range(2)]
    for ka, kb in states:
        i_from = index[(ka, kb)]
        k0 = [ka, kb]
        k1 = [n_a - ka, n_b - kb]
        tot0, tot1 = sum(k0), sum(k1)
        for r in range(2):
            if k0[r] > 0:  # completion at node 0 -> node 1
                rate = mu[0][r] * k0[r] / tot0
                tgt = (ka - 1, kb) if r == 0 else (ka, kb - 1)
                Q[i_from, index[tgt]] += rate
            if k1[r] > 0:  # completion at node 1 -> node 0
                rate = mu[1][r] * k1[r] / tot1
                tgt = (ka + 1, kb) if r == 0 else (ka, kb + 1)
                Q[i_from, index[tgt]] += rate
    np.fill_diagonal(Q, -Q.sum(axis=1))

    A = np.vstack([Q.T, np.ones(n_states)])
    rhs = np.zeros(n_states + 1)
    rhs[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    l_ctmc = [
        [sum(pi[index[(ka, kb)]] * ka for ka, kb in states), sum(pi[index[(ka, kb)]] * kb for ka, kb in states)],
        [
            sum(pi[index[(ka, kb)]] * (n_a - ka) for ka, kb in states),
            sum(pi[index[(ka, kb)]] * (n_b - kb) for ka, kb in states),
        ],
    ]

    for r in range(2):
        for i in range(2):
            assert np.isclose(
                res.mean_jobs[r][i], l_ctmc[i][r], rtol=1e-8
            ), f"class {r}, node {i}: MVA {res.mean_jobs[r][i]} vs CTMC {l_ctmc[i][r]}"


def test_fcfs_class_dependent_rates_rejected():
    """BCMP FCFS stations require class-independent service rates."""
    bcmp = BCMPOpenNetworkCalc()
    bcmp.set_sources(arrival_rates=[0.3, 0.3], R=[OPEN_ROUTING, OPEN_ROUTING])
    try:
        bcmp.set_nodes(s=[[0.4, 0.5], [0.25, 0.25]], station_types=["fcfs", "fcfs"])
        raised = False
    except ValueError:
        raised = True
    assert raised


if __name__ == "__main__":
    test_open_single_class_reduces_to_jackson()
    test_open_two_identical_classes_aggregate()
    test_closed_single_chain_reduces_to_mva()
    test_closed_two_class_ps_vs_ctmc()
    test_fcfs_class_dependent_rates_rejected()
