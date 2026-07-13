"""
Test the tandem with finite buffers and blocking after service (BAS):
decomposition vs exact CTMC (2 nodes), vs simulation (3 nodes), and the
infinite-buffer limit vs the Jackson tandem.
"""

import numpy as np

from most_queue.sim.networks.tandem_blocking import TandemBlockingSim
from most_queue.theory.networks.blocking import TandemBlockingCalc, mm1k_probs
from most_queue.theory.networks.jackson_network import JacksonNetworkCalc

ARRIVAL_RATE = 0.8
MU = [1.0, 1.2]
CAPACITY = [4, 3]


def _ctmc_two_node(lam, mu, cap):
    """
    Exact CTMC of a 2-node BAS tandem. State (n1, n2, b): jobs at node 1
    (including a blocked one), jobs at node 2, b — node 1 server holds a
    completed job (possible only when n2 == K2).
    """
    k1, k2 = cap
    states = []
    for n1 in range(k1 + 1):
        for n2 in range(k2 + 1):
            states.append((n1, n2, 0))
            if n1 >= 1 and n2 == k2:
                states.append((n1, n2, 1))
    index = {s: i for i, s in enumerate(states)}
    Q = np.zeros((len(states), len(states)))

    for n1, n2, b in states:
        i = index[(n1, n2, b)]
        # External arrival (lost if node 1 full)
        if n1 < k1:
            Q[i, index[(n1 + 1, n2, b)]] += lam
        # Node 1 completion (server not blocked)
        if n1 >= 1 and b == 0:
            if n2 < k2:
                Q[i, index[(n1 - 1, n2 + 1, 0)]] += mu[0]
            else:
                Q[i, index[(n1, n2, 1)]] += mu[0]
        # Node 2 completion
        if n2 >= 1:
            if b == 1:  # blocked job enters node 2 immediately
                Q[i, index[(n1 - 1, n2, 0)]] += mu[1]
            else:
                Q[i, index[(n1, n2 - 1, 0)]] += mu[1]
    np.fill_diagonal(Q, -Q.sum(axis=1))

    A = np.vstack([Q.T, np.ones(len(states))])
    rhs = np.zeros(len(states) + 1)
    rhs[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    throughput = sum(pi[index[s]] * mu[1] for s in states if s[1] >= 1)
    l1 = sum(pi[index[s]] * s[0] for s in states)
    l2 = sum(pi[index[s]] * s[1] for s in states)
    loss = sum(pi[index[s]] for s in states if s[0] == cap[0])
    return throughput, [l1, l2], loss


def test_decomposition_vs_exact_ctmc():
    """Throughput and mean queue lengths within a few percent of the CTMC."""
    calc = TandemBlockingCalc()
    calc.set_sources(arrival_rate=ARRIVAL_RATE)
    calc.set_nodes(mu=MU, capacity=CAPACITY)
    res = calc.run()

    x_exact, l_exact, loss_exact = _ctmc_two_node(ARRIVAL_RATE, MU, CAPACITY)

    assert np.isclose(calc.throughput, x_exact, rtol=0.03), f"{calc.throughput} vs exact {x_exact}"
    assert np.allclose(res.mean_jobs, l_exact, rtol=0.15, atol=0.1)
    assert np.isclose(calc.loss_prob, loss_exact, rtol=0.25, atol=0.02)


def test_decomposition_vs_simulation_three_nodes():
    """3-node line with a tight middle buffer against the seeded simulator."""
    mu = [1.0, 0.9, 1.1]
    capacity = [5, 2, 3]

    calc = TandemBlockingCalc()
    calc.set_sources(arrival_rate=0.7)
    calc.set_nodes(mu=mu, capacity=capacity)
    calc.run()

    sim = TandemBlockingSim(seed=42)
    sim.set_sources(arrival_rate=0.7)
    sim.set_nodes(serv_params=[{"type": "M", "params": m} for m in mu], capacity=capacity)
    sim.run(200_000)

    assert np.isclose(calc.throughput, sim.throughput, rtol=0.03)
    assert np.isclose(calc.loss_prob, sim.loss_prob, rtol=0.3, atol=0.02)


def test_infinite_buffers_reduce_to_jackson():
    """capacity=None everywhere must reproduce the Jackson tandem."""
    calc = TandemBlockingCalc()
    calc.set_sources(arrival_rate=ARRIVAL_RATE)
    calc.set_nodes(mu=MU, capacity=[None, None])
    res = calc.run()

    routing = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    jackson = JacksonNetworkCalc()
    jackson.set_sources(arrival_rate=ARRIVAL_RATE, R=routing)
    jackson.set_nodes(mu=MU, n=[1, 1])
    jackson_res = jackson.run()

    assert np.isclose(calc.throughput, ARRIVAL_RATE, rtol=1e-9)
    assert calc.loss_prob == 0.0
    assert np.allclose(res.mean_jobs, jackson_res.mean_jobs, rtol=1e-9)
    assert np.isclose(res.v[0], jackson_res.v[0], rtol=1e-9)


def test_mm1k_probs_sanity():
    """M/M/1/K marginal: rho -> 0 concentrates at 0; sums to 1."""
    probs = mm1k_probs(0.5, 1.0, 5)
    assert np.isclose(probs.sum(), 1.0)
    assert probs[0] > probs[-1]


if __name__ == "__main__":
    test_decomposition_vs_exact_ctmc()
    test_decomposition_vs_simulation_three_nodes()
    test_infinite_buffers_reduce_to_jackson()
    test_mm1k_probs_sanity()
