"""
Test the Gelenbe G-network product-form solver: against a numerically solved
truncated CTMC (exact up to truncation error) and against the Jackson network
in the no-negatives special case.
"""

import numpy as np

from most_queue.theory.networks.g_network import GNetworkCalc
from most_queue.theory.networks.jackson_network import JacksonNetworkCalc

MU = [1.0, 1.5]
EXT_PLUS = [0.5, 0.2]
EXT_MINUS = [0.1, 0.0]
P_PLUS = np.array([[0.0, 0.4], [0.2, 0.0]])
P_MINUS = np.array([[0.0, 0.2], [0.1, 0.0]])


def _ctmc_mean_jobs(k_max: int = 60) -> list[float]:
    """
    Solve the exact global balance equations of the 2-node G-network on the
    truncated state space {0..k_max}^2.
    """
    m = k_max + 1
    n_states = m * m
    idx = lambda k1, k2: k1 * m + k2  # noqa: E731

    Q = np.zeros((n_states, n_states))

    def add(k1, k2, l1, l2, rate):
        if 0 <= l1 <= k_max and 0 <= l2 <= k_max and rate > 0:
            Q[idx(k1, k2), idx(l1, l2)] += rate

    for k1 in range(m):
        for k2 in range(m):
            k = [k1, k2]
            # External positive/negative arrivals
            add(k1, k2, k1 + 1, k2, EXT_PLUS[0])
            add(k1, k2, k1, k2 + 1, EXT_PLUS[1])
            if k1 > 0:
                add(k1, k2, k1 - 1, k2, EXT_MINUS[0])
            if k2 > 0:
                add(k1, k2, k1, k2 - 1, EXT_MINUS[1])
            # Service completions with movement / signals / exit
            for i in range(2):
                if k[i] == 0:
                    continue
                for j in range(2):
                    dep = [k1, k2]
                    dep[i] -= 1
                    # as positive customer to j
                    tgt = dep.copy()
                    tgt[j] += 1
                    add(k1, k2, tgt[0], tgt[1], MU[i] * P_PLUS[i, j])
                    # as negative signal to j (removes one if non-empty)
                    tgt = dep.copy()
                    if tgt[j] > 0:
                        tgt[j] -= 1
                    add(k1, k2, tgt[0], tgt[1], MU[i] * P_MINUS[i, j])
                # exit from the network
                d_i = 1.0 - P_PLUS[i].sum() - P_MINUS[i].sum()
                dep = [k1, k2]
                dep[i] -= 1
                add(k1, k2, dep[0], dep[1], MU[i] * d_i)

    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))

    # pi Q = 0, sum pi = 1
    A = np.vstack([Q.T, np.ones(n_states)])
    rhs = np.zeros(n_states + 1)
    rhs[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    l1 = sum(k1 * pi[idx(k1, k2)] for k1 in range(m) for k2 in range(m))
    l2 = sum(k2 * pi[idx(k1, k2)] for k1 in range(m) for k2 in range(m))
    return [float(l1), float(l2)]


def test_g_network_vs_exact_ctmc():
    """Product form must match the numerically solved CTMC."""
    calc = GNetworkCalc()
    calc.set_sources(positive_rates=EXT_PLUS, P_plus=P_PLUS, P_minus=P_MINUS, negative_rates=EXT_MINUS)
    calc.set_nodes(mu=MU)
    res = calc.run()

    ctmc_l = _ctmc_mean_jobs()
    assert np.allclose(res.mean_jobs, ctmc_l, rtol=1e-6), f"{res.mean_jobs} vs CTMC {ctmc_l}"


def test_g_network_reduces_to_jackson():
    """With no negative customers the G-network is an open Jackson network."""
    calc = GNetworkCalc()
    calc.set_sources(positive_rates=EXT_PLUS, P_plus=P_PLUS)
    calc.set_nodes(mu=MU)
    res = calc.run()

    # Same network in OpenNetworkCalc routing format: source row + exit column
    total = sum(EXT_PLUS)
    routing = np.zeros((3, 3))
    routing[0, :2] = np.array(EXT_PLUS) / total
    routing[1:, :2] = P_PLUS
    routing[1, 2] = 1.0 - P_PLUS[0].sum()
    routing[2, 2] = 1.0 - P_PLUS[1].sum()

    jackson = JacksonNetworkCalc()
    jackson.set_sources(arrival_rate=total, R=routing)
    jackson.set_nodes(mu=MU, n=[1, 1])
    jackson_res = jackson.run()

    assert np.allclose(res.loads, jackson_res.loads, rtol=1e-10)
    assert np.allclose(res.mean_jobs, jackson_res.mean_jobs, rtol=1e-10)
    assert np.isclose(res.v[0], jackson_res.v[0], rtol=1e-10)


def test_negatives_reduce_load():
    """Increasing the negative arrival rate must strictly decrease loads."""
    loads = []
    for neg in (0.0, 0.3, 0.8):
        calc = GNetworkCalc()
        calc.set_sources(
            positive_rates=EXT_PLUS,
            P_plus=P_PLUS,
            negative_rates=[neg, neg],
        )
        calc.set_nodes(mu=MU)
        res = calc.run()
        loads.append(sum(res.loads))

    assert loads[0] > loads[1] > loads[2]


if __name__ == "__main__":
    test_g_network_vs_exact_ctmc()
    test_g_network_reduces_to_jackson()
    test_negatives_reduce_load()
