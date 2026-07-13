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


def test_multiclass_reduces_to_single_class():
    """R=1 multiclass G-network must equal the single-class solver."""
    from most_queue.theory.networks.g_network import GNetworkMulticlassCalc

    single = GNetworkCalc()
    single.set_sources(positive_rates=EXT_PLUS, P_plus=P_PLUS, P_minus=P_MINUS, negative_rates=EXT_MINUS)
    single.set_nodes(mu=MU)
    res_single = single.run()

    multi = GNetworkMulticlassCalc()
    multi.set_sources(
        positive_rates=[[EXT_PLUS[0]], [EXT_PLUS[1]]],
        P_plus=[P_PLUS],
        P_minus=[P_MINUS],
        negative_rates=[[EXT_MINUS[0]], [EXT_MINUS[1]]],
    )
    multi.set_nodes(mu=[[MU[0]], [MU[1]]])
    res_multi = multi.run()

    assert np.allclose(res_multi.loads, res_single.loads, rtol=1e-10)
    assert np.allclose([res_multi.mean_jobs[0][i] for i in range(2)], res_single.mean_jobs, rtol=1e-10)


def test_multiclass_product_form_satisfies_global_balance():
    """The multinomial-geometric product form with q_ir from the solver must
    satisfy the exact global balance equations of the 2-class PS dynamics
    (class-r signal kills with probability k_ir / k_i) to truncation error."""
    import itertools
    import math

    from most_queue.theory.networks.g_network import GNetworkMulticlassCalc

    mu = [[1.0, 1.5], [2.0, 1.2]]
    ext_p = [[0.25, 0.15], [0.1, 0.2]]
    ext_m = [[0.1, 0.0], [0.0, 0.15]]
    p_plus = [np.array([[0.0, 0.3], [0.2, 0.0]]), np.array([[0.0, 0.4], [0.1, 0.0]])]
    p_minus = [np.array([[0.0, 0.1], [0.0, 0.0]]), np.array([[0.0, 0.0], [0.2, 0.0]])]
    k_max = 9

    calc = GNetworkMulticlassCalc()
    calc.set_sources(positive_rates=ext_p, P_plus=p_plus, P_minus=p_minus, negative_rates=ext_m)
    calc.set_nodes(mu=mu)
    calc.run()
    q = calc.q
    q_i = q.sum(axis=1)

    states = [s for s in itertools.product(range(k_max + 1), repeat=4) if s[0] + s[1] <= k_max and s[2] + s[3] <= k_max]
    idx = {s: n for n, s in enumerate(states)}
    Q = np.zeros((len(states), len(states)))

    def kcount(s, i, r):
        return s[2 * i + r]

    def move(s, i, r, d):
        lst = list(s)
        lst[2 * i + r] += d
        return tuple(lst)

    for s in states:
        n_from = idx[s]

        def add(tgt, rate):
            if rate > 0 and tgt in idx:
                Q[n_from, idx[tgt]] += rate

        def kill(s2, j, r, rate):
            kj = kcount(s2, j, r)
            tot_j = kcount(s2, j, 0) + kcount(s2, j, 1)
            if kj == 0:
                add(s2, rate)
            else:
                p_hit = kj / tot_j
                add(move(s2, j, r, -1), rate * p_hit)
                add(s2, rate * (1.0 - p_hit))

        for i in range(2):
            tot = kcount(s, i, 0) + kcount(s, i, 1)
            for r in range(2):
                add(move(s, i, r, +1), ext_p[i][r])
                kill(s, i, r, ext_m[i][r])
                if kcount(s, i, r) > 0:
                    rate = mu[i][r] * kcount(s, i, r) / tot
                    dep = move(s, i, r, -1)
                    stay = 1.0
                    for j in range(2):
                        add(move(dep, j, r, +1), rate * p_plus[r][i, j])
                        kill(dep, j, r, rate * p_minus[r][i, j])
                        stay -= p_plus[r][i, j] + p_minus[r][i, j]
                    add(dep, rate * stay)
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))

    pi = np.zeros(len(states))
    for s in states:
        val = 1.0
        for i in range(2):
            k0, k1 = s[2 * i], s[2 * i + 1]
            coef = math.factorial(k0 + k1) / (math.factorial(k0) * math.factorial(k1))
            val *= (1 - q_i[i]) * coef * q[i, 0] ** k0 * q[i, 1] ** k1
        pi[idx[s]] = val
    pi /= pi.sum()

    residual = np.max(np.abs(pi @ Q))
    assert residual < 1e-4, f"global balance residual {residual}"
