"""
Test the MMAP[2]/PH[2]/1 priority queue: exact Poisson/exponential reductions
(Cobham for NP, preemptive-resume solver for PR) and a correlated-arrivals
case against the priority simulator (superposition of two independent MAPs).
"""

import numpy as np

from most_queue.random.distributions import Cox2Params
from most_queue.random.map_ph import MAPParams
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.map_ph_priority import MapPh1PriorityCalc

L = [0.25, 0.35]
MU = [1.4, 1.0]


def _poisson_exp(discipline):
    calc = MapPh1PriorityCalc(discipline=discipline)
    calc.set_sources(
        D0=[[-(L[0] + L[1])]],
        D1_high=[[L[0]]],
        D1_low=[[L[1]]],
    )
    calc.set_servers(
        ph_high=([1.0], [[-MU[0]]]),
        ph_low=([1.0], [[-MU[1]]]),
    )
    return calc.run()


def test_np_poisson_exp_reduces_to_cobham():
    """One-phase MMAP + exponential PH, NP: exact Cobham waits."""
    res = _poisson_exp("NP")

    b = [[1 / MU[k], 2 / MU[k] ** 2] for k in range(2)]
    rho = [L[k] * b[k][0] for k in range(2)]
    w0 = sum(L[k] * b[k][1] / 2 for k in range(2))
    w_high = w0 / (1 - rho[0])
    w_low = w0 / ((1 - rho[0]) * (1 - rho[0] - rho[1]))

    assert np.isclose(res.w[0][0], w_high, rtol=1e-6)
    assert np.isclose(res.w[1][0], w_low, rtol=1e-6)


def test_pr_poisson_exp_reduces_to_closed_form():
    """One-phase MMAP + exponential PH, PR: matches the textbook two-class
    preemptive-resume closed forms
        T_H = E[S_H] / (1 - rho_H),
        T_L = E[S_L]/(1 - rho_H) + R / ((1 - rho_H)(1 - rho_H - rho_L)),
    R = sum lambda_k E[S_k^2]/2. (The library's MG1PreemptiveCalc returns a
    different low-class value on this case — flagged in EPIC-020 results.)"""
    res = _poisson_exp("PR")

    rho = [L[k] / MU[k] for k in range(2)]
    residual = sum(L[k] / MU[k] ** 2 for k in range(2))
    t_high = (1 / MU[0]) / (1 - rho[0])
    t_low = (1 / MU[1]) / (1 - rho[0]) + residual / ((1 - rho[0]) * (1 - rho[0] - rho[1]))

    assert np.isclose(res.v[0][0], t_high, rtol=1e-6)
    assert np.isclose(res.v[1][0], t_low, rtol=1e-6)


def _mmap_superposition():
    """Two independent MMPP-2 flows as one MMAP on the product phase space
    (moderate load so the CTMC truncation stays small)."""
    map_h = MAPParams(D0=np.array([[-0.8, 0.1], [0.2, -0.38]]), D1=np.array([[0.7, 0.0], [0.0, 0.18]]))
    map_l = MAPParams(D0=np.array([[-0.55, 0.2], [0.1, -0.28]]), D1=np.array([[0.35, 0.0], [0.0, 0.18]]))
    eye_h = np.eye(2)
    d0 = np.kron(map_h.D0, np.eye(2)) + np.kron(eye_h, map_l.D0)
    d1_high = np.kron(map_h.D1, np.eye(2))
    d1_low = np.kron(eye_h, map_l.D1)
    return map_h, map_l, d0, d1_high, d1_low


def test_correlated_arrivals_vs_simulation():
    """MMAP (superposition of independent MAPs) + Cox-2 service, NP: the CTMC
    against the priority simulator fed the same two independent MAP sources."""
    map_h, map_l, d0, d1_high, d1_low = _mmap_superposition()
    cox = [Cox2Params(p1=0.4, mu1=2.4, mu2=1.1), Cox2Params(p1=0.35, mu1=2.0, mu2=0.9)]

    def cox_to_ph(c):
        return ([1.0, 0.0], [[-c.mu1, c.p1 * c.mu1], [0.0, -c.mu2]])

    calc = MapPh1PriorityCalc(discipline="NP")
    calc.set_sources(D0=d0, D1_high=d1_high, D1_low=d1_low)
    calc.set_servers(ph_high=cox_to_ph(cox[0]), ph_low=cox_to_ph(cox[1]))
    # fixed truncation: q=60 agrees with q=100 to 4 decimals at this load
    res = calc.run(q_start=60, q_max=60)

    sim = PriorityQueueSimulator(1, 2, "NP")
    sim.set_sources([{"type": "MAP", "params": map_h}, {"type": "MAP", "params": map_l}])
    sim.set_servers([{"type": "C", "params": cox[0]}, {"type": "C", "params": cox[1]}])
    sim.run(300_000)

    for k in range(2):
        assert np.isclose(res.v[k][0], sim.v[k][0], rtol=0.07), f"class {k}: calc {res.v[k][0]} vs sim {sim.v[k][0]}"


if __name__ == "__main__":
    test_np_poisson_exp_reduces_to_cobham()
    test_pr_poisson_exp_reduces_to_closed_form()
    test_correlated_arrivals_vs_simulation()
