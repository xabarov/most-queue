"""
Test the two-class M/M/1 retrial queue with a priority class: reductions to
the classic retrial queue and to non-preemptive priorities, plus the paired
simulator.
"""

import numpy as np

from most_queue.sim.retrial_priority import MM1RetrialPrioritySim
from most_queue.theory.priority.retrial_priority import MM1RetrialPriorityCalc

GAMMA = 0.7
L = [0.3, 0.35]
MU = [1.2, 1.0]


def test_no_priority_class_reduces_to_classic_retrial():
    """lambda_0 -> 0: the orbit must match the Falin-Templeton closed form
    for the M/M/1 retrial queue."""
    lam, mu = 0.5, 1.0
    calc = MM1RetrialPriorityCalc(gamma=GAMMA)
    calc.set_sources(l=[1e-12, lam])
    calc.set_servers(mu=[1.0, mu])
    calc.run()

    rho = lam / mu
    orbit_exact = rho**2 / (1 - rho) + lam * rho / (GAMMA * (1 - rho))
    assert np.isclose(calc.mean_orbit, orbit_exact, rtol=1e-6)


def test_fast_retrials_reduce_to_np_priority():
    """gamma -> infinity: the orbit behaves as an ordinary queue, so waits
    approach the Cobham non-preemptive two-class formula."""
    calc = MM1RetrialPriorityCalc(gamma=2e4)
    calc.set_sources(l=L)
    calc.set_servers(mu=MU)
    res = calc.run()

    b = [[1 / MU[k], 2 / MU[k] ** 2] for k in range(2)]
    rho_k = [L[k] * b[k][0] for k in range(2)]
    w0 = sum(L[k] * b[k][1] / 2 for k in range(2))
    w_high = w0 / (1 - rho_k[0])
    w_low = w0 / ((1 - rho_k[0]) * (1 - rho_k[0] - rho_k[1]))

    assert np.isclose(res.w[0][0], w_high, rtol=0.01)
    assert np.isclose(res.w[1][0], w_low, rtol=0.01)


def test_vs_simulation():
    """Truncated CTMC against the seeded rate-based simulator."""
    calc = MM1RetrialPriorityCalc(gamma=GAMMA)
    calc.set_sources(l=L)
    calc.set_servers(mu=MU)
    res = calc.run()

    sim = MM1RetrialPrioritySim(gamma=GAMMA, seed=42)
    sim.set_sources(l=L)
    sim.set_servers(mu=MU)
    sim_res = sim.run(800_000)

    assert np.isclose(calc.mean_priority_queue, sim.mean_priority_queue, rtol=0.05)
    assert np.isclose(calc.mean_orbit, sim.mean_orbit, rtol=0.05)
    for k in range(2):
        assert np.isclose(res.v[k][0], sim_res.v[k][0], rtol=0.05), f"class {k}"


if __name__ == "__main__":
    test_no_priority_class_reduces_to_classic_retrial()
    test_fast_retrials_reduce_to_np_priority()
    test_vs_simulation()
