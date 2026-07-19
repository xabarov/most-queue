"""
Test the two-class non-preemptive priority M/M/n + M (Erlang-A with
priorities): reductions to the aggregate Erlang-A and the paired simulator.
"""

import numpy as np

from most_queue.sim.priority_impatience import MMnPriorityImpatienceSim
from most_queue.theory.impatience.mmn import MMnImpatienceCalc
from most_queue.theory.priority.impatience import MMnPriorityImpatienceCalc

N = 3
MU = 1.0
L = [1.2, 1.5]
THETA = [0.4, 0.4]


def test_total_queue_matches_erlang_a():
    """With theta_0 = theta_1 and class-independent mu, the TOTAL queue is an
    Erlang-A queue: E[q0] + E[q1] must match exactly (priority only splits)."""
    calc = MMnPriorityImpatienceCalc(n=N)
    calc.set_sources(l=L)
    calc.set_servers(mu=MU, theta=THETA)
    calc.run()

    erlang_a = MMnImpatienceCalc(n=N, theta=THETA[0])
    erlang_a.set_sources(l=sum(L))
    erlang_a.set_servers(mu=MU)
    erlang_a.run()

    assert np.isclose(sum(calc.mean_queue), erlang_a.get_mean_queue(), rtol=1e-6)


def test_single_class_reduces_to_erlang_a():
    """lambda_1 -> 0: class 0 alone must reproduce Erlang-A."""
    calc = MMnPriorityImpatienceCalc(n=N)
    calc.set_sources(l=[2.4, 1e-12])
    calc.set_servers(mu=MU, theta=[0.5, 0.5])
    calc.run()

    erlang_a = MMnImpatienceCalc(n=N, theta=0.5)
    erlang_a.set_sources(l=2.4)
    erlang_a.set_servers(mu=MU)
    erlang_a.run()

    assert np.isclose(calc.mean_queue[0], erlang_a.get_mean_queue(), rtol=1e-6)
    assert np.isclose(calc.abandon_probs[0], erlang_a.get_abandonment_probability(), rtol=1e-6)


def test_priority_reduces_high_class_wait():
    """The priority class must wait less; abandonment follows the same order."""
    calc = MMnPriorityImpatienceCalc(n=N)
    calc.set_sources(l=L)
    calc.set_servers(mu=MU, theta=[0.3, 0.6])
    res = calc.run()

    assert res.w[0][0] < res.w[1][0]
    assert calc.abandon_probs[0] < calc.abandon_probs[1]


def test_vs_simulation():
    """Truncated CTMC against the seeded rate-based simulator."""
    theta = [0.3, 0.7]
    calc = MMnPriorityImpatienceCalc(n=N)
    calc.set_sources(l=L)
    calc.set_servers(mu=MU, theta=theta)
    res = calc.run()

    sim = MMnPriorityImpatienceSim(n=N, seed=42)
    sim.set_sources(l=L)
    sim.set_servers(mu=MU, theta=theta)
    sim_res = sim.run(800_000)

    for k in range(2):
        assert np.isclose(res.w[k][0], sim_res.w[k][0], rtol=0.05), f"class {k}"
        assert np.isclose(calc.abandon_probs[k], sim.abandon_probs[k], rtol=0.06), f"abandon {k}"


if __name__ == "__main__":
    test_total_queue_matches_erlang_a()
    test_single_class_reduces_to_erlang_a()
    test_priority_reduces_high_class_wait()
    test_vs_simulation()
