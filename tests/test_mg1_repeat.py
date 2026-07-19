"""
Test the two-class M/G/1 preemptive-repeat analytics: RS solved exactly (CTMC)
against closed forms and against the existing RS simulator — the first
analytical benchmark for the simulator's RS discipline; Gaver completion-time
means for RS and RW.
"""

import numpy as np
import pytest

from most_queue.random.distributions import Cox2Params, CoxDistribution
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.preemptive.mg1_repeat import (
    MG1PreemptiveRepeatCalc,
    completion_time_mean,
)

L = [0.25, 0.3]
MU = [1.4, 1.0]


def test_rs_with_exponential_service_equals_preemptive_resume():
    """With exponential service, repeat-resampling is stochastically identical
    to preemptive resume: the textbook PR closed forms must hold."""
    calc = MG1PreemptiveRepeatCalc(kind="RS")
    calc.set_sources(l=L)
    calc.set_servers(b=[[1 / MU[k], 2 / MU[k] ** 2, 6 / MU[k] ** 3] for k in range(2)])
    res = calc.run()

    rho = [L[k] / MU[k] for k in range(2)]
    residual = sum(L[k] / MU[k] ** 2 for k in range(2))
    t_high = (1 / MU[0]) / (1 - rho[0])
    t_low = (1 / MU[1]) / (1 - rho[0]) + residual / ((1 - rho[0]) * (1 - rho[0] - rho[1]))

    assert np.isclose(res.v[0][0], t_high, rtol=1e-5)
    assert np.isclose(res.v[1][0], t_low, rtol=1e-5)


def test_completion_time_means():
    """Exponential low service: E[C_RS] = b_L (1 + a E[B]) (memoryless);
    Jensen: repeat-identical is never faster than resampling."""
    b_high = [1 / MU[0], 2 / MU[0] ** 2, 6 / MU[0] ** 3]
    b_low_exp = [1 / MU[1], 2 / MU[1] ** 2, 6 / MU[1] ** 3]

    a = L[0]
    mean_busy = b_high[0] / (1 - a * b_high[0])
    c_rs = completion_time_mean(b_low_exp, a, b_high, "RS")
    assert np.isclose(c_rs, b_low_exp[0] * (1 + a * mean_busy), rtol=1e-9)

    c_rw = completion_time_mean(b_low_exp, a, b_high, "RW")
    assert c_rw >= c_rs - 1e-12


def test_rw_queueing_not_implemented():
    """RW has no finite Markov representation: run() must refuse clearly."""
    calc = MG1PreemptiveRepeatCalc(kind="RW")
    calc.set_sources(l=L)
    calc.set_servers(b=[[1 / MU[k], 2 / MU[k] ** 2, 6 / MU[k] ** 3] for k in range(2)])
    with pytest.raises(NotImplementedError):
        calc.run()
    assert calc.completion_means["RW"] is not None


def test_rs_vs_simulator():
    """RS CTMC against PriorityQueueSimulator's RS discipline — closing the
    'discipline exists only in the simulator' gap. Both sides use the same
    Cox-2 service distributions (repeat dynamics depend on the full
    distribution, not just its moments, so the fit must be exact)."""
    cox = [Cox2Params(p1=0.4, mu1=2.4, mu2=1.1), Cox2Params(p1=0.35, mu1=2.0, mu2=0.9)]
    b = [CoxDistribution.calc_theory_moments(c, 4) for c in cox]

    calc = MG1PreemptiveRepeatCalc(kind="RS")
    calc.set_sources(l=L)
    calc.set_servers(b=b)
    res = calc.run()

    sim = PriorityQueueSimulator(1, 2, "RS")
    sim.set_sources([{"type": "M", "params": L[0]}, {"type": "M", "params": L[1]}])
    sim.set_servers([{"type": "C", "params": cox[0]}, {"type": "C", "params": cox[1]}])
    sim.run(300_000)

    for k in range(2):
        assert np.isclose(res.v[k][0], sim.v[k][0], rtol=0.06), f"class {k}: calc {res.v[k][0]} vs sim {sim.v[k][0]}"


if __name__ == "__main__":
    test_rs_with_exponential_service_equals_preemptive_resume()
    test_completion_time_means()
    test_rw_queueing_not_implemented()
    test_rs_vs_simulator()
