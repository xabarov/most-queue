"""
Test the M/G/1 accumulating priority queue (Kleinrock delay-dependent /
Stanford-Taylor-Ziedins APQ): limits to FIFO and to classic non-preemptive
priorities, the conservation law, and the paired simulator.
"""

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.sim.accumulating_priority import AccumulatingPrioritySim
from most_queue.theory.priority.accumulating import MG1AccumulatingPriorityCalc
from most_queue.theory.priority.non_preemptive.mg1 import MG1NonPreemptiveCalc

ARRIVAL_RATES = [0.2, 0.3, 0.25]
SERVICE_CV = 1.3


def _service_moments():
    b, serv_params = [], []
    for mean in (0.4, 0.5, 0.6):
        params = H2Distribution.get_params_by_mean_and_cv(mean, SERVICE_CV)
        serv_params.append({"type": "H", "params": params})
        b.append(H2Distribution.calc_theory_moments(params, 4))
    return b, serv_params


def _run_apq(rates, b):
    calc = MG1AccumulatingPriorityCalc()
    calc.set_sources(l=ARRIVAL_RATES)
    calc.set_servers(b=b, rates=rates)
    return calc.run()


def test_equal_rates_reduce_to_fifo():
    """Equal accumulation rates: every class waits the M/G/1 FIFO wait."""
    b, _ = _service_moments()
    res = _run_apq([1.0, 1.0, 1.0], b)

    rho = sum(ARRIVAL_RATES[k] * b[k][0] for k in range(3))
    w0 = sum(ARRIVAL_RATES[k] * b[k][1] / 2 for k in range(3))
    w_fifo = w0 / (1 - rho)
    assert np.allclose([res.w[k][0] for k in range(3)], w_fifo, rtol=1e-12)


def test_extreme_rates_reduce_to_np_priority():
    """Rate ratios -> infinity reproduce the classic Cobham NP waits (exact
    formula; the library's MG1NonPreemptiveCalc agrees within its own ~1e-4
    R-factor accuracy)."""
    b, _ = _service_moments()
    res = _run_apq([1.0, 1e-10, 1e-20], b)

    rho_k = [ARRIVAL_RATES[k] * b[k][0] for k in range(3)]
    w0 = sum(ARRIVAL_RATES[k] * b[k][1] / 2 for k in range(3))
    sigma = np.cumsum(rho_k)
    cobham = [w0 / ((1 - (sigma[k - 1] if k > 0 else 0.0)) * (1 - sigma[k])) for k in range(3)]
    for k in range(3):
        assert np.isclose(res.w[k][0], cobham[k], rtol=1e-8), f"class {k}"

    np_calc = MG1NonPreemptiveCalc()
    np_calc.set_sources(ARRIVAL_RATES)
    np_calc.set_servers(b)
    np_res = np_calc.run()
    for k in range(3):
        assert np.isclose(res.w[k][0], np_res.w[k][0], rtol=1e-3), f"class {k} vs library NP"


def test_conservation_law():
    """sum rho_k W_k must equal rho * W0 / (1 - rho) for any rates."""
    b, _ = _service_moments()
    res = _run_apq([5.0, 2.0, 1.0], b)

    rho_k = [ARRIVAL_RATES[k] * b[k][0] for k in range(3)]
    rho = sum(rho_k)
    w0 = sum(ARRIVAL_RATES[k] * b[k][1] / 2 for k in range(3))
    lhs = sum(rho_k[k] * res.w[k][0] for k in range(3))
    assert np.isclose(lhs, rho * w0 / (1 - rho), rtol=1e-12)


def test_apq_vs_simulation():
    """Kleinrock recursion against the seeded APQ simulator."""
    b, serv_params = _service_moments()
    rates = [4.0, 2.0, 1.0]
    res = _run_apq(rates, b)

    sim = AccumulatingPrioritySim(seed=42)
    sim.set_sources(l=ARRIVAL_RATES)
    sim.set_servers(serv_params=serv_params, rates=rates)
    sim_res = sim.run(400_000)

    for k in range(3):
        assert np.isclose(
            res.w[k][0], sim_res.w[k][0], rtol=0.05
        ), f"class {k}: calc {res.w[k][0]} vs sim {sim_res.w[k][0]}"


if __name__ == "__main__":
    test_equal_rates_reduce_to_fifo()
    test_extreme_rates_reduce_to_np_priority()
    test_conservation_law()
    test_apq_vs_simulation()
