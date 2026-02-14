"""
Cross-check Takahashi-Takami vs simulation at CV=1.05.

H2/H2/3, H2/M/3, M/H2/3 — same H2 params for calc and sim (no refit).
Slow test: 1.5M jobs per scenario. Run with: pytest tests/test_tt_vs_sim_cv105.py -v
"""

import os

import numpy as np
import pytest
import yaml

from most_queue.io.tables import probs_print
from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.params import H2Params
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.gmc_takahasi import H2MnCalc
from most_queue.theory.fifo.hkhk_takahasi import HkHkNCalc
from most_queue.theory.fifo.mgn_takahasi import MGnCalc

cur_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(cur_dir, "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

NUM_OF_CHANNELS = 3
ARRIVAL_RATE = float(params["arrival"]["rate"])
UTILIZATION_FACTOR = 0.7
CV = 1.05
NUM_OF_JOBS = 1_500_000

# Target: |Δp0| ≤ 0.02, p[:10] close
PROBS_ATOL = 0.02
PROBS_RTOL = 0.08


def _h2_real(h2):
    """Convert possibly complex H2Params to real."""
    return H2Params(
        p1=float(np.real(h2.p1)),
        mu1=float(np.real(h2.mu1)),
        mu2=float(np.real(h2.mu2)),
    )


@pytest.mark.slow
def test_h2_h2_cv105():
    """H2/H2/3 at CV=1.05: TT vs QsSim, identical H2 params."""
    a1 = 1.0 / ARRIVAL_RATE
    h2_arr = _h2_real(H2Distribution.get_params_by_mean_and_cv(a1, CV, is_clx=True))
    b_mean = UTILIZATION_FACTOR * NUM_OF_CHANNELS * a1
    h2_srv = _h2_real(H2Distribution.get_params_by_mean_and_cv(b_mean, CV, is_clx=True))

    u_arr = [h2_arr.p1, 1.0 - h2_arr.p1]
    lam_arr = [h2_arr.mu1, h2_arr.mu2]
    y_srv = [h2_srv.p1, 1.0 - h2_srv.p1]
    mu_srv = [h2_srv.mu1, h2_srv.mu2]

    calc = HkHkNCalc(n=NUM_OF_CHANNELS, k=2)
    calc.set_sources(u=u_arr, lam=lam_arr)
    calc.set_servers(y=y_srv, mu=mu_srv)
    calc_results = calc.run()

    qs = QsSim(NUM_OF_CHANNELS)
    qs.set_sources(h2_arr, "H")
    qs.set_servers(h2_srv, "H")
    sim_results = qs.run(NUM_OF_JOBS)

    print("\nH2/H2/3 @ CV=1.05")
    probs_print(sim_results.p, calc_results.p, 10)
    dp0 = abs(calc_results.p[0] - sim_results.p[0])
    assert dp0 <= PROBS_ATOL, f"|Δp0|={dp0:.4f} > {PROBS_ATOL}"
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL)


@pytest.mark.slow
def test_h2_m_cv105():
    """H2/M/3 at CV=1.05: TT vs QsSim, identical H2 arrival params."""
    a1 = 1.0 / ARRIVAL_RATE
    h2_params = _h2_real(H2Distribution.get_params_by_mean_and_cv(a1, CV, is_clx=True))
    b_mean = UTILIZATION_FACTOR * NUM_OF_CHANNELS * a1

    calc = H2MnCalc(n=NUM_OF_CHANNELS)
    calc.set_sources(h2_params)
    calc.set_servers(b_mean)
    calc_results = calc.run()

    qs = QsSim(NUM_OF_CHANNELS)
    qs.set_sources(h2_params, "H")
    qs.set_servers(1.0 / b_mean, "M")
    sim_results = qs.run(NUM_OF_JOBS)

    print("\nH2/M/3 @ CV=1.05")
    probs_print(sim_results.p, calc_results.p, 10)
    dp0 = abs(calc_results.p[0] - sim_results.p[0])
    assert dp0 <= PROBS_ATOL, f"|Δp0|={dp0:.4f} > {PROBS_ATOL}"
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL)


@pytest.mark.slow
def test_m_h2_cv105():
    """M/H2/3 at CV=1.05: TT vs QsSim with H2 service (not Gamma), identical params."""
    a1 = 1.0 / ARRIVAL_RATE
    lam = ARRIVAL_RATE
    b_mean = UTILIZATION_FACTOR * NUM_OF_CHANNELS * a1
    h2_srv = _h2_real(H2Distribution.get_params_by_mean_and_cv(b_mean, CV, is_clx=True))

    calc = MGnCalc(n=NUM_OF_CHANNELS)
    calc.set_sources(lam)
    calc.set_servers(h2_srv)
    calc_results = calc.run()

    qs = QsSim(NUM_OF_CHANNELS)
    qs.set_sources(lam, "M")
    qs.set_servers(h2_srv, "H")
    sim_results = qs.run(NUM_OF_JOBS)

    print("\nM/H2/3 @ CV=1.05")
    probs_print(sim_results.p, calc_results.p, 10)
    dp0 = abs(calc_results.p[0] - sim_results.p[0])
    assert dp0 <= PROBS_ATOL, f"|Δp0|={dp0:.4f} > {PROBS_ATOL}"
    calc_p = np.asarray(calc_results.p, dtype=float)
    assert np.allclose(sim_results.p[:10], calc_p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL)


if __name__ == "__main__":
    # Run directly to see printed tables without pytest (-s).
    test_h2_h2_cv105()
    test_h2_m_cv105()
    test_m_h2_cv105()
