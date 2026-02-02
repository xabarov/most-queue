"""
Testing the Takahashi-Takami method for calculating an H2/M/n queue.

H2 (hyperexponential) arrival, M (exponential) service.
Verification via simulation:

- For **CV >= 1** we use **H2/M/n simulation** with the same (real) H2 parameters.
- For **CV < 1** real H2 does not exist; TT calculations can still use **complex-fit H2**,
  but `QsSim` cannot simulate complex H2 params. In this case we validate using **Gamma/M/n simulation**
  with the same mean and CV as a proxy (looser tolerances).

Tip: to see printed tables under pytest, run with `-s`.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import (
    print_sojourn_moments,
    print_waiting_moments,
    probs_print,
)
from most_queue.random.distributions import GammaDistribution, H2Distribution
from most_queue.random.utils.params import H2Params
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.gmc_takahasi import H2MnCalc

cur_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(cur_dir, "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

NUM_OF_CHANNELS = int(params["num_of_channels"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]
MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

# Tolerances for H2/M/n
PROBS_ATOL = 0.02
PROBS_RTOL = 0.08
NUM_OF_JOBS_H2M = 500000

ARRIVAL_CV_GE1 = float(params["arrival"].get("cv_ge1", 1.2))
ARRIVAL_CV_LT1 = float(params["arrival"].get("cv_lt1", 0.8))

# Proxy tolerances (Gamma simulation vs complex-fit H2 TT) for CV<1
PROXY_PROBS_ATOL = 0.06
PROXY_PROBS_RTOL = 0.20


def _is_simulatable_h2(h2: H2Params) -> bool:
    """QsSim cannot simulate complex H2 parameters (CV<1 complex-fit)."""
    vals = [h2.p1, h2.mu1, h2.mu2]
    if any(abs(np.imag(v)) > 1e-12 for v in vals):
        return False
    p1 = float(np.real(h2.p1))
    mu1 = float(np.real(h2.mu1))
    mu2 = float(np.real(h2.mu2))
    return 0.0 < p1 < 1.0 and mu1 > 0.0 and mu2 > 0.0


def _as_real(h2: H2Params) -> H2Params:
    """Convert possibly complex H2Params to real numbers (for simulation)."""
    return H2Params(
        p1=float(np.real(h2.p1)),
        mu1=float(np.real(h2.mu1)),
        mu2=float(np.real(h2.mu2)),
    )


def _run_case(arrival_cv: float):
    a1 = 1.0 / ARRIVAL_RATE
    h2_raw = H2Distribution.get_params_by_mean_and_cv(a1, arrival_cv, is_clx=True)
    h2_params = H2Params(p1=h2_raw.p1, mu1=h2_raw.mu1, mu2=h2_raw.mu2)

    b_mean = UTILIZATION_FACTOR * NUM_OF_CHANNELS * a1

    calc = H2MnCalc(n=NUM_OF_CHANNELS)
    calc.set_sources(h2_params)
    calc.set_servers(b_mean)

    calc_results = calc.run()

    simulatable = _is_simulatable_h2(h2_params)

    qs = QsSim(NUM_OF_CHANNELS)
    if simulatable:
        qs.set_sources(_as_real(h2_params), "H")
    else:
        qs.set_sources(GammaDistribution.get_params_by_mean_and_cv(a1, arrival_cv), "Gamma")
    qs.set_servers(1.0 / b_mean, "M")  # rate for M

    sim_results = qs.run(NUM_OF_JOBS_H2M)

    mode = "H2/M (same real params)" if simulatable else "proxy (Gamma arrivals for CV<1)"
    print("\nH2/M/n: Takahashi–Takami validation")
    print(f"n={NUM_OF_CHANNELS}, rho={UTILIZATION_FACTOR:.2f}, arrival CV={arrival_cv:.2f}")
    print(f"mode: {mode}, TT iterations: {calc.num_of_iter_}")

    probs_print(sim_results.p, calc_results.p, 10)
    print_sojourn_moments(sim_results.v, calc_results.v)
    print_waiting_moments(sim_results.w, calc_results.w)

    # Always check calc sanity
    p = np.asarray(calc_results.p, dtype=float)
    assert np.isclose(p.sum(), 1.0, atol=1e-9, rtol=1e-8)
    assert np.all(np.isfinite(p))
    assert np.min(p) >= -1e-12

    if not simulatable:
        assert np.allclose(
            sim_results.p[:10], calc_results.p[:10], atol=PROXY_PROBS_ATOL, rtol=PROXY_PROBS_RTOL
        ), ERROR_MSG
        return

    assert np.allclose(sim_results.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.w, calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


def test_h2_m_n_calc_cv_ge1():
    """CV>=1: real H2, same H2 params for calc and sim (no refit)."""
    _run_case(ARRIVAL_CV_GE1)


def test_h2_m_n_calc_cv_lt1():
    """CV<1: complex-fit H2 in TT, Gamma simulation proxy with same mean/CV."""
    _run_case(ARRIVAL_CV_LT1)


if __name__ == "__main__":
    # Run directly to see printed tables without pytest (-s).
    test_h2_m_n_calc_cv_ge1()
    test_h2_m_n_calc_cv_lt1()
