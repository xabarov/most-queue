"""
Testing the Takahashi-Takami method for calculating an H2/H2/n queue.

H2 (hyperexponential) arrival, H2 (hyperexponential) service.
Verification via simulation.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import (
    print_sojourn_moments,
    print_waiting_moments,
    probs_print,
)
from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.params import H2Params
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.hkhk_takahasi import HkHkNCalc

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

# Tolerances for H2/H2/n (both arrival and service non-Markovian)
PROBS_ATOL = 0.02
PROBS_RTOL = 0.08
MOMENTS_ATOL = 3.0
MOMENTS_RTOL = 0.6
NUM_OF_JOBS_H2H2 = 500000

ARRIVAL_CV = float(params["arrival"].get("cv", 1.2))
SERVICE_CV = float(params["service"].get("cv", 1.2))


def _is_simulatable_h2(h2: H2Params) -> bool:
    """QsSim cannot simulate complex H2 parameters (CV<1 complex-fit)."""
    vals = [h2.p1, h2.mu1, h2.mu2]
    if any(abs(np.imag(v)) > 1e-12 for v in vals):
        return False
    # basic sanity for simulation
    p1 = float(np.real(h2.p1))
    mu1 = float(np.real(h2.mu1))
    mu2 = float(np.real(h2.mu2))
    return 0.0 < p1 < 1.0 and mu1 > 0.0 and mu2 > 0.0


def test_h2_h2_n_calc():
    """Test HkHkNCalc against QsSim for H2/H2/n. Same H2 params for both (no refit)."""
    a1 = 1.0 / ARRIVAL_RATE
    h2_arr = H2Distribution.get_params_by_mean_and_cv(a1, ARRIVAL_CV, is_clx=True)
    h2_arr_params = H2Params(p1=h2_arr.p1, mu1=h2_arr.mu1, mu2=h2_arr.mu2)

    b_mean = UTILIZATION_FACTOR * NUM_OF_CHANNELS * a1
    h2_srv = H2Distribution.get_params_by_mean_and_cv(b_mean, SERVICE_CV, is_clx=True)
    h2_srv_params = H2Params(p1=h2_srv.p1, mu1=h2_srv.mu1, mu2=h2_srv.mu2)

    u_arr = [h2_arr_params.p1, 1.0 - h2_arr_params.p1]
    lam_arr = [h2_arr_params.mu1, h2_arr_params.mu2]
    y_srv = [h2_srv_params.p1, 1.0 - h2_srv_params.p1]
    mu_srv = [h2_srv_params.mu1, h2_srv_params.mu2]

    calc = HkHkNCalc(n=NUM_OF_CHANNELS, k=2)
    calc.set_sources(u=u_arr, lam=lam_arr)
    calc.set_servers(y=y_srv, mu=mu_srv)

    calc_results = calc.run()

    # If parameters are truly complex (CV<1), simulation can't be used. Still ensure calc is sane.
    if not (_is_simulatable_h2(h2_arr_params) and _is_simulatable_h2(h2_srv_params)):
        p = np.asarray(calc_results.p, dtype=float)
        assert np.isclose(p.sum(), 1.0, atol=1e-9, rtol=1e-8)
        assert np.all(np.isfinite(p))
        assert np.min(p) >= -1e-12
        return

    h2_arr_real = H2Params(
        p1=float(np.real(h2_arr_params.p1)),
        mu1=float(np.real(h2_arr_params.mu1)),
        mu2=float(np.real(h2_arr_params.mu2)),
    )
    h2_srv_real = H2Params(
        p1=float(np.real(h2_srv_params.p1)),
        mu1=float(np.real(h2_srv_params.mu1)),
        mu2=float(np.real(h2_srv_params.mu2)),
    )
    qs = QsSim(NUM_OF_CHANNELS)
    qs.set_sources(h2_arr_real, "H")
    qs.set_servers(h2_srv_real, "H")

    sim_results = qs.run(NUM_OF_JOBS_H2H2)

    print("\nH2/H2/n: Takahashi-Takami vs simulation")
    print(f"Simulation - H2/H2/{NUM_OF_CHANNELS}")
    print(f"Calculation - H2/H2/{NUM_OF_CHANNELS}")
    print(f"Utilization: {UTILIZATION_FACTOR:.2f}, Arrival CV: {ARRIVAL_CV:.2f}, Service CV: {SERVICE_CV:.2f}")
    print(f"Iterations: {calc.num_of_iter_}")

    probs_print(sim_results.p, calc_results.p, 10)
    print_sojourn_moments(sim_results.v, calc_results.v)
    print_waiting_moments(sim_results.w, calc_results.w)

    assert np.allclose(sim_results.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.w, calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_h2_h2_n_calc()
