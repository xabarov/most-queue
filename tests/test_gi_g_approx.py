"""
Tests for two-moment GI/G/1 and GI/G/m approximations
(Kingman, Kraemer-Langenbach-Belz, Allen-Cunneen).
"""

import os

import numpy as np
import yaml

from most_queue.random.distributions import ExpDistribution, GammaDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.erlang import ErlangCCalc
from most_queue.theory.fifo.gi_g_approx import GIG1ApproxCalc, GIGmApproxCalc, kingman_bound_w1
from most_queue.theory.fifo.mg1 import MG1Calc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

NUM_OF_CHANNELS = int(params["num_of_channels"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

# approximation-specific tolerance for the mean waiting time (KLB is typically
# within a few percent; Allen-Cunneen is cruder)
KLB_W1_RTOL = 0.15
AC_W1_RTOL = 0.25


def _gamma_moments(mean: float, cv: float, num: int = 4) -> list[float]:
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean, cv)
    return GammaDistribution.calc_theory_moments(gamma_params, num)


def test_klb_reduces_to_pollaczek_khinchine():
    """
    For M/G/1 (exponential arrivals, cv2_a = 1) the KLB correction factor is 1 and
    the approximation coincides with the exact Pollaczek-Khinchine mean wait.
    """
    a = ExpDistribution.calc_theory_moments(ARRIVAL_RATE, 4)
    b = _gamma_moments(UTILIZATION_FACTOR / ARRIVAL_RATE, 1.2)

    approx = GIG1ApproxCalc()
    approx.set_sources(a)
    approx.set_servers(b)
    approx_results = approx.run()

    exact = MG1Calc()
    exact.set_sources(l=ARRIVAL_RATE)
    exact.set_servers(b)
    exact_results = exact.run()

    print(f"KLB w1={approx_results.w[0]:.5f}, P-K w1={exact_results.w[0]:.5f}")
    assert np.isclose(approx_results.w[0], exact_results.w[0], rtol=1e-8), ERROR_MSG
    assert np.isclose(approx_results.v[0], exact_results.v[0], rtol=1e-8), ERROR_MSG


def test_allen_cunneen_reduces_to_erlang_c():
    """
    For M/M/n (cv2_a = cv2_b = 1) the Allen-Cunneen formula is exact and
    coincides with the Erlang C mean wait.
    """
    service_rate = ARRIVAL_RATE / (UTILIZATION_FACTOR * NUM_OF_CHANNELS)
    a = ExpDistribution.calc_theory_moments(ARRIVAL_RATE, 4)
    b = ExpDistribution.calc_theory_moments(service_rate, 4)

    approx = GIGmApproxCalc(n=NUM_OF_CHANNELS)
    approx.set_sources(a)
    approx.set_servers(b)
    approx_results = approx.run()

    exact = ErlangCCalc(n=NUM_OF_CHANNELS)
    exact.set_sources(l=ARRIVAL_RATE)
    exact.set_servers(mu=service_rate)
    exact_results = exact.run()

    print(f"Allen-Cunneen w1={approx_results.w[0]:.5f}, Erlang C w1={exact_results.w[0]:.5f}")
    assert np.isclose(approx_results.w[0], exact_results.w[0], rtol=1e-8), ERROR_MSG


def test_gig1_klb_vs_sim_grid():
    """
    KLB approximation vs simulation on a small (cv_a, cv_b, rho) grid,
    including the Kingman upper-bound property.
    """
    grid = [
        (0.56, 1.2, 0.7),
        (1.2, 0.8, 0.7),
        (1.2, 1.2, 0.9),
    ]
    for cv_a, cv_b, rho in grid:
        a = _gamma_moments(1.0 / ARRIVAL_RATE, cv_a)
        b = _gamma_moments(rho / ARRIVAL_RATE, cv_b)

        approx = GIG1ApproxCalc()
        approx.set_sources(a)
        approx.set_servers(b)
        w1_klb = approx.run().w[0]

        qs = QsSim(1, seed=42)
        qs.set_sources(GammaDistribution.get_params_by_mean_and_cv(1.0 / ARRIVAL_RATE, cv_a), "Gamma")
        qs.set_servers(GammaDistribution.get_params_by_mean_and_cv(rho / ARRIVAL_RATE, cv_b), "Gamma")
        w1_sim = qs.run(NUM_OF_JOBS).w[0]

        w1_kingman = kingman_bound_w1(a, b)

        print(f"cv_a={cv_a}, cv_b={cv_b}, rho={rho}: sim={w1_sim:.4f}, klb={w1_klb:.4f}, kingman={w1_kingman:.4f}")
        assert np.isclose(w1_klb, w1_sim, rtol=KLB_W1_RTOL), ERROR_MSG
        # Kingman is an upper bound (allow small simulation noise)
        assert w1_kingman >= w1_sim * 0.97, ERROR_MSG


def test_gigm_allen_cunneen_vs_sim():
    """
    Allen-Cunneen GI/G/m approximation vs simulation (Gamma/Gamma).
    """
    cv_a, cv_b = 1.2, 0.8
    service_mean = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE
    a = _gamma_moments(1.0 / ARRIVAL_RATE, cv_a)
    b = _gamma_moments(service_mean, cv_b)

    approx = GIGmApproxCalc(n=NUM_OF_CHANNELS)
    approx.set_sources(a)
    approx.set_servers(b)
    w1_ac = approx.run().w[0]

    qs = QsSim(NUM_OF_CHANNELS, seed=42)
    qs.set_sources(GammaDistribution.get_params_by_mean_and_cv(1.0 / ARRIVAL_RATE, cv_a), "Gamma")
    qs.set_servers(GammaDistribution.get_params_by_mean_and_cv(service_mean, cv_b), "Gamma")
    w1_sim = qs.run(NUM_OF_JOBS).w[0]

    print(f"GI/G/{NUM_OF_CHANNELS}: sim={w1_sim:.4f}, allen-cunneen={w1_ac:.4f}")
    assert np.isclose(w1_ac, w1_sim, rtol=AC_W1_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_klb_reduces_to_pollaczek_khinchine()
    test_allen_cunneen_reduces_to_erlang_c()
    test_gig1_klb_vs_sim_grid()
    test_gigm_allen_cunneen_vs_sim()
