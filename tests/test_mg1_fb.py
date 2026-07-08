"""
Tests for the M/G/1 FB (Foreground-Background / LAS) calculator
against a dedicated batch-sharing simulator.
"""

import os

import numpy as np
import yaml

from most_queue.random.distributions import GammaDistribution
from most_queue.sim.single_server_disciplines import FBSim
from most_queue.theory.fifo.mg1_ps import MG1PSCalc
from most_queue.theory.srpt import MG1FbCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_CV = float(params["service"]["cv"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

SERVICE_MEAN = UTILIZATION_FACTOR / ARRIVAL_RATE


def test_mg1_fb_vs_sim():
    """
    FB calculator vs batch-sharing simulation (Gamma service, CV > 1).
    """
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(SERVICE_MEAN, SERVICE_CV)

    calc = MG1FbCalc()
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(gamma_params, "Gamma")
    calc_results = calc.run()

    sim = FBSim()
    sim.set_sources(ARRIVAL_RATE, "M")
    sim.set_servers(gamma_params, "Gamma")
    sim_results = sim.run(NUM_OF_JOBS)

    print(f"FB: v1 sim={sim_results.v[0]:.4f}, calc={calc_results.v[0]:.4f}")
    assert np.isclose(sim_results.v[0], calc_results.v[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.isclose(sim_results.w[0], calc_results.w[0], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


def test_mg1_fb_exponential_equals_ps():
    """
    For exponential service FB and PS have the same mean sojourn time
    b1 / (1 - rho) — the exponential distribution is the boundary case
    between DHR (FB wins) and IHR (PS wins).
    """
    mu = 1.0 / SERVICE_MEAN

    fb = MG1FbCalc()
    fb.set_sources(ARRIVAL_RATE)
    fb.set_servers(mu, "M")
    fb_results = fb.run()

    expected = SERVICE_MEAN / (1.0 - UTILIZATION_FACTOR)
    print(f"FB (exp service): v1={fb_results.v[0]:.5f}, PS/FCFS mean={expected:.5f}")
    assert np.isclose(fb_results.v[0], expected, rtol=1e-3), ERROR_MSG


def test_mg1_fb_beats_ps_for_dhr():
    """
    For Gamma service with CV > 1 (decreasing hazard rate) FB gives a smaller
    mean sojourn time than PS.
    """
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(SERVICE_MEAN, SERVICE_CV)
    b = GammaDistribution.calc_theory_moments(gamma_params, 4)

    fb = MG1FbCalc()
    fb.set_sources(ARRIVAL_RATE)
    fb.set_servers(gamma_params, "Gamma")
    fb_results = fb.run()

    ps = MG1PSCalc()
    ps.set_sources(l=ARRIVAL_RATE)
    ps.set_servers(b)
    ps_results = ps.run()

    print(f"DHR service: FB v1={fb_results.v[0]:.4f} < PS v1={ps_results.v[0]:.4f}")
    assert fb_results.v[0] < ps_results.v[0], ERROR_MSG


if __name__ == "__main__":
    test_mg1_fb_vs_sim()
    test_mg1_fb_exponential_equals_ps()
    test_mg1_fb_beats_ps_for_dhr()
