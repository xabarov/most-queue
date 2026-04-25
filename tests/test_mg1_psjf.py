"""
M/G/1 PSJF: MG1PsjfCalc vs SizeBasedQsSim.
"""

from __future__ import annotations

import os

import numpy as np
import yaml

from most_queue.random.distributions import GammaDistribution, H2Distribution
from most_queue.sim.size_based import SizeBasedQsSim
from most_queue.theory.srpt import MG1PsjfCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

SERVICE_TIME_CV_GE1 = float(params["service"]["cv_ge1"])
SERVICE_TIME_CV_LT1 = float(params["service"]["cv_lt1"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
NUM_OF_JOBS_SRPT = int(params.get("num_of_jobs_srpt", NUM_OF_JOBS))
UTILIZATION_FACTOR = float(params["utilization_factor"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

NUM_OF_CHANNELS = 1


def _h2_params():
    b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE
    return H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV_GE1)


def _gamma_params():
    b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE
    return GammaDistribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV_LT1)


def test_mg1_psjf_h2():
    """MG1PsjfCalc vs SizeBasedQsSim(PSJF): H2 service, rho=0.7, cv=1.2."""
    h2_params = _h2_params()

    calc = MG1PsjfCalc()
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(h2_params, "H")
    calc_res = calc.run()

    sim = SizeBasedQsSim(NUM_OF_CHANNELS, discipline="PSJF", verbose=False)
    sim.generator = np.random.default_rng(93001)
    sim.set_servers(h2_params, "H")
    sim.set_sources(ARRIVAL_RATE, "M")
    sim_res = sim.run(NUM_OF_JOBS_SRPT)

    assert np.allclose(sim_res.w[:1], calc_res.w[:1], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(sim_res.v[:1], calc_res.v[:1], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)


def test_mg1_psjf_gamma():
    """MG1PsjfCalc vs SizeBasedQsSim(PSJF): Gamma service, rho=0.7, cv=0.8."""
    gamma_params = _gamma_params()

    calc = MG1PsjfCalc()
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(gamma_params, "Gamma")
    calc_res = calc.run()

    sim = SizeBasedQsSim(NUM_OF_CHANNELS, discipline="PSJF", verbose=False)
    sim.generator = np.random.default_rng(93002)
    sim.set_servers(gamma_params, "Gamma")
    sim.set_sources(ARRIVAL_RATE, "M")
    sim_res = sim.run(NUM_OF_JOBS_SRPT)

    assert np.allclose(sim_res.w[:1], calc_res.w[:1], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(sim_res.v[:1], calc_res.v[:1], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
