"""
M/G/1 SPJF: MG1SpjfCalc vs SizeBasedQsSim; perfect predictions match SJF.
"""

from __future__ import annotations

import os

import numpy as np
import yaml

from most_queue.random.distributions import H2Distribution
from most_queue.sim.size_based import PerfectPredictor as SimPerfectPredictor
from most_queue.sim.size_based import SizeBasedQsSim
from most_queue.sim.utils.predictor import ExpNoiseSimPredictor
from most_queue.theory.srpt import MG1SjfCalc, MG1SpjfCalc
from most_queue.theory.srpt.utils.predictor import ExpNoisePredictor, PerfectPredictor as TheoryPerfectPredictor

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

SERVICE_TIME_CV_GE1 = float(params["service"]["cv_ge1"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

NUM_OF_CHANNELS = 1


def _h2_params():
    b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE
    return H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV_GE1)


def test_mg1_spjf_perfect_theory_matches_sjf_theory():
    """MG1SpjfCalc(PerfectPredictor) theory must equal MG1SjfCalc theory for H2 service."""
    h2_params = _h2_params()

    sjf = MG1SjfCalc()
    sjf.set_sources(ARRIVAL_RATE)
    sjf.set_servers(h2_params, "H")
    sjf_res = sjf.run()

    spjf = MG1SpjfCalc()
    spjf.set_sources(ARRIVAL_RATE)
    spjf.set_servers(h2_params, "H")
    spjf.set_predictor(TheoryPerfectPredictor())
    spjf_res = spjf.run()

    assert np.allclose(spjf_res.w[:1], sjf_res.w[:1], rtol=1e-5, atol=1e-4)
    assert np.allclose(spjf_res.v[:1], sjf_res.v[:1], rtol=1e-5, atol=1e-4)


def test_mg1_spjf_perfect_sim_matches_sjf_sim():
    """SPJF(PerfectPredictor) sim converges to SJF sim with identical RNG seed."""
    h2_params = _h2_params()
    seed = 94001

    sjf = SizeBasedQsSim(NUM_OF_CHANNELS, discipline="SJF", verbose=False)
    sjf.generator = np.random.default_rng(seed)
    sjf.set_servers(h2_params, "H")
    sjf.set_sources(ARRIVAL_RATE, "M")
    r_sjf = sjf.run(NUM_OF_JOBS)

    spjf = SizeBasedQsSim(NUM_OF_CHANNELS, discipline="SPJF", verbose=False)
    spjf.generator = np.random.default_rng(seed)
    spjf.set_servers(h2_params, "H")
    spjf.set_sources(ARRIVAL_RATE, "M")
    spjf.set_predictor(SimPerfectPredictor())
    r_spjf = spjf.run(NUM_OF_JOBS)

    assert np.allclose(r_spjf.w[:1], r_sjf.w[:1], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(r_spjf.v[:1], r_sjf.v[:1], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)


def test_mg1_spjf_exp_noise_theory_vs_sim():
    """MG1SpjfCalc(ExpNoise) vs SizeBasedQsSim(SPJF, ExpNoiseSimPredictor): H2, rho=0.7."""
    h2_params = _h2_params()

    calc = MG1SpjfCalc()
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(h2_params, "H")
    calc.set_predictor(ExpNoisePredictor())
    calc_res = calc.run()

    sim = SizeBasedQsSim(NUM_OF_CHANNELS, discipline="SPJF", verbose=False)
    sim.generator = np.random.default_rng(94002)
    sim.set_servers(h2_params, "H")
    sim.set_sources(ARRIVAL_RATE, "M")
    sim.set_predictor(ExpNoiseSimPredictor())
    sim_res = sim.run(NUM_OF_JOBS)

    assert np.allclose(sim_res.w[:1], calc_res.w[:1], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(sim_res.v[:1], calc_res.v[:1], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
