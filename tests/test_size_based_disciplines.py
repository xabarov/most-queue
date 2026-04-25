"""Sanity checks for SizeBasedQsSim disciplines."""

import os

import numpy as np
import yaml

from most_queue.random.distributions import H2Distribution
from most_queue.sim.size_based import PerfectPredictor, SizeBasedQsSim

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

SERVICE_TIME_CV = float(params["service"]["cv"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

NUM_OF_CHANNELS = 1


def _h2_setup():
    b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE
    h2_params = H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV)
    return h2_params


def test_srpt_mean_sojourn_strictly_le_fcfs():
    """E[T_SRPT] <= E[T_FCFS]: SRPT is optimal for M/G/1 (Schrage 1968).

    Both simulators draw from independent seeds so they see different sample paths.
    With 100 k jobs the estimation error is small enough that the strict inequality
    should comfortably hold; we give one standard-deviation slack via MOMENTS_ATOL.
    """
    h2_params = _h2_setup()

    fcfs = SizeBasedQsSim(NUM_OF_CHANNELS, discipline="FCFS")
    fcfs.generator = np.random.default_rng(100001)
    fcfs.set_servers(h2_params, "H")
    fcfs.set_sources(ARRIVAL_RATE, "M")
    r_fcfs = fcfs.run(NUM_OF_JOBS)

    srpt = SizeBasedQsSim(NUM_OF_CHANNELS, discipline="SRPT")
    srpt.generator = np.random.default_rng(200002)
    srpt.set_servers(h2_params, "H")
    srpt.set_sources(ARRIVAL_RATE, "M")
    r_srpt = srpt.run(NUM_OF_JOBS)

    assert r_srpt.v[0] <= r_fcfs.v[0] + MOMENTS_ATOL


def test_spjf_perfect_close_to_sjf():
    h2_params = _h2_setup()
    seed = 7

    sjf = SizeBasedQsSim(NUM_OF_CHANNELS, discipline="SJF")
    sjf.generator = np.random.default_rng(seed)
    sjf.set_servers(h2_params, "H")
    sjf.set_sources(ARRIVAL_RATE, "M")
    r_sjf = sjf.run(NUM_OF_JOBS)

    spjf = SizeBasedQsSim(NUM_OF_CHANNELS, discipline="SPJF")
    spjf.generator = np.random.default_rng(seed)
    spjf.set_servers(h2_params, "H")
    spjf.set_sources(ARRIVAL_RATE, "M")
    spjf.set_predictor(PerfectPredictor())
    r_spjf = spjf.run(NUM_OF_JOBS)

    assert np.allclose(r_spjf.w[:2], r_sjf.w[:2], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(r_spjf.v[:2], r_sjf.v[:2], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)


def test_preemptive_disciplines_run_high_load():
    """Smoke: many preemptions without exceptions."""
    h2_params = _h2_setup()
    for disc in ("PSJF", "PSPJF", "SPRPT"):
        sim = SizeBasedQsSim(NUM_OF_CHANNELS, discipline=disc)
        sim.generator = np.random.default_rng(123)
        sim.set_servers(h2_params, "H")
        sim.set_sources(ARRIVAL_RATE, "M")
        r = sim.run(min(NUM_OF_JOBS, 50_000))
        assert r.v[0] > 0
        assert r.w[0] >= 0
