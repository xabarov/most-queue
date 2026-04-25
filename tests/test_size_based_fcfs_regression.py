"""
FCFS mode of SizeBasedQsSim should match QsSim (same RNG, no size-at-arrival for FCFS).
"""

import os

import numpy as np
import yaml

from most_queue.random.distributions import H2Distribution
from most_queue.sim.base import QsSim
from most_queue.sim.size_based import SizeBasedQsSim

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

SERVICE_TIME_CV = float(params["service"]["cv"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

NUM_OF_CHANNELS = 1
RNG_SEED = 20250426


def test_fcfs_size_based_matches_qssim():
    b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE
    h2_params = H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV)

    rng = np.random.default_rng(RNG_SEED)

    qs = QsSim(NUM_OF_CHANNELS)
    qs.generator = rng
    qs.set_servers(h2_params, "H")
    qs.set_sources(ARRIVAL_RATE, "M")
    sim_qs = qs.run(NUM_OF_JOBS)

    rng2 = np.random.default_rng(RNG_SEED)
    sb = SizeBasedQsSim(NUM_OF_CHANNELS, discipline="FCFS")
    sb.generator = rng2
    sb.set_servers(h2_params, "H")
    sb.set_sources(ARRIVAL_RATE, "M")
    sim_sb = sb.run(NUM_OF_JOBS)

    assert np.allclose(sim_sb.w[:2], sim_qs.w[:2], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(sim_sb.v[:2], sim_qs.v[:2], rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(np.array(sim_sb.p[:10]), np.array(sim_qs.p[:10]), atol=PROBS_ATOL, rtol=PROBS_RTOL)
