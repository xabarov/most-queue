"""
Cross-validate the exact max-n-Pareto layer (EPIC-022) against simulation:
SplitJoinCalc(approximation="pareto") vs ForkJoinSim with Pareto-distributed
sub-task service times.
"""

import os

import numpy as np
import yaml

from most_queue.random.utils.params import ParetoParams
from most_queue.sim.fork_join import ForkJoinSim
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.fork_join.split_join import SplitJoinCalc
from most_queue.theory.utils.max_dist import pareto_max_tail

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

NUM_OF_CHANNELS = int(params["num_of_channels"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
ERROR_MSG = params["error_msg"]

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

PARETO_PARAMS = ParetoParams(alpha=4.5, K=0.3)


def test_split_join_pareto_vs_sim():
    """Exact Split-Join sojourn moments (Pareto sub-task service) match DES."""
    calc = SplitJoinCalc(n=NUM_OF_CHANNELS, calc_params=CalcParams(approx_distr="pareto"))
    calc.set_sources(l=ARRIVAL_RATE)
    calc.set_servers(PARETO_PARAMS)
    calc_results = calc.run()

    qs = ForkJoinSim(NUM_OF_CHANNELS, NUM_OF_CHANNELS, True)
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.set_servers(PARETO_PARAMS, "Pa")
    sim_results = qs.run(NUM_OF_JOBS)

    assert np.allclose(
        sim_results.v[: len(calc_results.v)], calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL
    ), ERROR_MSG


def test_pareto_max_tail_vs_empirical_subtask_maximum():
    """
    pareto_max_tail matches the empirical tail of max(X_1..X_n) sampled
    directly (independent of the M/G/1 queueing layer) -- a sanity check on
    the exact formula itself, not on SplitJoinCalc's composition with MG1Calc.
    """
    rng = np.random.default_rng(7)
    n_trials = 200_000
    alpha, k_scale = PARETO_PARAMS.alpha, PARETO_PARAMS.K
    # inverse-CDF sampling: X = K * (1-U)^(-1/alpha), U ~ Uniform(0,1)
    samples = k_scale * (1.0 - rng.random((n_trials, NUM_OF_CHANNELS))) ** (-1.0 / alpha)
    maxima = samples.max(axis=1)

    for x in (k_scale * 1.5, k_scale * 3.0, k_scale * 10.0):
        empirical = float((maxima > x).mean())
        theory = pareto_max_tail(PARETO_PARAMS, NUM_OF_CHANNELS, x)
        assert np.isclose(empirical, theory, atol=0.01), (x, empirical, theory)


if __name__ == "__main__":
    test_split_join_pareto_vs_sim()
    test_pareto_max_tail_vs_empirical_subtask_maximum()
