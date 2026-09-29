"""
Cross-validate the SLA layer against simulation for M/G/n: fit-based
P(W > deadline) from MGnCalc (Takahasi-Takami) moments vs the empirical
deadline-violation frequency from QsSim.
"""

import os

import numpy as np
import yaml

from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.theory.utils.sla import deadline_violation_prob, slo_quantile

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

NUM_OF_CHANNELS = int(params["num_of_channels"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_TIME_CV = float(params["service"]["cv"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])

VIOLATION_ATOL = 0.03


def test_mgn_deadline_violation_prob_vs_sim():
    """Fit-based P(W > D) from MGnCalc moments matches DES empirical frequency."""
    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE
    b = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)

    tt = MGnCalc(n=NUM_OF_CHANNELS)
    tt.set_sources(l=ARRIVAL_RATE)
    tt.set_servers(b=b)
    calc_results = tt.run()

    deadlines = [slo_quantile(calc_results.w, p) for p in (0.3, 0.1)]

    qs = QsSim(NUM_OF_CHANNELS, seed=42)
    qs.set_sources(ARRIVAL_RATE, "M")
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    qs.set_servers(gamma_params, "Gamma")
    qs.set_deadline_thresholds(deadlines)
    qs.run(NUM_OF_JOBS)

    for d in deadlines:
        fit_prob = deadline_violation_prob(calc_results.w, d)
        sim_prob = qs.get_empirical_violation_prob(d)
        assert np.isclose(fit_prob, sim_prob, atol=VIOLATION_ATOL), (d, fit_prob, sim_prob)


if __name__ == "__main__":
    test_mgn_deadline_violation_prob_vs_sim()
