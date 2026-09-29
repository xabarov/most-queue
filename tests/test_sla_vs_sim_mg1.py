"""
Cross-validate the SLA layer (most_queue.theory.utils.sla) against simulation
for M/G/1: fit-based P(W > deadline) from MG1Calc moments vs the empirical
deadline-violation frequency from QsSim.
"""

import os

import numpy as np
import yaml

from most_queue.random.distributions import H2Distribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mg1 import MG1Calc
from most_queue.theory.utils.sla import deadline_violation_prob, slo_quantile

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
SERVICE_TIME_CV = float(params["service"]["cv"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])

# Deadline-violation probability is itself a statistical estimate from a
# finite sample (unlike raw moments, refined online) -- use an absolute
# tolerance in probability space, looser than moment comparisons.
VIOLATION_ATOL = 0.03


def test_mg1_deadline_violation_prob_vs_sim():
    """Fit-based P(W > D) from MG1Calc moments matches DES empirical frequency."""
    b1 = UTILIZATION_FACTOR / ARRIVAL_RATE
    h2_params = H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV)
    b = H2Distribution.calc_theory_moments(h2_params, 4)

    calc = MG1Calc()
    calc.set_sources(ARRIVAL_RATE)
    calc.set_servers(b)
    calc_results = calc.run()

    # Pick deadlines from the calculator's own SLO quantiles (p=0.3, p=0.1) --
    # squarely inside the range simulation can estimate accurately at this
    # sample size, not deep in the tail.
    deadlines = [slo_quantile(calc_results.w, p) for p in (0.3, 0.1)]

    qs = QsSim(1)
    qs.set_servers(h2_params, "H")
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.set_deadline_thresholds(deadlines)
    qs.run(NUM_OF_JOBS)

    for d in deadlines:
        fit_prob = deadline_violation_prob(calc_results.w, d)
        sim_prob = qs.get_empirical_violation_prob(d)
        assert np.isclose(fit_prob, sim_prob, atol=VIOLATION_ATOL), (d, fit_prob, sim_prob)


if __name__ == "__main__":
    test_mg1_deadline_violation_prob_vs_sim()
