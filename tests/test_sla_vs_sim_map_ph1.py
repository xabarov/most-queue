"""
Cross-validate the SLA layer against simulation for a bursty MAP(MMPP-2)/PH/1
queue: fit-based P(W > deadline) from MapPh1Calc moments vs the empirical
deadline-violation frequency from QsSim. Most relevant of the four
cross-validation targets for LLM-serving traffic, which is typically
autocorrelated rather than Poisson.
"""

import os

import numpy as np
import yaml

from most_queue.random.distributions import H2Distribution
from most_queue.random.map_ph import MAP, PHDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.matrix.map_ph1 import MapPh1Calc
from most_queue.theory.utils.sla import deadline_violation_prob, slo_quantile

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])

VIOLATION_ATOL = 0.03


def test_map_ph1_deadline_violation_prob_vs_sim():
    """Fit-based P(W > D) from MapPh1Calc (MMPP-2 arrivals) matches DES empirical frequency."""
    mmpp = MAP.mmpp([2.0 * ARRIVAL_RATE, 0.4 * ARRIVAL_RATE], np.array([[-0.2, 0.2], [0.3, -0.3]]))
    lam = MAP.arrival_rate(mmpp)
    service_mean = UTILIZATION_FACTOR / lam
    h2_srv = H2Distribution.get_params_by_mean_and_cv(service_mean, 1.2)
    ph_srv = PHDistribution.from_h2(h2_srv)

    calc = MapPh1Calc()
    calc.set_sources(mmpp)
    calc.set_servers(ph_srv)
    calc_results = calc.run()

    deadlines = [slo_quantile(calc_results.w, p) for p in (0.3, 0.1)]

    sim = QsSim(1, seed=42)
    sim.set_sources(mmpp, "MAP")
    sim.set_servers(ph_srv, "PH")
    sim.set_deadline_thresholds(deadlines)
    sim.run(NUM_OF_JOBS)

    for d in deadlines:
        fit_prob = deadline_violation_prob(calc_results.w, d)
        sim_prob = sim.get_empirical_violation_prob(d)
        assert np.isclose(fit_prob, sim_prob, atol=VIOLATION_ATOL), (d, fit_prob, sim_prob)


if __name__ == "__main__":
    test_map_ph1_deadline_violation_prob_vs_sim()
