"""
Cross-validate the SLA layer against simulation for a two-class non-preemptive
priority M/G/1: fit-based P(W_k > deadline) per class from MG1NonPreemptiveCalc
moments vs the empirical deadline-violation frequency from
PriorityQueueSimulator. Directly relevant to SLO-tiered LLM serving
(e.g. premium vs free-tier requests).
"""

import os

import numpy as np
import yaml

from most_queue.random.distributions import GammaDistribution
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.non_preemptive.mg1 import MG1NonPreemptiveCalc
from most_queue.theory.utils.sla import deadline_violation_prob, slo_quantile

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

NUM_OF_JOBS = int(params["num_of_jobs"])

ARRIVAL_RATES = [0.15, 0.15]
SERVICE_TIMES_AVE = [1.0, 1.0]
# NOTE: fit_h2 (Aliev's method) has a known numerically unstable branch right
# at the boundary of H2-feasibility (third raw moment close to the minimum
# achievable for the given mean/cv), where it degenerates to a near-zero-mean
# fit. cv=2.0 here keeps both classes' waiting-time moment triples comfortably
# inside the feasible region (margin (m3 - t_min)/t_min > 5%), away from that
# edge case.
SERVICE_TIME_CV = 2.0

# Looser than the single-class M/G/1 / M/G/n / MAP/PH/1 cross-validation
# tests: priority waiting-time moments have higher cv (residual busy-period
# structure), so the 3-moment H2/Gamma fit is less precise in the tail.
VIOLATION_ATOL = 0.05


def test_priority_deadline_violation_prob_vs_sim():
    """Fit-based per-class P(W_k > D) from MG1NonPreemptiveCalc matches DES empirical frequency."""
    num_of_classes = len(ARRIVAL_RATES)
    b2 = [(m**2) * (1 + SERVICE_TIME_CV**2) for m in SERVICE_TIMES_AVE]
    gamma_params = [GammaDistribution.get_params([SERVICE_TIMES_AVE[j], b2[j]]) for j in range(num_of_classes)]
    b = [GammaDistribution.calc_theory_moments(gamma_params[j], 4) for j in range(num_of_classes)]

    calc = MG1NonPreemptiveCalc()
    calc.set_sources(ARRIVAL_RATES)
    calc.set_servers(b)
    calc_results = calc.run()

    qs = PriorityQueueSimulator(1, num_of_classes, "NP")
    sources = [{"type": "M", "params": ARRIVAL_RATES[j]} for j in range(num_of_classes)]
    servers_params = [{"type": "Gamma", "params": gamma_params[j]} for j in range(num_of_classes)]
    qs.set_sources(sources)
    qs.set_servers(servers_params)

    deadlines_per_class = [[slo_quantile(calc_results.w[k], p) for p in (0.3, 0.1)] for k in range(num_of_classes)]
    all_deadlines = sorted({d for per_class in deadlines_per_class for d in per_class})
    qs.set_deadline_thresholds(all_deadlines)

    qs.run(NUM_OF_JOBS)

    for k in range(num_of_classes):
        for d in deadlines_per_class[k]:
            fit_prob = deadline_violation_prob(calc_results.w[k], d)
            sim_prob = qs.get_empirical_violation_prob(k, d)
            assert np.isclose(fit_prob, sim_prob, atol=VIOLATION_ATOL), (k, d, fit_prob, sim_prob)


if __name__ == "__main__":
    test_priority_deadline_violation_prob_vs_sim()
