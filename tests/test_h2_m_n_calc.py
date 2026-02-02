"""
Testing the Takahashi-Takami method for calculating an H2/M/n queue.

H2 (hyperexponential) arrival, M (exponential) service.
Verification via simulation.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import (
    print_sojourn_moments,
    print_waiting_moments,
    probs_print,
)
from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.gmc_takahasi import H2MnCalc

cur_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(cur_dir, "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)

NUM_OF_CHANNELS = int(params["num_of_channels"])
ARRIVAL_RATE = float(params["arrival"]["rate"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]
MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

# Looser prob tolerances for H2/M/n (arrival distribution sensitivity)
PROBS_ATOL = 0.12
PROBS_RTOL = 0.2

# H2 needs cv >= 1 for real params; use 1.2
ARRIVAL_CV = 1.2


def test_h2_m_n_calc():
    """Test H2MnCalc against QsSim for H2/M/n."""
    # Mean interarrival = 1/rate
    a1 = 1.0 / ARRIVAL_RATE
    a = gamma_moments_by_mean_and_cv(a1, ARRIVAL_CV)
    h2_params = H2Distribution.get_params(a)
    a = H2Distribution.calc_theory_moments(h2_params, 4)

    # Mean service for given utilization: rho = (1/a1)*b/n => b = rho*n*a1
    b_mean = UTILIZATION_FACTOR * NUM_OF_CHANNELS * a1

    calc = H2MnCalc(n=NUM_OF_CHANNELS)
    calc.set_sources(a)
    calc.set_servers(b_mean)

    calc_results = calc.run()

    qs = QsSim(NUM_OF_CHANNELS)
    qs.set_sources(h2_params, "H")
    qs.set_servers(1.0 / b_mean, "M")  # rate for M

    sim_results = qs.run(NUM_OF_JOBS)

    print("\nH2/M/n: Takahasi-Takami vs simulation")
    print(f"Simulation - H2/M/{NUM_OF_CHANNELS}")
    print(f"Calculation - H2/M/{NUM_OF_CHANNELS}")
    print(f"Utilization: {UTILIZATION_FACTOR:.2f}, Arrival CV: {ARRIVAL_CV:.2f}")
    print(f"Iterations: {calc.num_of_iter_}")

    probs_print(sim_results.p, calc_results.p, 10)
    print_sojourn_moments(sim_results.v, calc_results.v)
    print_waiting_moments(sim_results.w, calc_results.w)

    assert np.allclose(sim_results.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.w, calc_results.w, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG
    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


if __name__ == "__main__":
    test_h2_m_n_calc()
