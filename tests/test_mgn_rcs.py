"""
Test QS M/G/n queue with negative jobs and RCS discipline.
"""

import os

import numpy as np
import yaml

from most_queue.io.tables import print_raw_moments, print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives, RcsScenario
from most_queue.structs import NegativeArrivalsResults
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)


NUM_OF_CHANNELS = int(params["num_of_channels"])

SERVICE_TIME_CV = float(params["service"]["cv"])
NUM_OF_JOBS = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
ERROR_MSG = params["error_msg"]

MOMENTS_ATOL = float(params["moments_atol"])
MOMENTS_RTOL = float(params["moments_rtol"])

PROBS_ATOL = float(params["probs_atol"])
PROBS_RTOL = float(params["probs_rtol"])

ARRIVAL_RATE_POSITIVE = float(params["arrival"]["rate"])
ARRIVAL_RATE_NEGATIVE = 0.3 * ARRIVAL_RATE_POSITIVE


def test_mgn():
    """
    Test QS M/G/n queue with negative jobs and RCS discipline.
    """

    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE_POSITIVE  # average service time

    b = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)

    # Run simulation
    queue_sim = QsSimNegatives(NUM_OF_CHANNELS, NegativeServiceType.RCS)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, "M")
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, "M")
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, "Gamma")

    sim_results: NegativeArrivalsResults = queue_sim.run(NUM_OF_JOBS)

    # Run calc
    queue_calc = MGnNegativeRCSCalc(n=NUM_OF_CHANNELS)

    queue_calc.set_sources(l_pos=ARRIVAL_RATE_POSITIVE, l_neg=ARRIVAL_RATE_NEGATIVE)
    queue_calc.set_servers(b=b)

    calc_results: NegativeArrivalsResults = queue_calc.run()

    print(f"q = {queue_calc.get_q():0.3f}")

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    probs_print(sim_results.p, calc_results.p)
    print_sojourn_moments(sim_results.v, calc_results.v)
    print_waiting_moments(sim_results.w, calc_results.w)

    print_raw_moments(sim_results.v_served, calc_results.v_served, header="sojourn served")  # pylint: disable=no-member
    print_raw_moments(sim_results.v_broken, calc_results.v_broken, header="sojourn broken")  # pylint: disable=no-member

    assert np.allclose(sim_results.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


def test_mgn_rcs_requeue_smoke():
    """
    Smoke test for MGnNegativeRCSCalc(requeue_on_disaster=True):
    - no broken jobs (v_broken == 0),
    - all jobs are eventually served (v_served == v),
    - q == 1.
    """

    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE_POSITIVE  # average service time
    b = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)

    # Run simulation (RCS REQUEUE: interrupt one in-service job and requeue it)
    queue_sim = QsSimNegatives(
        NUM_OF_CHANNELS,
        NegativeServiceType.RCS,
        verbose=False,
        rcs_scenario=RcsScenario.REQUEUE,
    )
    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, "M")
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, "M")
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, "Gamma")
    sim_results: NegativeArrivalsResults = queue_sim.run(NUM_OF_JOBS)

    # Run calc (REQUEUE approximation via effective service time)
    queue_calc = MGnNegativeRCSCalc(n=NUM_OF_CHANNELS, requeue_on_disaster=True)
    queue_calc.set_sources(l_pos=ARRIVAL_RATE_POSITIVE, l_neg=ARRIVAL_RATE_NEGATIVE)
    queue_calc.set_servers(b=b)
    calc_results: NegativeArrivalsResults = queue_calc.run()

    # Print tables in the same format as test_mgn().
    sim_q = float(queue_sim.served) / float(queue_sim.total) if queue_sim.total > 0 else 0.0
    print(f"q(sim) = {sim_q:0.3f}, q(calc) = {queue_calc.get_q():0.3f}")
    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")
    probs_print(sim_results.p, calc_results.p)
    print_sojourn_moments(sim_results.v, calc_results.v)
    print_waiting_moments(sim_results.w, calc_results.w)

    # In REQUEUE mode, no one is removed by negatives; all jobs are eventually served.
    assert np.allclose(sim_results.v_broken, [0.0, 0.0, 0.0, 0.0]), ERROR_MSG  # pylint: disable=no-member
    assert np.allclose(calc_results.v_broken, [0.0, 0.0, 0.0, 0.0]), ERROR_MSG  # pylint: disable=no-member
    assert np.allclose(sim_results.v_served, sim_results.v), ERROR_MSG  # pylint: disable=no-member
    assert np.allclose(calc_results.v_served, calc_results.v), ERROR_MSG  # pylint: disable=no-member
    assert np.allclose(queue_calc.get_q(), 1.0), ERROR_MSG

    # Compare first moments (higher moments are more noise-sensitive)
    assert np.allclose(
        [sim_results.w[0], sim_results.v[0]],
        [calc_results.w[0], calc_results.v[0]],
        rtol=MOMENTS_RTOL,
        atol=MOMENTS_ATOL,
    ), ERROR_MSG

    # Numerical method should produce non-trivial second moments in REQUEUE mode
    # (used for SKO / standard deviation tables in the paper).
    assert calc_results.w[1] > 0.0, ERROR_MSG
    assert calc_results.v[1] > 0.0, ERROR_MSG


if __name__ == "__main__":
    test_mgn_rcs_requeue_smoke()
