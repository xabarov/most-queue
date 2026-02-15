"""
Test QS M/G/n queue with disasters.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

from most_queue.io.tables import print_raw_moments, print_waiting_moments, probs_print
from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.negative import DisasterScenario, NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

with open(params_path, "r", encoding="utf-8") as file:
    params = yaml.safe_load(file)


NUM_OF_CHANNELS = int(params["num_of_channels"])

SERVICE_TIME_CV = float(params["service"]["cv"])
NUM_OF_JOBS = int(params["num_of_jobs"]) * 3
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
    Test QS M/G/n queue with disasters.
    """

    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE_POSITIVE  # average service time

    b = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)

    # Run simulation
    queue_sim = QsSimNegatives(NUM_OF_CHANNELS, NegativeServiceType.DISASTER)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, "M")
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, "M")
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, "Gamma")

    sim_results = queue_sim.run(NUM_OF_JOBS)

    # Run calc
    queue_calc = MGnNegativeDisasterCalc(n=NUM_OF_CHANNELS)
    queue_calc.set_sources(l_pos=ARRIVAL_RATE_POSITIVE, l_neg=ARRIVAL_RATE_NEGATIVE)
    queue_calc.set_servers(b=b)

    calc_results = queue_calc.run()

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    probs_print(sim_results.p, calc_results.p)
    print_raw_moments(sim_results.v, calc_results.v, header="sojourn total")
    print_waiting_moments(sim_results.w, calc_results.w)

    print_raw_moments(sim_results.v_served, calc_results.v_served, header="sojourn served")
    print_raw_moments(sim_results.v_broken, calc_results.v_broken, header="sojourn broken")

    assert np.allclose(sim_results.v, calc_results.v, rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG

    assert np.allclose(sim_results.p[:10], calc_results.p[:10], atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG


def test_mgn_requeue_all():
    """
    Test QS M/G/n queue with DISASTER negatives in "REQUEUE_ALL" scenario:
    a negative arrival interrupts service and returns all jobs-in-service back to the queue.

    In this scenario there are no broken jobs; all jobs are eventually served.
    Theoretical calculation uses MGnNegativeDisasterCalc(requeue_on_disaster=True).
    """

    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE_POSITIVE  # average service time
    b = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)

    # Run simulation (REQUEUE_ALL)
    queue_sim = QsSimNegatives(
        NUM_OF_CHANNELS,
        NegativeServiceType.DISASTER,
        verbose=False,
        disaster_scenario=DisasterScenario.REQUEUE_ALL,
    )
    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, "M")
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, "M")
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, "Gamma")
    sim_results = queue_sim.run(NUM_OF_JOBS)

    # Run calc (REQUEUE)
    queue_calc = MGnNegativeDisasterCalc(n=NUM_OF_CHANNELS, requeue_on_disaster=True)
    queue_calc.set_sources(l_pos=ARRIVAL_RATE_POSITIVE, l_neg=ARRIVAL_RATE_NEGATIVE)
    queue_calc.set_servers(b=b)
    calc_results = queue_calc.run()

    print(f"Simulation duration: {sim_results.duration:.5f} sec")
    print(f"Calculation duration: {calc_results.duration:.5f} sec")

    probs_print(sim_results.p, calc_results.p)
    print_raw_moments(sim_results.v, calc_results.v, header="sojourn total")
    print_waiting_moments(sim_results.w, calc_results.w)

    # In REQUEUE scenario, there should be no broken jobs
    assert np.allclose(sim_results.v_broken, [0.0, 0.0, 0.0, 0.0]), ERROR_MSG
    assert np.allclose(calc_results.v_broken, [0.0, 0.0, 0.0, 0.0]), ERROR_MSG

    # All jobs are served => v_served equals v
    assert np.allclose(sim_results.v_served, sim_results.v), ERROR_MSG
    assert np.allclose(calc_results.v_served, calc_results.v), ERROR_MSG

    # Compare first moments (higher moments are more noise-sensitive)
    assert np.allclose(
        [sim_results.w[0], sim_results.v[0]],
        [calc_results.w[0], calc_results.v[0]],
        rtol=MOMENTS_RTOL,
        atol=MOMENTS_ATOL,
    ), ERROR_MSG


def plot_w_and_v_vs_l_neg(
    l_neg_min=0.05,
    l_neg_max=1.2,
    num_points=20,
    run_sim=False,
    save_path=None,
):
    """
    Строит график среднего времени ожидания E[W] и среднего времени пребывания E[V]
    в зависимости от интенсивности отрицательных заявок (DISASTER).

    Параметры системы берутся из default_params.yaml (n, λ_pos, ρ, CV).
    """
    l_neg_arr = np.linspace(l_neg_min, l_neg_max, num_points)
    w_calc = []
    v_calc = []
    w_sim = [] if run_sim else None
    v_sim = [] if run_sim else None

    b1 = NUM_OF_CHANNELS * UTILIZATION_FACTOR / ARRIVAL_RATE_POSITIVE
    b = gamma_moments_by_mean_and_cv(b1, SERVICE_TIME_CV)
    gamma_params = GammaDistribution.get_params([b[0], b[1]])

    for l_neg in l_neg_arr:
        calc = MGnNegativeDisasterCalc(n=NUM_OF_CHANNELS)
        calc.set_sources(l_pos=ARRIVAL_RATE_POSITIVE, l_neg=float(l_neg))
        calc.set_servers(b=b)
        res = calc.run()
        w_calc.append(np.real(res.w[0]))
        v_calc.append(np.real(res.v[0]))

        if run_sim:
            sim = QsSimNegatives(NUM_OF_CHANNELS, NegativeServiceType.DISASTER, verbose=False)
            sim.set_negative_sources(float(l_neg), "M")
            sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, "M")
            sim.set_servers(gamma_params, "Gamma")
            sim_res = sim.run(NUM_OF_JOBS)
            w_sim.append(float(sim_res.w[0]))
            v_sim.append(float(sim_res.v[0]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(l_neg_arr, w_calc, "b-o", label=r"E[W], расчёт", markersize=4)
    ax.plot(l_neg_arr, v_calc, "g-s", label=r"E[V], расчёт", markersize=4)
    if run_sim and w_sim is not None and v_sim is not None:
        ax.plot(l_neg_arr, w_sim, "b--", alpha=0.7, label=r"E[W], симуляция")
        ax.plot(l_neg_arr, v_sim, "g--", alpha=0.7, label=r"E[V], симуляция")
    ax.set_xlabel(r"Интенсивность отрицательных заявок $\lambda_{neg}$")
    ax.set_ylabel("Среднее время")
    ax.set_title(
        r"M/H$_2$/" + str(NUM_OF_CHANNELS) + r" DISASTER: E[W], E[V] vs $\lambda_{neg}$ "
        r"($\lambda_{pos}$="
        + f"{ARRIVAL_RATE_POSITIVE}, "
        + r"$\rho$="
        + f"{UTILIZATION_FACTOR}, CV={SERVICE_TIME_CV})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    test_mgn_requeue_all()
