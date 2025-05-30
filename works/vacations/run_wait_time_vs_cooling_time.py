"""
Run simulation and calculation for different cooling mean times
and plot the wait time averages for both calculation and simulation.
"""
import os

import numpy as np
from run_one_calc_vs_sim import (calc_moments_by_mean_and_coev,
                                 run_calculation, run_simulation)
from utils import calc_rel_error_percent, plot_probs, plot_w1, plot_w1_errors

ARRIVAL_RATE = 1.0

SERVICE_TIME_CV = 1.2

UTILIZATION = 0.7

WARM_UP_TIME_MEAN = 3.1
WARM_UP_TIME_CV = 0.87

COOL_TIME_MEAN = 3.5
COOL_TIME_CV = 1.1

COOL_DELAY_MEAN = 3.71
COOL_DELAY_CV = 1.2

NUM_OF_CHANNES = 3

NUM_OF_JOBS_PER_SIM = 300_000  # Number of jobs per simulation
NUM_OF_SIM_TO_AVERAGE = 10  # Number of simulations to average over


def run_wait_time_vs_cool_ave(cool_min: float = 1.0, cool_max: float = 5.0,
                              num_points: int = 20, save_path: str = None):
    """
    Run simulation and calculation for different cooling mean times
    """
    cools = np.linspace(cool_min, cool_max, num_points)

    w1_num = []
    w1_sim = []

    w1_rel_errors = []

    cool_probs_num = []
    cool_probs_sim = []

    total_num_time = 0
    total_sim_time = 0

    service_mean = NUM_OF_CHANNES*UTILIZATION/ARRIVAL_RATE

    b = calc_moments_by_mean_and_coev(service_mean, SERVICE_TIME_CV)
    b_w = calc_moments_by_mean_and_coev(WARM_UP_TIME_MEAN, WARM_UP_TIME_CV)
    b_d = calc_moments_by_mean_and_coev(COOL_DELAY_MEAN, COOL_DELAY_CV)

    for cool_num, cool_ave in enumerate(cools):
        print(
            f"Start {cool_num + 1}/{len(cools)} with cooling time={cool_ave:0.3f}... ")

        b_c = calc_moments_by_mean_and_coev(cool_ave, COOL_TIME_CV)

        num_results = run_calculation(
            arrival_rate=ARRIVAL_RATE, num_channels=NUM_OF_CHANNES, b=b, b_w=b_w, b_c=b_c, b_d=b_d)
        sim_results = run_simulation(
            arrival_rate=ARRIVAL_RATE, num_channels=NUM_OF_CHANNES, b=b, b_w=b_w, b_c=b_c, b_d=b_d,
            num_of_jobs=NUM_OF_JOBS_PER_SIM, ave_num=NUM_OF_SIM_TO_AVERAGE)

        w1_num.append(num_results["w"][0])
        w1_sim.append(sim_results["w"][0])

        w1_rel_errors.append(calc_rel_error_percent(
            sim_results["w"][0], num_results["w"][0]))

        cool_probs_num.append(num_results["cold_prob"])
        cool_probs_sim.append(sim_results["cold_prob"])

        total_num_time += num_results["process_time"]
        total_sim_time += sim_results["process_time"]

    # Print process time comparison
    print(f"Total process time for num: {total_num_time:.4g}")
    print(f"Total process time for sim: {total_sim_time:.4g}")

    if save_path:
        wait_time_save_path = os.path.join(
            save_path, 'w1_vs_cool_ave.png')

        plot_w1(cools, w1_num, w1_sim, x_label='Cooling Average',
                save_path=wait_time_save_path)

        w1_error_save_path = os.path.join(
            save_path, 'w1_error_vs_cool_ave.png')

        plot_w1_errors(cools, w1_rel_errors,
                       x_label='Cooling Average', save_path=w1_error_save_path)

        cool_probs_save_path = os.path.join(
            save_path, 'cooling_probs_vs_cool_ave.png')

        plot_probs(cools, cool_probs_num, cool_probs_sim, x_label='Cooling Average',
                   save_path=cool_probs_save_path)

    return  cools, w1_num, w1_sim, w1_rel_errors, cool_probs_sim, cool_probs_num

def run_wait_time_vs_cool_cv(cool_cv_min: float = 0.3, cool_cv_max: float = 3.0,
                             num_points: int = 20, save_path: str = None):
    """
    Run simulation and calculation for different cooling coefficient of variation
    """
    cools = np.linspace(cool_cv_min, cool_cv_max, num_points)

    w1_num = []
    w1_sim = []

    w1_rel_errors = []

    cool_probs_num = []
    cool_probs_sim = []

    total_num_time = 0
    total_sim_time = 0

    service_mean = NUM_OF_CHANNES*UTILIZATION/ARRIVAL_RATE

    b = calc_moments_by_mean_and_coev(service_mean, SERVICE_TIME_CV)
    b_w = calc_moments_by_mean_and_coev(WARM_UP_TIME_MEAN, WARM_UP_TIME_CV)
    b_d = calc_moments_by_mean_and_coev(COOL_DELAY_MEAN, COOL_DELAY_CV)

    for cool_num, cool_cv in enumerate(cools):
        print(
            f"Start {cool_num + 1}/{len(cools)} with cooling cv={cool_cv:0.3f}... ")

        b_c = calc_moments_by_mean_and_coev(COOL_TIME_MEAN, cool_cv)

        num_results = run_calculation(
            arrival_rate=ARRIVAL_RATE, num_channels=NUM_OF_CHANNES, b=b, b_w=b_w, b_c=b_c, b_d=b_d)
        sim_results = run_simulation(
            arrival_rate=ARRIVAL_RATE, num_channels=NUM_OF_CHANNES, b=b, b_w=b_w, b_c=b_c, b_d=b_d,
            num_of_jobs=NUM_OF_JOBS_PER_SIM, ave_num=NUM_OF_SIM_TO_AVERAGE)

        w1_num.append(num_results["w"][0])
        w1_sim.append(sim_results["w"][0])

        w1_rel_errors.append(calc_rel_error_percent(
            sim_results["w"][0], num_results["w"][0]))

        cool_probs_num.append(num_results["cold_prob"])
        cool_probs_sim.append(sim_results["cold_prob"])

        total_num_time += num_results["process_time"]
        total_sim_time += sim_results["process_time"]

    # Print process time comparison
    print(f"Total process time for num: {total_num_time:.4g}")
    print(f"Total process time for sim: {total_sim_time:.4g}")

    if save_path:
        wait_time_save_path = os.path.join(
            save_path, 'w1_vs_cool_cv.png')

        plot_w1(cools, w1_num, w1_sim, x_label='Cooling CV',
                save_path=wait_time_save_path)

        w1_error_save_path = os.path.join(
            save_path, 'w1_error_vs_cool_cv.png')

        plot_w1_errors(cools, w1_rel_errors,
                       x_label='Cooling CV', save_path=w1_error_save_path)

        cool_probs_save_path = os.path.join(
            save_path, 'cooling_probs_vs_cool_cv.png')

        plot_probs(cools, cool_probs_num, cool_probs_sim, x_label='Cooling CV',
                   save_path=cool_probs_save_path)

    return  cools, w1_num, w1_sim, w1_rel_errors, cool_probs_sim, cool_probs_num
    
if __name__ == "__main__":

    # cur file dir
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    result_path = os.path.join(cur_dir, 'results/cooling')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    run_wait_time_vs_cool_ave(save_path=result_path)
    run_wait_time_vs_cool_cv(save_path=result_path)
