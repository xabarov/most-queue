"""
Run simulation and calculation for different cooling mean times
and plot the wait time averages for both calculation and simulation.
"""
import os

import numpy as np

from run_one_calc_vs_sim import (
    calc_moments_by_mean_and_coev,
    run_calculation,
    run_simulation,
)
from utils import calc_rel_error_percent, plot_probs, plot_w1, plot_w1_errors


def run_cool_ave(qp, save_path: str = None):
    """
    Run simulation and calculation for different cooling mean times
    """
    cools = np.linspace(qp['cooling']['mean']['min'], qp['cooling']['mean']['max'],
                        qp['cooling']['mean']['num_points'])

    w1_num = []
    w1_sim = []

    w1_rel_errors = []

    cool_probs_num = []
    cool_probs_sim = []

    total_num_time = 0
    total_sim_time = 0

    service_mean = qp['channels']['base']*qp['utilization']['base']/qp['arrival_rate']

    b = calc_moments_by_mean_and_coev(service_mean, qp['service']['cv']['base'])
    b_w = calc_moments_by_mean_and_coev(
        qp['warmup']['mean']['base'], qp['warmup']['cv']['base'])
    b_d = calc_moments_by_mean_and_coev(qp['delay']['mean']['base'], qp['delay']['cv']['base'])

    for cool_num, cool_ave in enumerate(cools):
        print(
            f"Start {cool_num + 1}/{len(cools)} with cooling time={cool_ave:0.3f}... ")

        b_c = calc_moments_by_mean_and_coev(cool_ave, qp['cooling']['cv']['base'])

        num_results = run_calculation(
            arrival_rate=qp['arrival_rate'], num_channels=qp['channels']['base'], b=b, b_w=b_w, b_c=b_c, b_d=b_d)
        sim_results = run_simulation(
            arrival_rate=qp['arrival_rate'], num_channels=qp['channels']['base'], b=b, b_w=b_w, b_c=b_c, b_d=b_d,
            num_of_jobs=qp['jobs_per_sim'], ave_num=qp['sim_to_average'])

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

    return cools, w1_num, w1_sim, w1_rel_errors, cool_probs_sim, cool_probs_num


def run_cool_cv(qp, save_path: str = None):
    """
    Run simulation and calculation for different cooling coefficient of variation
    """
    cools = np.linspace(qp['cooling']['cv']['min'], qp['cooling']['cv']['max'],
                        qp['cooling']['cv']['num_points'])

    w1_num = []
    w1_sim = []

    w1_rel_errors = []

    cool_probs_num = []
    cool_probs_sim = []

    total_num_time = 0
    total_sim_time = 0

    service_mean = qp['channels']['base']*qp['utilization']['base']/qp['arrival_rate']

    b = calc_moments_by_mean_and_coev(service_mean, qp['service']['cv']['base'])
    b_w = calc_moments_by_mean_and_coev(
        qp['warmup']['mean']['base'], qp['warmup']['cv']['base'])
    b_d = calc_moments_by_mean_and_coev(qp['delay']['mean']['base'], qp['delay']['cv']['base'])

    for cool_num, cool_cv in enumerate(cools):
        print(
            f"Start {cool_num + 1}/{len(cools)} with cooling cv={cool_cv:0.3f}... ")

        b_c = calc_moments_by_mean_and_coev(qp['cooling']['mean']['base'], cool_cv)

        num_results = run_calculation(
            arrival_rate=qp['arrival_rate'], num_channels=qp['channels']['base'], b=b, b_w=b_w, b_c=b_c, b_d=b_d)
        sim_results = run_simulation(
            arrival_rate=qp['arrival_rate'], num_channels=qp['channels']['base'], b=b, b_w=b_w, b_c=b_c, b_d=b_d,
            num_of_jobs=qp['jobs_per_sim'], ave_num=qp['sim_to_average'])

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

    return cools, w1_num, w1_sim, w1_rel_errors, cool_probs_sim, cool_probs_num
