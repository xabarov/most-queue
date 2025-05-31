"""
Run simulation and calculation for different service time coefficient of variation 
and plot the wait time averages for both calculation and simulation.
"""
import os

import numpy as np

from run_one_calc_vs_sim import (
    calc_moments_by_mean_and_coev,
    run_calculation,
    run_simulation,
)
from utils import calc_rel_error_percent, plot_w1, plot_w1_errors


def run_service_cv(qp, save_path: str = None):
    """
    Run simulation and calculation for different service time coefficient of variation 
    """
    cvs = np.linspace(qp['service']['cv']['min'], qp['service']['cv']['max'],
                      qp['service']['cv']['num_points'] + 1)

    w1_num = []
    w1_sim = []

    w1_rel_errors = []

    total_num_time = 0
    total_sim_time = 0

    b_w = calc_moments_by_mean_and_coev(
        qp['warmup']['mean']['base'], qp['warmup']['cv']['base'])
    b_c = calc_moments_by_mean_and_coev(
        qp['cooling']['mean']['base'], qp['cooling']['cv']['base'])
    b_d = calc_moments_by_mean_and_coev(qp['delay']['mean']['base'], qp['delay']['cv']['base'])

    service_mean = qp['channels']['base']*qp['utilization']['base']/qp['arrival_rate']

    for cv_num, cv in enumerate(cvs):
        print(
            f"Start {cv_num + 1}/{len(cvs)} with service time cv={cv:0.3f}... ")

        b = calc_moments_by_mean_and_coev(service_mean, cv)

        num_results = run_calculation(
            arrival_rate=qp['arrival_rate'], num_channels=qp['channels']['base'], b=b, b_w=b_w, b_c=b_c, b_d=b_d)
        sim_results = run_simulation(
            arrival_rate=qp['arrival_rate'], num_channels=qp['channels']['base'], b=b, b_w=b_w, b_c=b_c, b_d=b_d,
            num_of_jobs=qp['jobs_per_sim'], ave_num=qp['sim_to_average'])

        w1_num.append(num_results["w"][0])
        w1_sim.append(sim_results["w"][0])

        w1_rel_errors.append(calc_rel_error_percent(
            sim_results["w"][0], num_results["w"][0]))

        total_num_time += num_results["process_time"]
        total_sim_time += sim_results["process_time"]

    # Print process time comparison
    print(f"Total process time for num: {total_num_time:.4g}")
    print(f"Total process time for sim: {total_sim_time:.4g}")

    if save_path:
        w1_save_path = os.path.join(save_path, 'w1_vs_service_cv.png')
        plot_w1(cvs, w1_num, w1_sim, save_path=w1_save_path,
                x_label="Service time CV", is_xs_int=False)

        w1_errors_save_path = os.path.join(
            save_path, 'w1_errors_vs_service_cv.png')
        plot_w1_errors(cvs, w1_rel_errors, save_path=w1_errors_save_path,
                       x_label="Service time CV", is_xs_int=False)

    return cvs, w1_num, w1_sim, w1_rel_errors
