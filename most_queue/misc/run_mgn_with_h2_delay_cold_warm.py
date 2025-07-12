"""
Test for M/H2/n queue with H2-warming, H2-cooling and H2-delay of the start of cooling.
Theoretical calculation is compared with simulation results.
"""
import os
import time

import numpy as np

from most_queue.misc.vacations_paper_utils import (
    calc_moments_by_mean_and_coev,
    dump_stat,
    load_stat,
    make_plot,
    print_table,
)
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.vacations import VacationQueueingSystemSimulator
from most_queue.theory.vacations.mgn_with_h2_delay_cold_warm import (
    MGnH2ServingColdWarmDelay,
)


def get_sim_stat(stat, n, l, buff, b, b_c, b_w, b_d, num_of_jobs, p_limit, sim_ave):
    """
    Get simulation statistics for an M/H2/n queue with H2-warming, 
    H2-cooling and H2-delay of the start of cooling.

    :param stat: statistic object
    :param n: number of servers
    :param l: arrival rate
    :param buff: buffer size
    :param b: initial moments of service time
    :param b_c: initial moments of cooling time
    :param b_w: initial moments of warming time
    :param b_d: initial moments of delay of the start of cooling time
    :param num_of_jobs: number of jobs to simulate
    :param p_limit: limit for the probability of state
    :param sim_ave: number of simulations to average the results
    :return: tuple of lists with simulated statistics (w_sim_mass, p_sim_mass, 
    warm_prob_sim_mass, cold_prob_sim_mass,
    """
    im_start = time.process_time()
    w_sim_mass = []
    p_sim_mass = []
    warm_prob_sim_mass = []
    cold_prob_sim_mass = []
    cold_delay_prob_sim_mass = []

    for j in range(sim_ave):
        print(f"\nStart {j + 1}/{sim_ave} simulation")
        sim = VacationQueueingSystemSimulator(n, buffer=buff)
        sim.set_sources(l, 'M')

        gamma_params = GammaDistribution.get_params(b)
        gamma_params_warm = GammaDistribution.get_params(b_w)
        gamma_params_cold = GammaDistribution.get_params(b_c)
        gamma_params_cold_delay = GammaDistribution.get_params(b_d)

        sim.set_servers(gamma_params, 'Gamma')
        sim.set_warm(gamma_params_warm, 'Gamma')
        sim.set_cold(gamma_params_cold, 'Gamma')
        sim.set_cold_delay(gamma_params_cold_delay, 'Gamma')

        sim.run(num_of_jobs)

        w_sim_mass.append(sim.w)
        p_sim_mass.append(sim.get_p()[:p_limit])
        warm_prob_sim_mass.append(sim.get_warmup_prob())
        cold_prob_sim_mass.append(sim.get_cold_prob())
        cold_delay_prob_sim_mass.append(sim.get_cold_delay_prob())

    # average all sim data

    w_ave = [0, 0, 0]
    p_ave = [0.0] * p_limit
    for k in range(3):
        for j in range(sim_ave):
            w_ave[k] += w_sim_mass[j][k]
        w_ave[k] /= sim_ave

    for k in range(p_limit):
        for j in range(sim_ave):
            p_ave[k] += p_sim_mass[j][k]
        p_ave[k] /= sim_ave

    im_time = time.process_time() - im_start

    stat["w_sim"] = w_ave
    stat["p_sim"] = p_ave
    stat["sim_time"] = im_time

    stat["sim_warm_prob"] = np.array(warm_prob_sim_mass).mean()

    stat["sim_cold_prob"] = np.array(cold_prob_sim_mass).mean()

    stat["sim_cold_delay_prob"] = np.array(cold_delay_prob_sim_mass).mean()


def get_tt_stat(stat, n, l, buff, b, b_c, b_w, b_d, p_limit, w_pls_dt, stable_w_pls, verbose=False):
    """
    Get statistics from Takahasi-Takami method.

    :param stat: statistic object
    :param n: number of servers
    :param l: arrival rate
    :param buff: buffer size (None if infinite)
    :param b: initial moments of service time
    :param b_c: initial moments of cooling time
    :param b_w: initial moments of warming time
    :param b_d: initial moments of delay time before cooling starts
    :param p_limit: limit for the sum of probabilities in the Takahasi-Takami method
    :param w_pls_dt: step for the Laplace-Stieltjes transform calculation
    :param stable_w_pls: flag for using a stable version 
      of the Laplace-Stieltjes transform calculation
    :param verbose: flag for printing debug information
    :return: None
    """
    tt_start = time.process_time()
    tt = MGnH2ServingColdWarmDelay(l, b, b_w, b_c, b_d, n,
                                   buffer=buff, verbose=verbose, w_pls_dt=w_pls_dt, stable_w_pls=stable_w_pls)

    tt.run()
    p_tt = tt.get_p()
    w_tt = tt.get_w()  # .get_w() -> wait times

    tt_time = time.process_time() - tt_start

    stat["w_tt"] = w_tt
    stat["tt_time"] = tt_time
    stat["p_tt"] = p_tt[:p_limit]
    stat["tt_num_of_iter"] = tt.num_of_iter_

    stat["tt_warm_prob"] = tt.get_warmup_prob()
    stat["tt_cold_prob"] = tt.get_cold_prob()
    stat["tt_cold_delay_prob"] = tt.get_cold_delay_prob()


def run_ro(b1_service, coev_service,
           b1_warm, coev_warm,
           b1_cold, coev_cold,
           b1_cold_delay, coev_cold_delay,
           n=1, num_of_jobs=300000,
           num_of_roes=12, min_ro=0.1, max_ro=0.9,
           p_limit=20, w_pls_dt=1e-3, stable_w_pls=False, sim_ave=3,
           verbose=False):
    """
    Run a series of simulations and theoretical calculations for an M/H2/n queue with H2-warming,
    H2-cooling and H2-delay of the start of cooling depending on load factor (rho).
    Parameters:
    ----------
    b1_service: mean service time
    coev_service: service time coefficient of variation

    b1_warm: setup (or "warm-up ") mean time
    coev_warm: warm-up coefficient of variation

    b1_cold: vacation  (or "cooling") mean time
    coev_cold: vacation  (or "cooling") coefficient of variation

    b1_cold_delay: average cooling start delay time
    coev_cold_delay: coefficient of variation of cooling start delay

    n: number of channels
    num_of_jobs - number of jobs for the simulation model

    num_of_roes - number of utilization factors
    min_ro - min value of utilization factor
    max_ro - max value of utilization factor

    p_limit - max number of probabilities

    w_pls_dt -  some variable to stabilize derivative of the Laplace-Stieltjes transform
                for the waiting time initial moments calculation

    stable_w_pls -  if True the algorithm try to fit w_pls_dt value
                    taking into account the values of transition intensities

    sim_ave - number of runs of the simulation model to average values (reduce variance)

    verbose - is it necessary to display related information
    """

    # ro = l*b1/n Будем подбирать l от ro
    roes = np.linspace(min_ro, max_ro, num_of_roes)

    experiment_stats = []

    for ro_num, ro in enumerate(roes):
        print(f"Start {ro_num + 1}/{len(roes)} with ro={ro:0.3f}... ")

        stat = {}
        l = n * ro / b1_service

        stat["l"] = l
        stat["ro"] = ro
        stat["n"] = n

        b = calc_moments_by_mean_and_coev(b1_service, coev_service)
        b_w = calc_moments_by_mean_and_coev(b1_warm, coev_warm)
        b_c = calc_moments_by_mean_and_coev(b1_cold, coev_cold)
        b_d = calc_moments_by_mean_and_coev(b1_cold_delay, coev_cold_delay)

        stat["b"] = b
        stat["coev_service"] = coev_service

        stat["b_w"] = b_w
        stat["coev_warm"] = coev_warm

        stat["b_c"] = b_c
        stat["coev_cold"] = coev_cold

        stat["b_d"] = b_d
        stat["coev_cold_delay"] = coev_cold_delay

        get_tt_stat(stat, n, l, None, b, b_c, b_w, b_d, p_limit,
                    w_pls_dt, stable_w_pls, verbose=verbose)

        get_sim_stat(stat, n, l, None, b, b_c, b_w,
                     b_d, num_of_jobs, p_limit, sim_ave)

        experiment_stats.append(stat)

    return experiment_stats


def run_n(b1_service, coev_service,
          b1_warm, coev_warm,
          b1_cold, coev_cold,
          b1_cold_delay, coev_cold_delay,
          num_of_jobs=300000,
          ro=0.7, n_min=1, n_max=30,
          p_limit=20, w_pls_dt=1e-3, stable_w_pls=False, sim_ave=3,
          verbose=False):
    """
    Run a series of simulations and theoretical calculations for an M/H2/n queue with H2-warming,
    H2-cooling and H2-delay of the start of cooling depending on the number of servers.
    Parameters:
    ----------

    b1_service: mean service time
    coev_service: service time coefficient of variation

    b1_warm: setup (or "warm-up ") mean time
    coev_warm: warm-up coefficient of variation

    b1_cold: vacation  (or "cooling") mean time
    coev_cold: vacation  (or "cooling") coefficient of variation

    b1_cold_delay: average cooling start delay time
    coev_cold_delay: coefficient of variation of cooling start delay

    ro - QS utilization factor

    num_of_jobs - number of jobs for the simulation model

    n_min: min value of number of channels
    n_max: max value of number of channels

    p_limit - max number of probabilities

    w_pls_dt -  some variable to stabilize derivative of the Laplace-Stieltjes transform
                for the waiting time initial moments calculation

    stable_w_pls -  if True the algorithm try to fit w_pls_dt value
                    taking into account the values of transition intensities

    sim_ave - number of runs of the simulation model to average values (reduce variance)

    verbose - is it necessary to display related information
    """

    # ro = l*b1/n Будем подбирать l от ro
    ns = [n for n in range(n_min, n_max + 1)]

    experiment_stats = []

    for n in ns:
        print(f"Start {n}/{len(ns)}... ")

        stat = {}
        l = n * ro / b1_service

        stat["l"] = l
        stat["ro"] = ro
        stat["n"] = n

        b = calc_moments_by_mean_and_coev(b1_service, coev_service)
        b_w = calc_moments_by_mean_and_coev(b1_warm, coev_warm)
        b_c = calc_moments_by_mean_and_coev(b1_cold, coev_cold)
        b_d = calc_moments_by_mean_and_coev(b1_cold_delay, coev_cold_delay)

        stat["b"] = b
        stat["coev_service"] = coev_service

        stat["b_w"] = b_w
        stat["coev_warm"] = coev_warm

        stat["b_c"] = b_c
        stat["coev_cold"] = coev_cold

        stat["b_d"] = b_d
        stat["coev_cold_delay"] = coev_cold_delay

        get_tt_stat(stat, n, l, None, b, b_c, b_w, b_d, p_limit,
                    w_pls_dt, stable_w_pls, verbose=verbose)

        get_sim_stat(stat, n, l, None, b, b_c, b_w,
                     b_d, num_of_jobs, p_limit, sim_ave)

        experiment_stats.append(stat)

    return experiment_stats


def run_delay_mean(b1_service, coev_service,
                   b1_warm, coev_warm,
                   b1_cold, coev_cold,
                   coev_cold_delay,
                   n=1, num_of_jobs=300000, ro=0.7,
                   num_of_delays=12, min_delay=0.1, max_delay=10,
                   p_limit=20, w_pls_dt=1e-3, stable_w_pls=False, sim_ave=3,
                   verbose=False):
    """
    Run a series of simulations and theoretical calculations for an M/H2/n queue with H2-warming,
    H2-cooling and H2-delay of the start of cooling depending on mean delay time.
    Parameters:

    b1_service: mean service time
    coev_service: service time coefficient of variation

    b1_warm: setup (or "warm-up ") mean time
    coev_warm: warm-up coefficient of variation

    b1_cold: vacation  (or "cooling") mean time
    coev_cold: vacation  (or "cooling") coefficient of variation

    coev_cold_delay: coefficient of variation of cooling start delay

    n - number of channels
    num_of_jobs - number of jobs for the simulation model
    ro - QS utilization factor

    num_of_delays - number of cooling delays times
    min_delay - min value of cooling delay
    max_delay - max value of cooling delay

    p_limit - max number of probabilities

    w_pls_dt -  some variable to stabilize derivative of the Laplace-Stieltjes transform
                for the waiting time initial moments calculation

    stable_w_pls -  if True the algorithm try to fit w_pls_dt value
                    taking into account the values of transition intensities

    sim_ave - number of runs of the simulation model to average values (reduce variance)

    verbose - is it necessary to display related information
    """

    ds = np.linspace(min_delay, max_delay, num_of_delays)

    experiment_stats = []

    for d_num, d in enumerate(ds):
        print(f"Start {d_num + 1}/{len(ds)} with delta={d:0.3f}... ")
        stat = {}
        l = n * ro / b1_service

        stat["l"] = l
        stat["ro"] = ro
        stat["n"] = n

        b = calc_moments_by_mean_and_coev(b1_service, coev_service)
        b_w = calc_moments_by_mean_and_coev(b1_warm, coev_warm)
        b_c = calc_moments_by_mean_and_coev(b1_cold, coev_cold)
        b_d = calc_moments_by_mean_and_coev(d, coev_cold_delay)

        stat["b"] = b
        stat["coev_service"] = coev_service

        stat["b_w"] = b_w
        stat["coev_warm"] = coev_warm

        stat["b_c"] = b_c
        stat["coev_cold"] = coev_cold

        stat["b_d"] = b_d
        stat["coev_cold_delay"] = coev_cold_delay

        get_tt_stat(stat, n, l, None, b, b_c, b_w, b_d, p_limit,
                    w_pls_dt, stable_w_pls, verbose=verbose)
        get_sim_stat(stat, n, l, None, b, b_c, b_w,
                     b_d, num_of_jobs, p_limit, sim_ave)

        experiment_stats.append(stat)

    return experiment_stats


def test_all():
    """
    Runs all tests for the M/H2/n queue with H2-warming, 
    H2-cooling and H2-delay of the start of cooling.
    """

    n = 3
    ro = 0.7

    # if results directory does not exist, create it
    results_path = os.path.join(os.path.dirname(__file__), 'results')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    ro_dir = os.path.join(results_path, "ro_test")
    ro_json_filename = os.path.join(ro_dir, f"n_{n}.json")

    if not os.path.exists(ro_json_filename):

        if not os.path.exists(ro_dir):
            os.makedirs(ro_dir)

        ro_stat = run_ro(b1_service=10.0, coev_service=1.2,
                         b1_warm=3.1, coev_warm=0.87,
                         b1_cold=4.1, coev_cold=1.1,
                         b1_cold_delay=3.71, coev_cold_delay=1.2,
                         n=n, num_of_jobs=300000,
                         num_of_roes=10, min_ro=0.1, max_ro=0.9, w_pls_dt=1e-3,
                         stable_w_pls=True, sim_ave=1)

        dump_stat(ro_stat, save_name=ro_json_filename)

    else:
        ro_stat = load_stat(ro_json_filename)

    print_table(ro_stat)
    # make_plot(ro_stat, param_name='ro', mode='abs')
    n_dir = os.path.join(results_path, "n_test")
    n_json_filename = os.path.join(n_dir, f"ro_{ro:0.3f}.json")

    if not os.path.exists(n_json_filename):

        if not os.path.exists(n_dir):
            os.makedirs(n_dir)

        n_stat = run_n(b1_service=10.0, coev_service=1.2,
                       b1_warm=3.1, coev_warm=0.87,
                       b1_cold=4.1, coev_cold=1.1,
                       b1_cold_delay=3.71, coev_cold_delay=1.2,
                       num_of_jobs=300000,
                       n_min=1, n_max=10, ro=ro, w_pls_dt=1e-3,
                       stable_w_pls=True, sim_ave=1)

        dump_stat(n_stat, save_name=n_json_filename)

    else:

        n_stat = load_stat(n_json_filename)

    print_table(n_stat)
    # make_plot(n_stat, param_name='n', mode='abs')

    delay_dir = os.path.join(results_path, "delay_mean_test")

    delay_json_filename = os.path.join(f"n_{n}_ro_{ro}.json")

    if not os.path.exists(delay_json_filename):

        if not os.path.exists(delay_dir):
            os.makedirs(delay_dir)

        delay_stat = run_delay_mean(b1_service=10.0, coev_service=1.2,
                                    b1_warm=3.1, coev_warm=0.87,
                                    b1_cold=4.1, coev_cold=1.1, ro=ro,
                                    coev_cold_delay=1.2,
                                    n=n, num_of_jobs=1000000,
                                    num_of_delays=10, min_delay=0.1, max_delay=10,
                                    w_pls_dt=1e-3, stable_w_pls=True, sim_ave=1)

        dump_stat(delay_stat, save_name=delay_json_filename)

    else:

        delay_stat = load_stat(delay_json_filename)

    print_table(delay_stat)
    # make_plot(delay_stat, param_name='delay_mean', mode='abs')


if __name__ == "__main__":

    # test_all()
    UTILIZATION = 0.7
    N_STAT = run_n(b1_service=10.0, coev_service=1.2,
                   b1_warm=3.1, coev_warm=0.87,
                   b1_cold=4.1, coev_cold=1.1,
                   b1_cold_delay=3.71, coev_cold_delay=1.2,
                   num_of_jobs=300000,
                   n_min=1, n_max=10, ro=UTILIZATION, w_pls_dt=1e-3,
                   stable_w_pls=True, sim_ave=1)

    print_table(N_STAT)
    make_plot(N_STAT, param_name='n', mode='abs',
              save_path="tests/vacations_ro_0.7.png")
