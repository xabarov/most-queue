"""
Testing the Fork-Join and Split-Join systems
"""
import os

import numpy as np
import yaml

from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution, GammaParams
from most_queue.sim.fork_join import ForkJoinSim
from most_queue.theory.fork_join.m_m_n import ForkJoinMarkovianCalc
from most_queue.theory.fork_join.split_join import SplitJoinCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params['num_of_channels'])

ARRIVAL_RATE = float(params['arrival']['rate'])

SERVICE_TIME_CV = float(params['service']['cv'])

NUM_OF_JOBS = int(params['num_of_jobs'])
UTILIZATION_FACTOR = float(params['utilization_factor'])
ERROR_MSG = params['error_msg']

MOMENTS_ATOL = float(params['moments_atol'])
MOMENTS_RTOL = float(params['moments_rtol'])

JOBS_REQUIRED = 2  # or k in (n,k) fork-join queueing system

SERVICE_TIME_AVERAGE = 0.35


def print_results_fj(v1_sim, v1_varma, v1_nelson_tantawi, k: int):
    """
    Prints the average sojourn time in Fork-Join system for different methods.
    """
    print(
        f"\nAverage sojourn time in Fork-Join ({NUM_OF_CHANNELS}, {k}):")

    # Print comparision results as table with cols Method | Value.
    print("-" * 30)
    print(f"{'Method': ^15} | {'Value': ^15}")
    print("-" * 30)
    print(f"{'Sim': ^15} | {v1_sim: .6f}")
    print(f"{'Varma': ^15} | {v1_varma: .6f}")
    print(f"{'Nelson-Tantawi': ^15} | {v1_nelson_tantawi: .6f}")
    print("-" * 30)

    # Assert that the simulation result is close to the theoretical value
    assert abs(v1_sim - v1_varma) < 0.02, ERROR_MSG
    assert abs(v1_sim - v1_nelson_tantawi) < 0.02, ERROR_MSG


def print_results_sj(coev: float, ro: float, v_sim: list, v_num: list):
    """
    Print results of Split-Join Queueing System simulation.
     Args:
         coev: Coefficient of variation of service time.
         ro: Utilization coefficient.
         v_sim: List of simulated sojourn times.
         v_num: List of theoretical sojourn times.
    """
    print("\n")
    print("-" * 60)
    print(f'{"Split-Join Queueing System":^60s}')
    print("-" * 60)
    print(
        f"Coefficient of variation of service time: {coev}")
    print(f"Utilization coefficient: {ro:.3f}")
    times_print(v_sim, v_num, False)


def run_sim_fj(k: int, mu: float):
    """
    Run Fork-Join (n,k) simulation for M/M/n systems.
    """
    qs = ForkJoinSim(NUM_OF_CHANNELS, k, False)

    # Set the input stream. The method needs to be passed distribution
    # parameters and type of distribution. M - exponential
    qs.set_sources(ARRIVAL_RATE, 'M')

    # Set the service channels.
    qs.set_servers(mu, 'M')

    # Start the simulation
    qs.run(NUM_OF_JOBS)

    return qs.v


def run_sim_sj(service_params: GammaParams):
    """
    Run Split-Join (n,n) simulation for M/Gamma/n systems.
    """
    # Create an instance of the simulation class and pass the number of servers for service
    # Simulation class  supports a Fork-Join (n, k) type queueing system.
    # For specifying a Split-Join queueing system, you need to pass the
    # third parameter True, otherwise by default - Fork-Join

    qs = ForkJoinSim(NUM_OF_CHANNELS, NUM_OF_CHANNELS, True)

    qs.set_sources(ARRIVAL_RATE, 'M')
    qs.set_servers(service_params, 'Gamma')

    # Start the simulation
    qs.run(NUM_OF_JOBS)

    # Get a list of initial moments of sojourn time
    return qs.v


def test_fj_sim():
    """
    Testing the Fork-Join system simulation
    """

    mu = 1.0/SERVICE_TIME_AVERAGE    # service rate

    v1_sim = run_sim_fj(NUM_OF_CHANNELS, mu)[0]

    fj_calc_markov = ForkJoinMarkovianCalc(ARRIVAL_RATE, mu, NUM_OF_CHANNELS)

    v1_varma = fj_calc_markov.get_v1_fj_varma()
    v1_nelson_tantawi = fj_calc_markov.get_v1_fj_nelson_tantawi()

    print_results_fj(v1_sim, v1_varma, v1_nelson_tantawi, k=NUM_OF_CHANNELS)

    # Run Fork-Join (n, k) simulation and calculation of the average sojourn time
    v1_sim = run_sim_fj(JOBS_REQUIRED, mu)[0]

    fj_calc_markov = ForkJoinMarkovianCalc(
        ARRIVAL_RATE, mu, NUM_OF_CHANNELS, JOBS_REQUIRED)

    v1_varma = fj_calc_markov.get_v1_varma_nk()
    v1_nelson_tantawi = fj_calc_markov.get_v1_fj_nelson_nk()

    print_results_fj(v1_sim, v1_varma, v1_nelson_tantawi, k=JOBS_REQUIRED)


def test_sj_sim():
    """
    Testing the calculation of a Split-Join system
        | Ryzhikov, Yu. I. Method for calculating the duration of task processing 
        | in a queueing system with consideration of Split-Join processes 
        | Yu. I. Ryzhikov, V. A. Lokhviatsky, R. S. Habarov
        | Journal of Higher Educational Institutions. Instrument Engineering. 
        – 2019. – Vol. 62. – No. 5. – pp. 419-423. –
        | DOI 10.17586/0021-3454-2019-62-5-419-423.

    For verification, we use simulation modeling.

    """

    gamma_params = GammaDistribution.get_params_by_mean_and_coev(
        SERVICE_TIME_AVERAGE, SERVICE_TIME_CV)

    b = GammaDistribution.calc_theory_moments(gamma_params)
    v_sim = run_sim_sj(gamma_params)

    sj_calc = SplitJoinCalc(ARRIVAL_RATE, NUM_OF_CHANNELS, b)
    v_num = sj_calc.get_v()
    ro = sj_calc.get_ro()

    print_results_sj(SERVICE_TIME_CV, ro, v_sim, v_num)

    assert np.allclose(np.array(v_sim[:2]), np.array(
        v_num), rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":
    test_sj_sim()
    test_fj_sim()
