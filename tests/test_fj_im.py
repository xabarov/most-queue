"""
Testing the Fork-Join and Split-Join systems
"""
import numpy as np

from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution, GammaParams
from most_queue.sim.fork_join import ForkJoinSim
from most_queue.theory.fork_join.m_m_n import ForkJoinMarkovianCalc
from most_queue.theory.fork_join.split_join import SplitJoinCalc

NUM_OF_SERVERS = 3  # or n in (n,k) fork-join queueing system
JOBS_REQUIRED = 2  # or k in (n,k) fork-join queueing system

NUM_OF_JOBS = 300000
ARRIVAL_RATE = 1.0
SERVICE_TIME_AVERAGE = 0.35
# coefficient of variation for service time distributions
SERVICE_TIME_CV = [0.8, 1.2]


def print_results_fj(v1_sim, v1_varma, v1_nelson_tantawi, k: int):
    """
    Prints the average sojourn time in Fork-Join system for different methods.
    """
    print(
        f"\nAverage sojourn time in Fork-Join ({NUM_OF_SERVERS}, {k}):")

    # Print comparision results as table with cols Method | Value.
    print("-" * 30)
    print(f"{'Method': ^15} | {'Value': ^15}")
    print("-" * 30)
    print(f"{'Sim': ^15} | {v1_sim: .6f}")
    print(f"{'Varma': ^15} | {v1_varma: .6f}")
    print(f"{'Nelson-Tantawi': ^15} | {v1_nelson_tantawi: .6f}")
    print("-" * 30)

    # Assert that the simulation result is close to the theoretical value
    prefix = "The simulation result is not close to the theoretical value for "
    assert abs(v1_sim - v1_varma) < 0.02, prefix + "Varma's formula"
    assert abs(v1_sim - v1_nelson_tantawi) < 0.02, prefix + \
        "Nelson-Tantawi's formula"


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
    qs = ForkJoinSim(NUM_OF_SERVERS, k, False)

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

    qs = ForkJoinSim(NUM_OF_SERVERS, NUM_OF_SERVERS, True)

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

    v1_sim = run_sim_fj(NUM_OF_SERVERS, mu)[0]

    fj_calc_markov = ForkJoinMarkovianCalc(ARRIVAL_RATE, mu, NUM_OF_SERVERS)

    v1_varma = fj_calc_markov.get_v1_fj_varma()
    v1_nelson_tantawi = fj_calc_markov.get_v1_fj_nelson_tantawi()

    print_results_fj(v1_sim, v1_varma, v1_nelson_tantawi, k=NUM_OF_SERVERS)

    # Run Fork-Join (n, k) simulation and calculation of the average sojourn time
    v1_sim = run_sim_fj(JOBS_REQUIRED, mu)[0]

    fj_calc_markov = ForkJoinMarkovianCalc(
        ARRIVAL_RATE, mu, NUM_OF_SERVERS, JOBS_REQUIRED)

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
    for coev in SERVICE_TIME_CV:
        params = GammaDistribution.get_params_by_mean_and_coev(
            SERVICE_TIME_AVERAGE, coev)

        b = GammaDistribution.calc_theory_moments(params)
        v_sim = run_sim_sj(params)

        sj_calc = SplitJoinCalc(ARRIVAL_RATE, NUM_OF_SERVERS, b)
        v_num = sj_calc.get_v()
        ro = sj_calc.get_ro()

        print_results_sj(coev, ro, v_sim, v_num)

        assert np.allclose(np.array(v_sim[:2]), np.array(v_num), rtol=3)


if __name__ == "__main__":
    test_sj_sim()
    test_fj_sim()
