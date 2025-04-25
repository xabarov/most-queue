"""
Testing the Fork-Join and Split-Join systems
"""
from most_queue.general.tables import times_print
from most_queue.rand_distribution import ErlangDistribution, H2Distribution
from most_queue.sim.queueing_systems.fork_join import ForkJoinSim
from most_queue.theory.queueing_systems.fork_join.m_m_n import ForkJoinMarkovianCalc
from most_queue.theory.queueing_systems.fork_join.split_join import SplitJoinCalc


def test_fj_sim():
    """
    Testing the Fork-Join system simulation
    """

    n = 3  # number of servers
    l = 1.0  # job arrival rate
    # number of jobs for IM. The higher this number, the higher the accuracy of simulation modeling.
    num_of_jobs = 300000

    # We will choose the initial moments of service time "b" by average and coefficient of variation
    # using the method most_queue.sim.rand_distribution.H2Distribution.get_params_by_mean_and_coev()
    b1 = 0.37  # average service time

    mu = 1.0/b1  # service rate

    qs = ForkJoinSim(n, n, False)

    # Set the input stream. The method needs to be passed distribution parameters and type of distribution. M - exponential
    qs.set_sources(l, 'M')

    # Set the service channels. The method needs to be passed distribution parameters and type of distribution.
    # H - hyperexponential second order
    qs.set_servers(mu, 'M')

    # Start the simulation
    qs.run(num_of_jobs)

    # Get a first initial moment of sojourn time
    v1_sim = qs.v[0]

    fj_calc_markov = ForkJoinMarkovianCalc(l, mu, n)

    vi_varma = fj_calc_markov.get_v1_fj_varma()
    vi_nelson_tantawi = fj_calc_markov.get_v1_fj_nelson_tantawi()

    print(f"\nAverage sojourn time in Fork-Join ({n}, {n}):")

    # Print comparision results as table with cols Method | Value.
    print("-" * 30)
    print(f"{'Method': ^15} | {'Value': ^15}")
    print("-" * 30)
    print(f"{'Sim': ^15} | {v1_sim: .6f}")
    print(f"{'Varma': ^15} | {vi_varma: .6f}")
    print(f"{'Nelson-Tantawi': ^15} | {vi_nelson_tantawi: .6f}")
    print("-" * 30)

    # Assert that the simulation result is close to the theoretical value
    assert abs(
        v1_sim - vi_varma) < 0.02, "The simulation result is not close to the theoretical value for Varma's formula"
    assert abs(v1_sim - vi_nelson_tantawi) < 0.02, "The simulation result is not close to the theoretical value for Nelson-Tantawi's formula"

    # Run Fork-Join (n, k) simulation and calculation of the average sojourn time

    k = 2

    qs = ForkJoinSim(n, k, False)

    # Set the input stream. The method needs to be passed distribution parameters and type of distribution. M - exponential
    qs.set_sources(l, 'M')

    # Set the service channels. The method needs to be passed distribution parameters and type of distribution.
    # H - hyperexponential second order
    qs.set_servers(mu, 'M')

    # Start the simulation
    qs.run(num_of_jobs)

    # Get a first initial moment of sojourn time
    v1_sim = qs.v[0]

    fj_calc_markov = ForkJoinMarkovianCalc(l, mu, n, k)

    vi_varma = fj_calc_markov.get_v1_varma_nk()
    vi_nelson_tantawi = fj_calc_markov.get_v1_fj_nelson_nk()

    print(f"\nAverage sojourn time in Fork-Join ({n}, {k}):")

    # Print comparision results as table with cols Method | Value.
    print("-" * 30)
    print(f"{'Method': ^15} | {'Value': ^15}")
    print("-" * 30)
    print(f"{'Sim': ^15} | {v1_sim: .6f}")
    print(f"{'Varma': ^15} | {vi_varma: .6f}")
    print(f"{'Nelson-Tantawi': ^15} | {vi_nelson_tantawi: .6f}")
    print("-" * 30)

    # Assert that the simulation result is close to the theoretical value
    assert abs(
        v1_sim - vi_varma) < 0.02, "The simulation result is not close to the theoretical value for Varma's formula"
    assert abs(v1_sim - vi_nelson_tantawi) < 0.02, "The simulation result is not close to the theoretical value for Nelson-Tantawi's formula"


def test_sj_sim():
    """
    Testing the calculation of a Split-Join system
        | Ryzhikov, Yu. I. Method for calculating the duration of task processing in a queueing system
        | with consideration of Split-Join processes / Yu. I. Ryzhikov, V. A. Lokhviatsky, R. S. Habarov
        | Journal of Higher Educational Institutions. Instrument Engineering. – 2019. – Vol. 62. – No. 5. – pp. 419-423. –
        | DOI 10.17586/0021-3454-2019-62-5-419-423.

    For verification, we use simulation modeling.

    """
    n = 3  # number of servers
    l = 1.0  # job arrival rate
    # number of jobs for IM. The higher this number, the higher the accuracy of simulation modeling.
    num_of_jobs = 300000

    # We will choose the initial moments of service time "b" by average and coefficient of variation
    # using the method most_queue.sim.rand_distribution.H2Distribution.get_params_by_mean_and_coev()
    b1 = 0.37  # average service time
    coev = 1.5  # coefficient of variation of service time

    # parameters of the H2 distribution [y1, mu1, mu2]
    params = H2Distribution.get_params_by_mean_and_coev(b1, coev)

    # Calculate the first four moments, we need one more than the required moments of time spent in the queueing system
    b = H2Distribution.calc_theory_moments(params, 4)

    # To verify, we use IM.
    # Create an instance of the simulation class and pass the number of servers for service
    # Simulation class  supports a Fork-Join (n, k) type queueing system. In our case, k = n
    # For specifying a Split-Join queueing system, you need to pass the third parameter True, otherwise by default - Fork-Join

    qs = ForkJoinSim(n, n, True)

    # Set the input stream. The method needs to be passed distribution parameters and type of distribution. M - exponential
    qs.set_sources(l, 'M')

    # Set the service channels. The method needs to be passed distribution parameters and type of distribution.
    # H - hyperexponential second order
    qs.set_servers(params, 'H')

    # Start the simulation
    qs.run(num_of_jobs)

    # Get a list of initial moments of sojourn time
    v_sim = qs.v

    # Calculate the initial moments of the distribution maximum using the method fj_calc.getMaxMoments.
    # The input is the number of servers and the list of initial moments

    sj_calc = SplitJoinCalc(l, n, b)
    v_ch = sj_calc.get_v()
    ro = sj_calc.get_ro()

    # Further calculation as in a regular M/G/1 queueing system with initial moments of the distribution maximum of the random variable

    print("\n")
    print("-" * 60)
    print(f'{"Split-Join Queueing System":^60s}')
    print("-" * 60)
    print(f"Coefficient of variation of service time: {coev}")
    print(f"Utilization coefficient: {ro:.3f}")
    times_print(v_sim, v_ch, False)

    # Also for a coefficient of variation < 1 (approximating the distribution with Erlang)
    coev = 0.8
    b1 = 0.5
    params = ErlangDistribution.get_params_by_mean_and_coev(b1, coev)
    b = ErlangDistribution.calc_theory_moments(params, 4)

    qs = ForkJoinSim(n, n, True)
    qs.set_sources(l, 'M')
    qs.set_servers(params, 'E')
    qs.run(num_of_jobs)
    v_sim = qs.v

    sj_calc = SplitJoinCalc(l, n, b)
    v_ch = sj_calc.get_v()
    ro = sj_calc.get_ro()

    print(f"Coefficient of variation of service time: {coev}")
    print(f"Utilization coefficient: {ro:.3f}")
    times_print(v_sim, v_ch, False)

    assert 100*abs(v_ch[0] - v_sim[0])/max(v_ch[0], v_sim[0]) < 10


if __name__ == "__main__":
    test_fj_sim()
