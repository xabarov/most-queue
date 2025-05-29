"""
Testing the GI/M/n queueing system calculation.
For verification, we use imitational modeling.
"""
from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution, ParetoDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.gi_m_n import GiMn

ARRIVAL_RATE = 1.0
SERVICE_TIME_AVERAGE = 0.9
ARRIVAL_CV = 1.6  # Coefficient of Variation of arrival time distribution
NUM_OF_CHANNELS = 4
UTILIZATION = 0.8  # system utilization coefficient
NUM_OF_JOBS = 300000


def test_gi_m_n():
    """
    Testing the GI/M/n queueing system calculation.
    For verification, we use imitational modeling.
    """

    a1 = 1.0 / ARRIVAL_RATE  # average interval between arrivals
    b1 = UTILIZATION * NUM_OF_CHANNELS / ARRIVAL_RATE  # average service time given ro
    mu = 1 / b1  # service intensity

    # calculate parameters of the approximating Gamma distribution for arrival times
    gamma_params = GammaDistribution.get_params_by_mean_and_coev(
        a1, ARRIVAL_CV)
    print(gamma_params)
    a = GammaDistribution.calc_theory_moments(gamma_params)

    # calculate initial moments of sojourn and waiting times in the queueing system

    gi_m_n_calc = GiMn(a, mu, NUM_OF_CHANNELS)
    v_num = gi_m_n_calc.get_v()
    w_num = gi_m_n_calc.get_w()

    # calculate probabilities of states in the queueing system
    p_num = gi_m_n_calc.get_p()

    # for verification, we use simulation.
    # create an instance of the Simulation class and pass the number of servers
    qs = QsSim(NUM_OF_CHANNELS)

    # set the ariival distribution paprams.
    # The method needs to be passed parameters as a list and the type of distribution.
    qs.set_sources(gamma_params, "Gamma")

    # set the service channels.
    # The method should receive parameters (in our case, the service intensity)
    # and the type of distribution - M (exponential).
    qs.set_servers(mu, "M")

    # start the simulation:
    qs.run(NUM_OF_JOBS)

    # get the list of initial moments of sojourn and waiting times in the queueing system
    v_sim = qs.v
    w_sim = qs.w

    # get the distribution of probabilities of states in the queueing system
    p_sim = qs.get_p()

    # Output results
    print("\nGamma")

    times_print(w_sim, w_num)
    times_print(v_sim, v_num, is_w=False)
    probs_print(p_sim, p_num)

    # Also for Pareto distribution
    pa_params = ParetoDistribution.get_params_by_mean_and_coev(a1, ARRIVAL_CV)
    print(pa_params)
    a = ParetoDistribution.calc_theory_moments(pa_params)
    gi_m_n_calc = GiMn(a, mu, NUM_OF_CHANNELS, approx_distr='Pa')
    v_num = gi_m_n_calc.get_v()
    w_num = gi_m_n_calc.get_w()

    p_num = gi_m_n_calc.get_p()

    qs = QsSim(NUM_OF_CHANNELS)
    qs.set_sources(pa_params, "Pa")
    qs.set_servers(mu, "M")
    qs.run(NUM_OF_JOBS)
    v_sim = qs.v
    p_sim = qs.get_p()
    w_sim = qs.w

    print("\nPareto")

    times_print(w_sim, w_num)
    times_print(v_sim, v_num, is_w=False)
    probs_print(p_sim, p_num)


if __name__ == "__main__":
    test_gi_m_n()
