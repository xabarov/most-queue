"""
Testing the GI/M/n queueing system calculation.
For verification, we use imitational modeling.
"""
from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution, ParetoDistribution
from most_queue.sim.queueing_systems.base import QsSim
from most_queue.theory.queueing_systems.fifo.gi_m_n import GiMn


def test_gi_m_n():
    """
    Testing the GI/M/n queueing system calculation.
    For verification, we use imitational modeling.
    """

    l = 1.0  # arrival intensity
    a1 = 1.0 / l  # average interval between arrivals
    n = 4   # number of servers
    ro = 0.8  # system load coefficient
    b1 = ro * n / l  # average service time given ro
    mu = 1 / b1  # service intensity
    a_coev = 1.6  # coefficient of variation of arrival times

    # number of jobs for simulation. The higher, the more accurate the simulation
    num_of_jobs = 300000

    # calculate parameters of the approximating Gamma distribution for the input flow given the mean and coefficient of variation
    gamma_params = GammaDistribution.get_params_by_mean_and_coev(a1, a_coev)
    print(gamma_params)
    a = GammaDistribution.calc_theory_moments(gamma_params)

    # calculate initial moments of soujourn and waiting times in the queueing system

    gi_m_n_calc = GiMn(a, mu, n)
    v_ch = gi_m_n_calc.get_v()
    w_ch = gi_m_n_calc.get_w()

    # calculate probabilities of states in the queueing system
    p_ch = gi_m_n_calc.get_p()

    # for verification, we use simulation.
    # create an instance of the Simulation class and pass the number of servers
    qs = QsSim(n)

    # set the ariival distribution paprams.
    # The method needs to be passed parameters as a list and the type of distribution.
    qs.set_sources(gamma_params, "Gamma")

    # set the service channels.
    # The method should receive parameters (in our case, the service intensity)
    # and the type of distribution - M (exponential).
    qs.set_servers(mu, "M")

    # start the simulation:
    qs.run(num_of_jobs)

    # get the list of initial moments of soujourn and waiting times in the queueing system
    v_sim = qs.v
    w_sim = qs.w

    # get the distribution of probabilities of states in the queueing system
    p_sim = qs.get_p()

    # Output results
    print("\nGamma")

    times_print(w_sim, w_ch)
    times_print(v_sim, v_ch, is_w=False)
    probs_print(p_sim, p_ch)

    # Also for Pareto distribution
    pa_params = ParetoDistribution.get_params_by_mean_and_coev(a1, a_coev)
    print(pa_params)
    a = ParetoDistribution.calc_theory_moments(pa_params)
    gi_m_n_calc = GiMn(a, mu, n, approx_distr='Pa')
    v_ch = gi_m_n_calc.get_v()
    w_ch = gi_m_n_calc.get_w()

    p_ch = gi_m_n_calc.get_p()

    qs = QsSim(n)
    qs.set_sources(pa_params, "Pa")
    qs.set_servers(mu, "M")
    qs.run(num_of_jobs)
    v_sim = qs.v
    p_sim = qs.get_p()
    w_sim = qs.w

    print("\nPareto")

    times_print(w_sim, w_ch)
    times_print(v_sim, v_ch, is_w=False)
    probs_print(p_sim, p_ch)


if __name__ == "__main__":
    test_gi_m_n()
