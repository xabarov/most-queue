"""
Test of GI/M/1 queueing system calculation.
For verification, we use simulation 
"""
import numpy as np

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution, ParetoDistribution
from most_queue.sim.queueing_systems.fifo import QueueingSystemSimulator
from most_queue.theory.queueing_systems.fifo.gi_m_1 import GiM1


def test_gi_m_1():
    """
    Test of GI/M/1 queueing system calculation.
    For verification, we use simulation 
    """

    l = 1  # input intensity of the incoming stream
    a1 = 1 / l  # average interval between requests
    b1 = 0.9  # average service time
    mu = 1 / b1  # service intensity
    a_coev = 1.6  # coefficient of variation in the input stream
    num_of_jobs = 300000  # number of jobs for IM. The higher, the higher the accuracy of sim

    # calculation of parameters approximating Gamma-distribution for the incoming stream based on the given average and coefficient of variation
    gamma_params = GammaDistribution.get_params_by_mean_and_coev(a1, a_coev)
    print(gamma_params)
    a = GammaDistribution.calc_theory_moments(gamma_params)

    # calculation of initial moments of time spent and waiting in the queueing system
    gm1_calc = GiM1(a, mu)
    v_ch = gm1_calc.get_v()
    w_ch = gm1_calc.get_w()

    # calculation of probabilities of states in the queueing system
    p_ch = gm1_calc.get_p()

    # for verification, we use sim.
    # create an instance of the sim class and pass the number of service channels
    qs = QueueingSystemSimulator(1)

    # set the input stream. The method needs to be passed parameters of distribution as a list and type of distribution.
    qs.set_sources(gamma_params, "Gamma")

    # set the service channels. Parameters (in our case, the service intensity)
    # and type of distribution - M (exponential).
    qs.set_servers(mu, "M")

    # start IM:
    qs.run(num_of_jobs)

    # get the list of initial moments of time spent and waiting in the queueing system
    v_sim = qs.v
    w_sim = qs.w

    # get the distribution of probabilities of states in the queueing system
    p_sim = qs.get_p()

    # Output results
    print("\nGamma\n")

    times_print(v_sim, v_ch, False)
    probs_print(p_sim, p_ch)

    # Also for Pareto distribution

    pareto_params = ParetoDistribution.get_params_by_mean_and_coev(a1, a_coev)
    print(pareto_params)
    a = ParetoDistribution.calc_theory_moments(pareto_params)

    gm1_calc = GiM1(a, mu, approx_distr="Pa")
    v_ch = gm1_calc.get_v()
    w_ch = gm1_calc.get_w()

    # calculation of probabilities of system states
    p_ch = gm1_calc.get_p()

    qs = QueueingSystemSimulator(1)
    qs.set_sources(pareto_params, "Pa")
    qs.set_servers(mu, "M")
    qs.run(num_of_jobs)
    v_sim = qs.v
    w_sim = qs.w
    p_sim = qs.get_p()

    assert np.allclose(np.array(v_sim), np.array(v_ch), rtol=30e-1)
    assert np.allclose(np.array(w_sim), np.array(w_ch), rtol=30e-1)
    assert np.allclose(np.array(p_sim[:10]), np.array(p_ch[:10]), rtol=1e-1)

    print("\nPareto\n")

    times_print(v_sim, v_ch, False)
    probs_print(p_sim, p_ch)


if __name__ == "__main__":
    test_gi_m_1()
