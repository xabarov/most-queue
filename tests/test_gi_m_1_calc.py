"""
Test of GI/M/1 queueing system calculation.
For verification, we use simulation 
"""
import numpy as np

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution, ParetoDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.gi_m_1 import GiM1

ARRIVAL_RATE = 1.0
SERVICE_TIME_AVERAGE = 0.9
ARRIVAL_CV = 1.6  # Coefficient of Variation of arrival time distribution

NUM_OF_JOBS = 300000


def test_gi_m_1():
    """
    Test of GI/M/1 queueing system calculation.
    For verification, we use simulation 
    """

    a1 = 1 / ARRIVAL_RATE    # average interval between requests
    mu = 1 / SERVICE_TIME_AVERAGE  # service intensity

    # calculation of parameters approximating Gamma-distribution for arrival times
    gamma_params = GammaDistribution.get_params_by_mean_and_coev(
        a1, ARRIVAL_CV)
    print(gamma_params)
    a = GammaDistribution.calc_theory_moments(gamma_params)

    # calculation of initial moments of time spent and waiting in the queueing system
    gm1_calc = GiM1(a, mu)
    v_num = gm1_calc.get_v()
    w_num = gm1_calc.get_w()

    # calculation of probabilities of states in the queueing system
    p_num = gm1_calc.get_p()

    # for verification, we use sim.
    # create an instance of the sim class and pass the number of service channels
    qs = QsSim(1)

    # set the input stream. The method needs to be passed parameters
    # of distribution as a list and type of distribution.
    qs.set_sources(gamma_params, "Gamma")

    # set the service channels. Parameters (in our case, the service intensity)
    # and type of distribution - M (exponential).
    qs.set_servers(mu, "M")

    # start simulation
    qs.run(NUM_OF_JOBS)

    # get the list of initial moments of time spent and waiting in the queueing system
    v_sim = qs.v
    w_sim = qs.w

    # get the distribution of probabilities of states in the queueing system
    p_sim = qs.get_p()

    # Output results
    print("\nGamma\n")

    times_print(v_sim, v_num, False)
    probs_print(p_sim, p_num)

    # Also for Pareto distribution

    pareto_params = ParetoDistribution.get_params_by_mean_and_coev(
        a1, ARRIVAL_CV)
    print(pareto_params)
    a = ParetoDistribution.calc_theory_moments(pareto_params)

    gm1_calc = GiM1(a, mu, approx_distr="Pa")
    v_num = gm1_calc.get_v()
    w_num = gm1_calc.get_w()

    # calculation of probabilities of system states
    p_num = gm1_calc.get_p()

    qs = QsSim(1)
    qs.set_sources(pareto_params, "Pa")
    qs.set_servers(mu, "M")
    qs.run(NUM_OF_JOBS)
    v_sim = qs.v
    w_sim = qs.w
    p_sim = qs.get_p()

    assert np.allclose(np.array(v_sim), np.array(v_num), rtol=30e-1)
    assert np.allclose(np.array(w_sim), np.array(w_num), rtol=30e-1)
    assert np.allclose(np.array(p_sim[:10]), np.array(p_num[:10]), rtol=1e-1)

    print("\nPareto\n")

    times_print(v_sim, v_num, False)
    probs_print(p_sim, p_num)


if __name__ == "__main__":
    test_gi_m_1()
