"""
Test QS M/G/n queue with disasters.
"""
from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mg1_disasters import MG1Disasters


def test_mg1():
    """
    Test QS M/G/1 queue with disasters.
    """

    l_pos = 1.0  # arrival rate of positive jobs
    l_neg = 0.2  # arrival rate of negative jobs
    n = 1
    num_of_jobs = 300_000
    ro = 0.7
    b1 = n * ro / l_pos  # average service time
    b_coev = 1.57

    b_params = GammaDistribution.get_params_by_mean_and_coev(b1, b_coev)
    b = GammaDistribution.calc_theory_moments(b_params)

    # Run calc
    queue_calc = MG1Disasters(l_pos, l_neg, b, approximation='gamma')

    v_calc = queue_calc.get_v()

    # Run simulation
    queue_sim = QsSimNegatives(
        n, NegativeServiceType.DISASTER)

    queue_sim.set_negative_sources(l_neg, 'M')
    queue_sim.set_positive_sources(l_pos, 'M')
    queue_sim.set_servers(b_params, 'Gamma')

    queue_sim.run(num_of_jobs)

    v_sim = queue_sim.get_v()

    times_print(v_sim, v_calc, is_w=False)


if __name__ == "__main__":
    test_mg1()
