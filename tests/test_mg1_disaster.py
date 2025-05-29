"""
Test QS M/G/n queue with disasters.
"""
from most_queue.general.tables import times_print_with_two_numerical
from most_queue.rand_distribution import GammaDistribution, H2Distribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mg1_disasters import MG1Disasters
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc

ARRIVAL_RATE_POSITIVE = 1.0
ARRIVAL_RATE_NEGATIVE = 0.8
NUM_OF_JOBS = 300000
SERVICE_TIME_CV = 2.15
UTILIZATION = 0.7


def test_mg1():
    """
    Test QS M/G/1 queue with disasters.
    Compare theoretical and simulated moments.
    MG1 calculate moments using Pollaczek-Khintchine formula.
        Jain, Gautam, and Karl Sigman. "A Pollaczek–Khintchine formula 
        for M/G/1 queues with disasters."
        Journal of Applied Probability 33.4 (1996): 1191-1200.
    T-T is ours method (based on Takahasi-Takagi)
    """

    b1 = 1 * UTILIZATION / ARRIVAL_RATE_POSITIVE  # average service time
    b_coev = 2.15  # coefficient of variation for service time

    approximation = 'gamma'

    if approximation == 'h2':
        b_params = H2Distribution.get_params_by_mean_and_coev(b1, b_coev)
        b = H2Distribution.calc_theory_moments(b_params)
    else:
        b_params = GammaDistribution.get_params_by_mean_and_coev(b1, b_coev)
        b = GammaDistribution.calc_theory_moments(b_params)

    # Run calc
    mg1_queue_calc = MG1Disasters(
        ARRIVAL_RATE_POSITIVE, ARRIVAL_RATE_NEGATIVE, b, approximation=approximation)

    v_calc1 = mg1_queue_calc.get_v()

    mgn_queue_calc = MGnNegativeDisasterCalc(
        1, ARRIVAL_RATE_POSITIVE, ARRIVAL_RATE_NEGATIVE, b)

    mgn_queue_calc.run()

    v_calc2 = mgn_queue_calc.get_v()

    # Run simulation
    queue_sim = QsSimNegatives(
        1, NegativeServiceType.DISASTER)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, 'M')
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, 'M')

    if approximation == 'h2':
        queue_sim.set_servers(b_params, 'H')
    else:
        queue_sim.set_servers(b_params, 'Gamma')

    queue_sim.run(NUM_OF_JOBS)

    v_sim = queue_sim.get_v()

    times_print_with_two_numerical(v_sim, v_calc1, v_calc2, is_w=False,
                                   num1_header='MG1', num2_header='T-T')


if __name__ == "__main__":
    test_mg1()
