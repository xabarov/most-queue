"""
Test QS M/G/n queue with negative jobs and RCS discipline.
"""
import math

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc

ARRIVAL_RATE_NEGATIVE = 0.3
ARRIVAL_RATE_POSITIVE = 1.0
NUM_OF_JOBS = 300000
NUM_OF_CHANNELS = 3
UTILIZATION = 0.7
SERVICE_TIME_CV = 2.1


def test_mgn():
    """
    Test QS M/G/n queue with negative jobs and RCS discipline.
    """

    b1 = NUM_OF_CHANNELS * UTILIZATION / ARRIVAL_RATE_POSITIVE  # average service time

    b = [0.0] * 3
    alpha = 1 / (SERVICE_TIME_CV ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * (math.pow(SERVICE_TIME_CV, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    print(f'Service time moments: {b}')

    # Run simulation
    queue_sim = QsSimNegatives(
        NUM_OF_CHANNELS, NegativeServiceType.RCS)

    queue_sim.set_negative_sources(ARRIVAL_RATE_NEGATIVE, 'M')
    queue_sim.set_positive_sources(ARRIVAL_RATE_POSITIVE, 'M')
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, 'Gamma')

    queue_sim.run(NUM_OF_JOBS)

    p_sim = queue_sim.get_p()
    v_sim = queue_sim.get_v()
    v_sim_served = queue_sim.get_v_served()
    v_sim_broken = queue_sim.get_v_broken()

    w_sim = queue_sim.get_w()

    # Run calc
    queue_calc = MGnNegativeRCSCalc(
        NUM_OF_CHANNELS, ARRIVAL_RATE_POSITIVE, ARRIVAL_RATE_NEGATIVE,
        b, verbose=False, accuracy=1e-8)

    queue_calc.run()

    p_calc = queue_calc.get_p()
    v_calc = queue_calc.get_v()
    v_calc_served = queue_calc.get_v_served()
    v_calc_broken = queue_calc.get_v_broken()
    w_calc = queue_calc.get_w()

    print(f'q = {queue_calc.get_q():0.3f}')

    probs_print(p_sim, p_calc)
    times_print(v_sim, v_calc, is_w=False, header='Total sojourn time')
    times_print(v_sim_served, v_calc_served, is_w=False,
                header='Served sojourn time')
    times_print(v_sim_broken, v_calc_broken, is_w=False,
                header='Broken sojourn time')
    times_print(w_sim, w_calc)


if __name__ == "__main__":
    test_mgn()
