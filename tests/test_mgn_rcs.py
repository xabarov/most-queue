"""
Test QS M/G/n queue with negative jobs and RCS discipline.
"""
import math

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.queueing_systems.negative import (
    NegativeServiceType,
    QueueingSystemSimulatorWithNegatives,
)
from most_queue.theory.queueing_systems.negative.mgn_rcs import MGnNegativeRCSCalc


def test_mgn():
    """
    Test QS M/G/n queue with negative jobs and RCS discipline.
    """

    l_pos = 1.0  # arrival rate of positive jobs
    l_neg = 0.9  # arrival rate of negative jobs
    n = 3
    num_of_jobs = 300000
    ro = 0.7
    b1 = n * ro / l_pos  # average service time
    b_coev = 1.3

    b = [0.0] * 3
    alpha = 1 / (b_coev ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * (math.pow(b_coev, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)
    
    print(f'B: {b}')

    # Create a simulator with 1 server and 1 queue
    queue_sim = QueueingSystemSimulatorWithNegatives(
        n, NegativeServiceType.RCS)

    queue_sim.set_negative_sources(l_neg, 'M')
    queue_sim.set_positive_sources(l_pos, 'M')
    gamma_params = GammaDistribution.get_params([b[0], b[1]])
    queue_sim.set_servers(gamma_params, 'Gamma')

    queue_sim.run(num_of_jobs)

    p_sim = queue_sim.get_p()
    v_sim_served = queue_sim.get_v_served()
    v_sim_breaked = queue_sim.get_v_breaked()
    
    print(f'V served {v_sim_served}, breaked {v_sim_breaked}')
    print(f'served {queue_sim.served}, breaked {queue_sim.breaked}')
    
    w_sim = queue_sim.get_w()

    queue_calc = MGnNegativeRCSCalc(n, l_pos, l_neg, b, verbose=False, accuracy=1e-8)

    queue_calc.run()

    p_calc = queue_calc.get_p()
    v_calc = queue_calc.get_v(derivate=False)
    w_calc = queue_calc.get_w(derivate=False)
    
    print(f'q = {queue_calc.get_q()}')

    probs_print(p_sim, p_calc)
    times_print(v_sim_served, v_calc, is_w=False)
    times_print(w_sim, w_calc)


if __name__ == "__main__":
    test_mgn()
