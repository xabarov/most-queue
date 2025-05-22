"""
Test QS M/G/n queue with disasters.
"""
from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution, H2Distribution
from most_queue.sim.queueing_systems.negative import NegativeServiceType, QsSimNegatives
from most_queue.theory.queueing_systems.fifo.mg1 import MG1Calculation
from most_queue.theory.queueing_systems.negative.mg1_disasters import MG1Disasters


def test_mg1():
    """
    Test QS M/G/1 queue with disasters.
    """

    l_pos = 1.0  # arrival rate of positive jobs
    l_neg = 0.3  # arrival rate of negative jobs
    n = 1
    num_of_jobs = 100_000
    ro = 0.75
    b1 = n * ro / l_pos  # average service time
    b_coev = 1.57

    params = H2Distribution.get_params_by_mean_and_coev(b1, b_coev)
    b = H2Distribution.calc_theory_moments(params)

    print(f'Service time moments: {b}')

    # Run calc
    queue_calc = MG1Disasters(l_pos, l_neg, b, approximation='h2')

    p0_calc = 1.0 -queue_calc.nu
    v_calc = queue_calc.get_v()

    # Run simulation
    queue_sim = QsSimNegatives(
        n, NegativeServiceType.DISASTER)

    queue_sim.set_negative_sources(l_neg, 'M')
    queue_sim.set_positive_sources(l_pos, 'M')
    queue_sim.set_servers(params, 'H')

    queue_sim.run(num_of_jobs)

    v_sim = queue_sim.get_v()
    w_sim = queue_sim.get_w()
    p0_sim = queue_sim.get_p()[0]

    times_print(v_sim, v_calc, is_w=False)
    print(f'p0 calc: {p0_calc:.4f}, p0 sim: {p0_sim:.4f}')
    print(f'w_sim: {w_sim}')
    
    mg1_calc = MG1Calculation(l_pos, b)
    v_calc = mg1_calc.get_v()
    w_calc = mg1_calc.get_w()
    times_print(v_sim, v_calc, is_w=False, header='v M/G/1 without disaster')
    times_print(w_sim, w_calc, is_w=True, header='w M/G/1 without disaster')


if __name__ == "__main__":
    test_mg1()
