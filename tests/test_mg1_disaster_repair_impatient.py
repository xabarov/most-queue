import math

from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution, H2Distribution
from most_queue.sim.queueing_systems.disaster_repairs_impatient import \
    QsSimNegativeImpatience
from most_queue.theory.queueing_systems.negative.mg1_disaster_repair_impatience import MG1DisasterRepairImpatienceCalc


def calc_service_moments(utilization_factor: float,
                         service_time_variation_coef: float,
                         l_pos: float):
    """
    Gamma service time moments calculation.
    """
    b1 = 1 * utilization_factor / l_pos  # average service time

    b = [0.0] * 3
    alpha = 1 / (service_time_variation_coef ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * \
        (math.pow(service_time_variation_coef, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    return b


def calc_repeair_moments(mean, cv):
    """
    Gamma repair time moments calculation.
    """
    b1 = mean

    b = [0.0] * 3
    alpha = 1 / (cv ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * \
        (math.pow(cv, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    return b


def test_mg1_h2_disaster_repair_impatient():
    """
    Test the  M/H2/1 queueing systems with disasters, repairs and impatience.
    """

    num_of_jobs = 100_000
    l_neg = 0.9
    l_pos = 1.0
    ro = 0.7
    n = 1
    b_coev = 1.2

    imaptience_exp_alpha = 0.3

    repair_time_mean = 2.0
    repair_time_cv = 1.5

    repair_moments = calc_repeair_moments(repair_time_mean, repair_time_cv)

    b = calc_service_moments(ro, b_coev, l_pos)

    h2_params = H2Distribution.get_params(b)
    h2_params_repair = H2Distribution.get_params(repair_moments)

    # # Run simulation
    queue_sim = QsSimNegativeImpatience(n)

    queue_sim.set_negative_sources(l_neg, 'M')
    queue_sim.set_positive_sources(l_pos, 'M')
    queue_sim.set_leave_time(imaptience_exp_alpha, 'M')
    
    queue_sim.set_servers(h2_params, 'H')
    queue_sim.set_repair_time(h2_params_repair, 'H')

    queue_sim.run(num_of_jobs)

    v1_sim_up = queue_sim.get_v_up()[0]
    v1_sim_down = queue_sim.get_v_down()[0]

    print(f'v1_sim_down: {v1_sim_down:0.4f}')
    print(f'v1_sim_up: {v1_sim_up:0.4f}')

    mg1_calc = MG1DisasterRepairImpatienceCalc(
        l_pos, l_neg, b, repair_moments,
        impatience_rate=imaptience_exp_alpha, approx_dist='h2')

    # print(f'L0: {mg1_calc.calc_ave_jobs_in_down_state():0.2f}')
    # print(f'L1: {mg1_calc.calc_ave_jobs_in_up_state():0.2f}')

    print(f'v1_calc_down: {mg1_calc.calc_v1_in_down_state():0.4f}')
    print(f'v1_calc_up: {mg1_calc.calc_v1_in_up_state():0.4f}')


if __name__ == "__main__":
    test_mg1_h2_disaster_repair_impatient()
