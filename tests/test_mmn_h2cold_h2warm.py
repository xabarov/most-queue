"""
Test function to compare results of the simulation and Takahashi-Takami (TT) algorithm
for an MMn queueing system with H2 cold and warm-up phases.
"""
import time

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.vacations import VacationQueueingSystemSimulator
from most_queue.theory.vacations.mmn_with_h2_cold_and_h2_warmup import MMnHyperExpWarmAndCold


def calculate_gamma_moments(mean, cv):
    """
    Helper function to calculate Gamma distribution parameters.
    """
    alpha = 1 / (cv ** 2)
    b1 = mean
    b2 = (b1 ** 2) * (cv ** 2 + 1)
    b3 = b2 * b1 * (1 + 2 / alpha)

    return [b1, b2, b3]


ARRIVAL_RATE = 1.0
NUM_OF_JOBS = 300000
NUM_OF_CHANNELS = 3
UTILIZATION = 0.7

WARMUP_CV = 1.2  # CV for warm-up phase
COLD_CV = 1.8  # CV for cold phase
WARMUP_TIME_PROPORTION = 0.3  # percentage of mean service time for warm-up phase
COLD_TIME_PROPORTION = 0.2  # percentage of mean service time for cold phase


def test_mmn_h2cold_h2_warm():
    """
    Test function to compare results of the Implicit Method (IM) and Takahashi-Takami (TT) algorithm
    for an MMn queueing system with H2 cold and warm-up phases.
    """

    b1 = NUM_OF_CHANNELS * UTILIZATION / ARRIVAL_RATE
    mean_warmup_time = b1 * WARMUP_TIME_PROPORTION
    mean_cold_time = b1 * COLD_TIME_PROPORTION

    # Initialize the Vacation Queueing System Simulator
    simulator = VacationQueueingSystemSimulator(NUM_OF_CHANNELS, buffer=None)

    service_rate = 1.0 / b1
    simulator.set_servers(service_rate, 'M')

    # Set warm-up phase parameters
    b_w = calculate_gamma_moments(mean_warmup_time, WARMUP_CV)
    warmup_params = GammaDistribution.get_params(b_w)
    simulator.set_warm(warmup_params, 'Gamma')

    # Set cold phase parameters
    b_c = calculate_gamma_moments(mean_cold_time, COLD_CV)
    cold_params = GammaDistribution.get_params(b_c)
    simulator.set_cold(cold_params, 'Gamma')

    # Configure the simulator
    simulator.set_sources(ARRIVAL_RATE, 'M')

    # Run simulations
    im_start_time = time.process_time()
    simulator.run(NUM_OF_JOBS)
    im_execution_time = time.process_time() - im_start_time

    tt_start_time = time.process_time()
    tt = MMnHyperExpWarmAndCold(ARRIVAL_RATE, service_rate, b_w,
                           b_c, NUM_OF_CHANNELS, buffer=None, accuracy=1e-14)
    tt.run()
    tt_execution_time = time.process_time() - tt_start_time

    num_of_iter = tt.num_of_iter_

    print('warms starts', simulator.warm_phase.starts_times)
    print('warms after cold starts', simulator.warm_after_cold_starts)
    print('cold starts', simulator.cold_phase.starts_times)
    print("zero wait arrivals num", simulator.zero_wait_arrivals_num)

    print(f"\nComparison of results calculated using the Takahashi-Takami method and simulation.\n"
          f"Sim - M/Gamma/{{{NUM_OF_CHANNELS:2d}}} with Gamma warming\n"
          f"Takahashi-Takami - M/M/{{{NUM_OF_CHANNELS:2d}}} with H2-warming and H2-cooling "
          f"with complex parameters\n"
          f"Utilization coefficient: {UTILIZATION:.2f}")
    print(f'Variation coefficient of warming time {WARMUP_CV:.3f}')
    print(f'Variation coefficient of cooling time {COLD_CV:.3f}')
    print(
        f"Number of iterations in the Takahashi-Takami algorithm: {num_of_iter:4d}")
    print(
        f"Probability of being in the warming state\n"
        f"\tSim: {simulator.get_warmup_prob():.3f}\n"
        f"\tCalc: {tt.get_warmup_prob():.3f}")
    print(
        f"Probability of being in the cooling state\n"
        f"\tSim: {simulator.get_cold_prob():.3f}\n"
        f"\tCalc: {tt.get_cold_prob():.3f}")
    print(
        f"Execution time of the Takahashi-Takami algorithm: {tt_execution_time:.3f} s")
    print(f"Simulatiion time: {im_execution_time:.3f} s")

    probs_print(p_sim=simulator.get_p(), p_num=tt.get_p(), size=10)

    times_print(sim_moments=simulator.get_w(), calc_moments=tt.get_w())


if __name__ == "__main__":
    test_mmn_h2cold_h2_warm()
