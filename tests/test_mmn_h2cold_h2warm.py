"""
Test function to compare results of the simulation and Takahashi-Takami (TT) algorithm
for an MMn queueing system with H2 cold and warm-up phases.
"""
import time

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.queueing_systems.vacations import \
    VacationQueueingSystemSimulator
from most_queue.theory.queueing_systems.vacations.mmn_with_h2_cold_and_h2_warmup import \
    MMn_H2warm_H2cold


def calculate_gamma_moments(mean, cv):
    """
    Helper function to calculate Gamma distribution parameters.
    """
    alpha = 1 / (cv ** 2)
    b1 = mean
    b2 = (b1 ** 2) * (cv ** 2 + 1)
    b3 = b2 * b1 * (1 + 2 / alpha)

    return [b1, b2, b3]
    

def test_mmn_h2cold_h2_warm():
    """
    Test function to compare results of the Implicit Method (IM) and Takahashi-Takami (TT) algorithm
    for an MMn queueing system with H2 cold and warm-up phases.
    """

    # Parameters setup
    num_channels = 1          # Number of servers (channels)
    arrival_rate = 1.0       # Arrival rate
    utilization = 0.2        # System load coefficient

    # Warm phase parameters
    mean_warmup_time = num_channels * 0.3 / arrival_rate
    cv_warmup = 1.01         # Coefficient of variation for warm-up time

    # Cold phase parameters
    mean_cold_time = num_channels * 0.0001 / arrival_rate
    cv_cold = 1.01           # Coefficient of variation for cold time

    # Simulation settings
    num_jobs = 300_000      # Number of jobs to simulate
    verbose = False          # Verbosity level (set to True if needed)
    
    # Initialize the Vacation Queueing System Simulator
    simulator = VacationQueueingSystemSimulator(num_channels, buffer=None)

    # Set warm-up phase parameters
    b_w = calculate_gamma_moments(mean_warmup_time, cv_warmup)
    warmup_params = GammaDistribution.get_params(b_w)
    simulator.set_warm(warmup_params, 'Gamma')

    # Set cold phase parameters
    b_c = calculate_gamma_moments(mean_cold_time, cv_cold)
    cold_params = GammaDistribution.get_params(b_c)
    simulator.set_cold(cold_params, 'Gamma')

    # Configure the simulator
    simulator.set_sources(arrival_rate, 'M')
    service_rate = 1.0 / (num_channels * utilization / arrival_rate)
    simulator.set_servers(service_rate, 'M')

    # Run simulations
    im_start_time = time.process_time()
    simulator.run(num_jobs)
    im_execution_time = time.process_time() - im_start_time

    tt_start_time = time.process_time()
    tt = MMn_H2warm_H2cold(arrival_rate, service_rate, b_w, b_c, num_channels, buffer=None,
                           verbose=verbose, accuracy=1e-14)
    tt.run()
    tt_execution_time = time.process_time() - tt_start_time

    num_of_iter = tt.num_of_iter_

    print('warms starts', simulator.warm_phase.starts_times)
    print('warms after cold starts', simulator.warm_after_cold_starts)
    print('cold starts', simulator.cold_phase.starts_times)
    print("zero wait arrivals num", simulator.zero_wait_arrivals_num)

    print(f"\nComparison of results calculated using the Takahashi-Takami method and simulation.\n"
          f"Sim - M/Gamma/{{{num_channels:2d}}} with Gamma warming\n"
          f"Takahashi-Takami - M/M/{{{num_channels:2d}}} with H2-warming and H2-cooling "
          f"with complex parameters\n"
          f"Utilization coefficient: {utilization:.2f}")
    print(f'Variation coefficient of warming time {cv_warmup:.3f}')
    print(f'Variation coefficient of cooling time {cv_cold:.3f}')
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
    
    probs_print(p_sim=simulator.get_p(), p_ch=tt.get_p(), size=10)

    times_print(sim_moments=simulator.get_w(), calc_moments=tt.get_w())


if __name__ == "__main__":
    test_mmn_h2cold_h2_warm()
