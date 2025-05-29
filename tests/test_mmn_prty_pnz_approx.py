"""
Test the MMn_PRTY_PNZ_Cox_approx class for priority queueing systems.
"""
from most_queue.general.tables import probs_print, times_print
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.preemptive.mmn_2cls_pr_busy_approx import (
    MMn_PRTY_PNZ_Cox_approx,
)

ARRIVAL_RATE_HIGH = 1.0
ARRIVAL_RATE_LOW = 1.4
NUM_OF_CHANNELS = 3

NUM_OF_JOBS = 300000
NUM_OF_CHANNELS = 3
UTILIZATION_HIGH = 0.6
UTILIZATION = 0.8


def test_mmn_prty():
    """
    Test function for evaluating the performance of a priority queueing system approximation.
    This function compares simulation results with theoretical calculations using Cox approximation.
    """

    num_priority = 2 # Number of priority classes

    # Setting up arrival streams and server parameters
    sources = []
    servers_params = []

    # Defining arrival rates and service rates for each class
    arrival_rates = [ARRIVAL_RATE_HIGH, ARRIVAL_RATE_LOW]

    mu_high = ARRIVAL_RATE_HIGH / (NUM_OF_CHANNELS*UTILIZATION_HIGH)
    mu_low = ARRIVAL_RATE_LOW / \
        (NUM_OF_CHANNELS*(UTILIZATION - UTILIZATION_HIGH))
    service_rates = [mu_high, mu_low]

    for j in range(num_priority):
        sources.append({'type': 'M', 'params': arrival_rates[j]})
        servers_params.append({'type': 'M', 'params': service_rates[j]})

    # Setting up the priority queueing system simulator
    qs = PriorityQueueSimulator(NUM_OF_CHANNELS, num_priority, "PR")
    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # Running the simulation
    qs.run(NUM_OF_JOBS)

    # Getting simulation results
    p_sim = qs.get_p()
    v_sim = qs.v

    # Setting up and running the theoretical approximation model
    tt = MMn_PRTY_PNZ_Cox_approx(
        NUM_OF_CHANNELS, mu_low, mu_high, ARRIVAL_RATE_LOW, ARRIVAL_RATE_HIGH)
    tt.run()
    p_tt = tt.get_p()
    v_tt = tt.get_second_class_v1()

    # Printing comparison results
    print("\nComparison of theoretical calculation with Cox approximation and simulation results.")
    print(f"Utilization factor for low-priority class: {UTILIZATION:^6.2f}")
    print(
        f"Utilization factor for high-priority class: {UTILIZATION_HIGH:^6.2f}")

    print(f"Number of simulated jobs: {NUM_OF_JOBS:d}\n")

    print("Probabilities of system states for low-priority class")

    # Printing probability comparison table
    probs_print(p_sim=p_sim[1], p_num=p_tt, size=10)

    # Printing time moments comparison
    times_print(sim_moments=[v_sim[1][0]], calc_moments=[v_tt], is_w=False)

    # Asserting the accuracy of the results
    assert 100 * abs(v_tt - v_sim[1][0]) / max(v_tt, v_sim[1][0]) < 10


if __name__ == "__main__":
    test_mmn_prty()
