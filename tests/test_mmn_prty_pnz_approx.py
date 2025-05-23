"""
Test the MMn_PRTY_PNZ_Cox_approx class for priority queueing systems.
"""
from most_queue.general.tables import probs_print, times_print
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.preemptive.mmn_2cls_pr_busy_approx import (
    MMn_PRTY_PNZ_Cox_approx,
)


def test_mmn_prty():
    """
    Test function for evaluating the performance of a priority queueing system approximation.
    This function compares simulation results with theoretical calculations using Cox approximation.
    """

    # Simulation parameters
    num_of_jobs = 100000  # Number of jobs to simulate
    n = 3                # Number of servers/channels
    num_priority = 2                # Number of priority classes

    # Service rates (mu)
    mu_high = 1.5           # Service rate for high-priority class
    mu_low = 1.3           # Service rate for low-priority class

    # Arrival rates (lambda)
    lambda_high = 1.0       # Arrival rate for high-priority class
    lambda_low = 1.4       # Arrival rate for low-priority class

    rho = 0.8            # System utilization factor

    # Setting up arrival streams and server parameters
    sources = []
    servers_params = []

    # Defining arrival rates and service rates for each class
    arrival_rates = [lambda_high, lambda_low]
    service_rates = [mu_high, mu_low]

    for j in range(num_priority):
        sources.append({'type': 'M', 'params': arrival_rates[j]})
        servers_params.append({'type': 'M', 'params': service_rates[j]})

    # Setting up the priority queueing system simulator
    qs = PriorityQueueSimulator(n, num_priority, "PR")
    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # Running the simulation
    qs.run(num_of_jobs)

    # Getting simulation results
    p_sim = qs.get_p()
    v_sim = qs.v

    # Setting up and running the theoretical approximation model
    tt = MMn_PRTY_PNZ_Cox_approx(n, mu_low, mu_high, lambda_low, lambda_high)
    tt.run()
    p_tt = tt.get_p()
    v_tt = tt.get_second_class_v1()

    # Printing comparison results
    print("\nComparison of theoretical calculation with Cox approximation and simulation results.")
    print(f"Utilization factor: {rho:^6.2f}")
    print(f"Number of simulated jobs: {num_of_jobs:d}\n")

    print("Probabilities of system states for low-priority class")

    # Printing probability comparison table
    probs_print(p_sim=p_sim[1], p_ch=p_tt, size=10)

    # Printing time moments comparison
    times_print(sim_moments=[v_sim[1][0]], calc_moments=[v_tt], is_w=False)

    # Asserting the accuracy of the results
    assert 100 * abs(v_tt - v_sim[1][0]) / max(v_tt, v_sim[1][0]) < 10


if __name__ == "__main__":
    test_mmn_prty()
