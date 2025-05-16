"""
Test the M/M/n queue with three priority classes using Cox approximation.
"""
from most_queue.general.tables import probs_print
from most_queue.sim.queueing_systems.priority import PriorityQueueSimulator
from most_queue.theory.queueing_systems.fifo.mmnr import MMnrCalc
from most_queue.theory.queueing_systems.priority.preemptive.mmn_2cls_pr_busy_approx import \
    MMn_PRTY_PNZ_Cox_approx
from most_queue.theory.queueing_systems.priority.preemptive.mmn_3cls_busy_approx import \
    Mmn3_pnz_cox


def test_mmn3():
    """
    Test the M/M/n queue with three priority classes using Cox approximation.
    """
    # Simulation parameters
    num_jobs = 200000  # Number of jobs to simulate
    n_channels = 2     # Number of servers (channels)
    classes = 3        # Number of priority classes

    # Service rates for each class (high, medium, low priority)
    mu_high = 1.5
    mu_medium = 1.4
    mu_low = 1.3

    # Arrival rates for each class
    lambda_high = 0.9
    lambda_medium = 0.8
    lambda_low = 0.7

    # Total arrival rate
    total_lambda = lambda_high + lambda_medium + lambda_low

    # Calculate average service time and utilization factor
    avg_service_time = (lambda_low / total_lambda) * (1/mu_low) + \
                       (lambda_high / total_lambda) * (1/mu_high) + \
                       (lambda_medium / total_lambda) * (1/mu_medium)

    rho = total_lambda * avg_service_time / n_channels

    # Initialize simulation
    queue_simulator = PriorityQueueSimulator(
        num_of_channels=n_channels, num_of_classes=classes, prty_type="PR")

    # Set up sources and servers for each class
    sources = []
    server_params = []
    rates = [lambda_high, lambda_medium, lambda_low]
    mus = [mu_high, mu_medium, mu_low]

    for _class in range(classes):
        sources.append({'type': 'M', 'params': rates[_class]})
        server_params.append({'type': 'M', 'params': mus[_class]})

    queue_simulator.set_sources(sources)
    queue_simulator.set_servers(server_params)

    # Run simulation
    queue_simulator.run(num_jobs)

    # Get simulation results
    sim_probabilities = queue_simulator.get_p()
    sim_throughput = queue_simulator.v

    # Initialize analytical models
    analytical_model_3cls = Mmn3_pnz_cox(mu_low, mu_medium, mu_high,
                                         lambda_low, lambda_medium, lambda_high)

    analytical_model_2cls = MMn_PRTY_PNZ_Cox_approx(n=2, mu_L=mu_medium,
                                                    mu_H=mu_high,
                                                    l_L=lambda_medium, l_H=lambda_high)

    analytical_model_2cls.run()

    # Run calculations
    analytical_model_3cls.run()
    p_analytical = analytical_model_3cls.get_p()
    v_low_analytical = analytical_model_3cls.get_low_class_v1()
    v_med_analytical = analytical_model_2cls.get_second_class_v1()

    # Get results for reference class (high priority)
    mmnr_results = MMnrCalc(lambda_high, mu_high, n=2, r=100)
    v_high_analytical = mmnr_results.get_v()[0]

    # Display comparison results
    print("\nComparison of analytical calculations with Cox approximation and simulation results")
    print(f"Utilization factor (rho): {rho:.2f}")
    print(f"Number of simulated jobs: {num_jobs}\n")

    # Print probabilities for high priority class states
    print("Probabilities of system states for low priority class")
    probs_print(p_sim=sim_probabilities[2], p_ch=p_analytical, size=10)

    # Compare waiting times
    print("\n{:^40s}".format("Average waiting times in the system"))
    print("-" * 38)
    print("{:^10s}|{:^15s}|{:^15s}".format(
        "Class", "Analytical", "Simulation"))
    print("-" * 38)

    classes_to_compare = [0, 1, 2]
    analytical_values = [v_high_analytical, v_med_analytical, v_low_analytical]
    simulation_values = sim_throughput

    for cls in classes_to_compare:
        print(
            f"{cls+1:^10d}|{analytical_values[cls]:^15.3g}|{simulation_values[cls][0]:^15.3g}")

    # Validate results (example assertion)
    assert 100 * abs(v_high_analytical - simulation_values[0][0]) / \
        max(v_high_analytical, simulation_values[0][0]) < 10


if __name__ == "__main__":
    test_mmn3()
