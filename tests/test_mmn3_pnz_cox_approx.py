"""
Test the M/M/n queue with three priority classes using Cox approximation.
"""
from most_queue.general.tables import probs_print
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.theory.priority.preemptive.mmn_2cls_pr_busy_approx import (
    MMn_PRTY_PNZ_Cox_approx,
)
from most_queue.theory.priority.preemptive.mmn_3cls_busy_approx import Mmn3_pnz_cox

ARRIVAL_RATE_HIGH = 1.0
ARRIVAL_RATE_MEDIUM = 1.2
ARRIVAL_RATE_LOW = 1.4

NUM_OF_CHANNELS = 2

NUM_OF_JOBS = 300000
NUM_OF_CHANNELS = 2

SERVICE_RATE_HIGH = 1.5
SERVICE_RATE_MEDIUM = 1.8
SERVICE_RATE_LOW = 3.5


def test_mmn3():
    """
    Test the M/M/n queue with three priority classes using Cox approximation.
    """
    # Simulation parameters
    classes = 3        # Number of priority classes

    # Initialize simulation
    queue_simulator = PriorityQueueSimulator(
        num_of_channels=NUM_OF_CHANNELS, num_of_classes=classes, prty_type="PR")

    # Set up sources and servers for each class
    sources = []
    server_params = []
    rates = [ARRIVAL_RATE_HIGH, ARRIVAL_RATE_MEDIUM, ARRIVAL_RATE_LOW]
    mus = [SERVICE_RATE_HIGH, SERVICE_RATE_MEDIUM, SERVICE_RATE_LOW]

    for k in range(classes):
        sources.append({'type': 'M', 'params': rates[k]})
        server_params.append({'type': 'M', 'params': mus[k]})

    queue_simulator.set_sources(sources)
    queue_simulator.set_servers(server_params)

    # Run simulation
    queue_simulator.run(NUM_OF_JOBS)

    # Get simulation results
    sim_probabilities = queue_simulator.get_p()
    sim_throughput = queue_simulator.v

    # Initialize analytical models
    analytical_model_3cls = Mmn3_pnz_cox(SERVICE_RATE_LOW, SERVICE_RATE_MEDIUM, SERVICE_RATE_HIGH,
                                         ARRIVAL_RATE_LOW, ARRIVAL_RATE_MEDIUM, ARRIVAL_RATE_HIGH)

    analytical_model_2cls = MMn_PRTY_PNZ_Cox_approx(n=NUM_OF_CHANNELS, mu_L=SERVICE_RATE_MEDIUM,
                                                    mu_H=SERVICE_RATE_HIGH,
                                                    l_L=ARRIVAL_RATE_MEDIUM, l_H=ARRIVAL_RATE_HIGH)

    analytical_model_2cls.run()

    # Run calculations
    analytical_model_3cls.run()
    p_analytical = analytical_model_3cls.get_p()
    v_low_analytical = analytical_model_3cls.get_low_class_v1()
    v_med_analytical = analytical_model_2cls.get_second_class_v1()

    # Get results for reference class (high priority)
    mmnr_results = MMnrCalc(
        ARRIVAL_RATE_HIGH, SERVICE_RATE_HIGH, n=NUM_OF_CHANNELS, r=100)
    v_high_analytical = mmnr_results.get_v()[0]

    # Display comparison results
    print("\nComparison of analytical calculations with Cox approximation and simulation results")

    print(f"Number of simulated jobs: {NUM_OF_JOBS}\n")

    # Print probabilities for high priority class states
    print("Probabilities of system states for low priority class")
    probs_print(p_sim=sim_probabilities[2], p_num=p_analytical, size=10)

    # Compare waiting times
    print("Average waiting times in the system")
    print("-" * 38)
    cls_col, num_col, sim_col = 'Class', 'Num', 'Sim'
    print(f"{cls_col:^10s}|{num_col:^15s}|{sim_col:^15s}")
    print("-" * 38)

    classes_to_compare = [0, 1, 2]
    analytical_values = [v_high_analytical, v_med_analytical, v_low_analytical]
    simulation_values = sim_throughput

    for cls in classes_to_compare:
        print(
            f"{cls+1:^10d}|{analytical_values[cls]:^15.3g}|{simulation_values[cls][0]:^15.3g}")


if __name__ == "__main__":
    test_mmn3()
