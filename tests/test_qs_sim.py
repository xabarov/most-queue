"""
Test the simulation of a queueing system
For verification, compare with results for M/M/3 and M/D/3 systems
"""
import numpy as np

from most_queue.general.tables import probs_print, times_print
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.m_d_n import MDn
from most_queue.theory.fifo.mmnr import MMnrCalc


def test_sim():
    """
    Test the simulation of a queueing system
    For verification, compare with results for M/M/3 and M/D/3 systems
    """
    # System parameters
    num_servers = 3          # Number of channels/servers
    arrival_rate = 1.0      # Arrival rate (lambda)
    queue_length = 30       # Buffer size/r queue length
    utilization = 0.8       # Server utilization (rho)

    # Calculate service rate based on utilization
    service_rate = arrival_rate / (utilization * num_servers)

    # Initialize simulation model
    qs = QsSim(num_servers, buffer=queue_length)

    # Set arrival process parameters and distribution (M for Markovian)
    qs.set_sources(arrival_rate, 'M')

    # Set service time parameters and distribution (M for Markovian)
    qs.set_servers(service_rate, 'M')

    # Run simulation with 300,000 arrivals
    qs.run(300000)

    # Get simulated waiting times
    w_sim = qs.w

    # Calculate theoretical waiting times using MMnr model
    mmnr = MMnrCalc(arrival_rate, service_rate, num_servers, queue_length)
    w_theory = mmnr.get_w()

    # Print comparison of simulation and theoretical results
    times_print(w_sim, w_theory, True)

    # Reset for next part of test
    qs = QsSim(num_servers)

    # Set arrival process again (M distribution)
    qs.set_sources(arrival_rate, 'M')

    # Set deterministic service times (D distribution)
    qs.set_servers(1.0 / service_rate, 'D')

    # Run simulation with 1,000,000 arrivals
    qs.run(1000000)

    # Calculate theoretical probabilities using MDn model
    mdn = MDn(arrival_rate, 1.0 / service_rate, num_servers)
    p_theory = mdn.calc_p()

    # Get simulated state probabilities
    p_sim = qs.get_p()

    # Print comparison of simulation and theoretical probabilities
    probs_print(p_sim, p_theory, 10)

    # Assert that first 10 probabilities match within tolerance
    np.allclose(np.array(p_sim[:10]), np.array(p_theory[:10]), atol=1e-2)


if __name__ == "__main__":
    test_sim()
