"""
Testing the M/D/n queueing system calculation.
For verification, we use simulation modeling 
"""
import numpy as np

from most_queue.sim.queueing_systems.fifo import QueueingSystemSimulator
from most_queue.theory.queueing_systems.fifo.m_d_n import MDn
from most_queue.general.tables import probs_print


def test_mdn():
    """
    Testing the M/D/n queueing system calculation.
    For verification, we use simulation modeling 
    """
    l = 1.0  # arrivals intensity
    ro = 0.8  # load factor
    n = 2  # number of service channels
    num_of_jobs = 300000  # number of jobs for simulation

    b = ro * n / l  # service time from given ro

    # calculation of the probabilities of queueing system states
    mdn = MDn(l, b, n)
    p_ch = mdn.calc_p()

    # for verification, we use simulation modeling
    # create an instance of the simulation class and pass the number of service channels
    qs = QueueingSystemSimulator(n)

    # set arrivals. The method needs to be passed distribution parameters and type of distribution.
    qs.set_sources(l, "M")

    # set the service channels. To the method we pass parameters (in our case, service time)
    # and type of distribution - D (deterministic).
    qs.set_servers(b, "D")

    # start the simulation. The method takes the number of jobs to simulate.
    qs.run(num_of_jobs)

    # get the distribution of queueing system states probabilities
    p_sim = qs.get_p()

    assert np.allclose(np.array(p_sim[:10]), np.array(p_ch[:10]), atol=1e-2)

    # Output results
    probs_print(p_ch, p_sim)


if __name__ == "__main__":
    test_mdn()
