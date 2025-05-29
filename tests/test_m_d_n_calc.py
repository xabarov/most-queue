"""
Testing the M/D/n queueing system calculation.
For verification, we use simulation modeling 
"""
import numpy as np

from most_queue.general.tables import probs_print
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.m_d_n import MDn

NUM_OF_SERVERS = 2
ARRIVAL_RATE = 1.0
UTILIZATION = 0.8
NUM_OF_JOBS = 300000


def test_mdn():
    """
    Testing the M/D/n queueing system calculation.
    For verification, we use simulation modeling 
    """

    b = UTILIZATION * NUM_OF_SERVERS / ARRIVAL_RATE  # service time from given ro

    # calculation of the probabilities of queueing system states
    mdn = MDn(ARRIVAL_RATE, b, NUM_OF_SERVERS)
    p_num = mdn.calc_p()

    # for verification, we use simulation modeling
    # create an instance of the simulation class and pass the number of service channels
    qs = QsSim(NUM_OF_SERVERS)

    # set arrivals. The method needs to be passed distribution parameters and type of distribution.
    qs.set_sources(ARRIVAL_RATE, "M")

    # set the service channels. To the method we pass parameters (in our case, service time)
    # and type of distribution - D (deterministic).
    qs.set_servers(b, "D")

    # start the simulation. The method takes the number of jobs to simulate.
    qs.run(NUM_OF_JOBS)

    # get the distribution of queueing system states probabilities
    p_sim = qs.get_p()

    assert np.allclose(np.array(p_sim[:10]), np.array(p_num[:10]), atol=1e-2)

    # Output results
    probs_print(p_num, p_sim)


if __name__ == "__main__":
    test_mdn()
