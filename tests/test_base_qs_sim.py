"""
Test the simulation model for an M/M/n/r system
"""
import numpy as np

from most_queue.general.tables import probs_print, times_print
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.m_d_n import MDn
from most_queue.theory.fifo.mmnr import MMnrCalc

ERROR_MSG = "System simulation results do not match theoretical values"

NUM_OF_CHANNELS = 3
NUM_OF_JOBS = 300000
ARRIVAL_RATE = 1.0
UTILIZATION_FACTOR = 0.8
BUFFER = 30


def test_sim_mmnr():
    """
    Test the simulation model for an M/M/n/r system
    """
    mu = ARRIVAL_RATE / (UTILIZATION_FACTOR *
                         NUM_OF_CHANNELS)  # Service intensity

    # Create simulation instance
    qs = QsSim(NUM_OF_CHANNELS, buffer=BUFFER)

    # Set arrival process parameters and distribution as exponential
    qs.set_sources(ARRIVAL_RATE, 'M')
    # Set service time parameters and distribution as exponential
    qs.set_servers(mu, 'M')

    # Run the simulation
    qs.run(NUM_OF_JOBS)
    # Get initial moments of waiting time. Also can get .v for sojourn times,
    # probabilities of states .get_p(), periods of continuous occupancy .pppz
    w_sim = qs.w

    mmnr = MMnrCalc(ARRIVAL_RATE, mu, NUM_OF_CHANNELS, BUFFER)
    w = mmnr.get_w()
    times_print(w_sim, w)

    # Verify simulation results against theoretical calculations
    assert np.allclose(w_sim, w, rtol=0.2), ERROR_MSG


def test_sim_mdn():
    """
    Test the simulation model for a M/D/n
    """
    qs = QsSim(NUM_OF_CHANNELS)

    mu = ARRIVAL_RATE / (UTILIZATION_FACTOR *
                         NUM_OF_CHANNELS)  # Service intensity

    qs.set_sources(ARRIVAL_RATE, 'M')
    # Using same load coefficient as before
    qs.set_servers(1.0 / mu, 'D')

    qs.run(NUM_OF_JOBS)

    mdn = MDn(ARRIVAL_RATE, 1 / mu, NUM_OF_CHANNELS)
    p_num = mdn.calc_p()
    p_sim = qs.get_p()

    probs_print(p_sim=p_sim, p_num=p_num, size=10)

    assert np.allclose(p_sim[:10], p_num[:10], atol=1e-2), ERROR_MSG


if __name__ == "__main__":
    test_sim_mmnr()
    test_sim_mdn()
