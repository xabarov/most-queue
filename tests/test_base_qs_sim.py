import numpy as np

from most_queue.general.tables import probs_print, times_print
from most_queue.sim.queueing_systems.base import QsSim
from most_queue.theory.queueing_systems.fifo.mmnr import MMnrCalc
from most_queue.theory.queueing_systems.fifo.m_d_n import MDn


def test_sim_mmnr():
    """
    Test the simulation model for an M/M/n/r system
    """
    n = 3  # Number of channels
    l = 1.0  # Arrival rate intensity
    r = 30  # Queue length
    ro = 0.8  # Load coefficient
    mu = l / (ro * n)  # Service intensity

    # Create simulation instance
    qs = QsSim(n, buffer=r)

    # Set arrival process parameters and distribution
    qs.set_sources(l, 'M')
    # Set service time parameters and distribution
    qs.set_servers(mu, 'M')

    # Run the simulation
    qs.run(300000)
    # Get initial moments of waiting time. Also can get .v for sojourn times,
    # probabilities of states .get_p(), periods of continuous occupancy .pppz
    w_sim = qs.w

    mmnr = MMnrCalc(l, mu, n, r)
    w = mmnr.get_w()
    times_print(w_sim, w)

    # Verify simulation results against theoretical calculations
    assert np.allclose(w_sim, w, rtol=0.1), "MMQ system simulation results do not match theoretical values"

def test_sim_mdn():
    """
    Test the simulation model for a M/D/n
    """
    n = 3  # Number of channels
    l = 1.0  # Arrival rate intensity
    
    qs = QsSim(n)
    
    qs.set_sources(l, 'M')
    qs.set_servers(1.0 / (l / (n * 0.8)), 'D')  # Using same load coefficient as before

    qs.run(1000000)

    mdn = MDn(l, 1 / (l / (n * 0.8)), n)
    p_ch = mdn.calc_p()
    p_sim = qs.get_p()
    
    probs_print(p_sim=p_sim, p_ch=p_ch, size=10)

    assert np.allclose(p_sim[:10], p_ch[:10], atol=1e-2), "M/D/n system simulation results do not match theoretical values"