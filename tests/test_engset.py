"""
Test for Engset model (M/M/1 with a finite number of sources)
"""
import numpy as np

from most_queue.general.tables import probs_print, times_print
from most_queue.sim.finite_source import QueueingFiniteSourceSim
from most_queue.theory.closed.engset import Engset

ERROR_MSG = "QueueingFiniteSourceSim simulation results do not match theoretical values"

ARRIVAL_RATE = 0.3
SOURCE_NUM = 7
NUM_OF_JOBS = 100000
SERVICE_RATE = 1.0


def test_engset():
    """
    Test for Engset model (M/M/1 with a finite number of sources)
    """
    # Calculation of the Engset model
    engset = Engset(ARRIVAL_RATE, SERVICE_RATE, SOURCE_NUM)

    # Get probabilities of states of the system
    ps_ch = engset.get_p()

    job_in_sys_ave = engset.get_N()  # average number of requests in the system
    job_in_queue_ave = engset.get_Q()  # average queue length
    # Get probability that a randomly chosen source can send a request,
    # i.e. the readiness coefficient
    kg = engset.get_kg()

    print(f'job_in_sys_ave = {job_in_sys_ave:3.3f}')
    print(f'job_in_queue_ave = {job_in_queue_ave:3.3f}')
    print(f'readiness coefficient = {kg:3.3f}')

    w1 = engset.get_w1()  # average waiting time
    v1 = engset.get_v1()  # average sojourn time
    w_num = engset.get_w()  # waiting time initial moments
    v_num = engset.get_v()  # sourjourn time initial moments

    print(f'v1 = {v1:3.3f}, w1 = {w1:3.3f}')

    # Simulation of the system with a finite number of sources

    finite_source_sim = QueueingFiniteSourceSim(1, SOURCE_NUM)

    finite_source_sim.set_sources(ARRIVAL_RATE, 'M')
    finite_source_sim.set_servers(SERVICE_RATE, 'M')

    finite_source_sim.run(NUM_OF_JOBS)

    ps_sim = finite_source_sim.get_p()
    v_sim = finite_source_sim.v
    w_sim = finite_source_sim.w

    # Comparison of the results from the simulation and the analytical model

    probs_print(ps_sim, ps_ch)

    times_print(w_sim, w_num, is_w=True)
    times_print(v_sim, v_num, is_w=False)

    assert np.allclose(w_sim, w_num, rtol=0.1), ERROR_MSG


if __name__ == "__main__":

    test_engset()
