"""
Test for Engset model (M/M/1 with a finite number of sources)
"""
from most_queue.general.tables import probs_print, times_print
from most_queue.theory.queueing_systems.closed.engset import Engset
from most_queue.sim.queueing_systems.finite_source import QueueingFiniteSourceSim


def test_engset():
    """
    Test for Engset model (M/M/1 with a finite number of sources)
    """

    lam = 0.3  # arrival rate for each source
    mu = 1.0  # service rate
    m = 7  # number of sources

    jobs_for_simulation = 100000  # number of jobs for simulation

    # Calculation of the Engset model
    engset = Engset(lam, mu, m)

    # Get probabilities of states of the system
    ps_ch = engset.get_p()

    N = engset.get_N()  # average number of requests in the system
    Q = engset.get_Q()  # average queue length
    # Get probability that a randomly chosen source can send a request,  i.e. the readiness coefficient
    kg = engset.get_kg()

    print(f'N = {N:3.3f}, Q = {Q:3.3f}, kg = {kg:3.3f}')

    w1 = engset.get_w1()  # average waiting time
    v1 = engset.get_v1()  # average sojourn time
    w_ch = engset.get_w()  # waiting time initial moments
    v_ch = engset.get_v()  # sourjourn time initial moments

    print(f'v1 = {v1:3.3f}, w1 = {w1:3.3f}')

    # Simulation of the system with a finite number of sources

    finite_source_sim = QueueingFiniteSourceSim(1, m)

    finite_source_sim.set_sources(lam, 'M')
    finite_source_sim.set_servers(mu, 'M')

    finite_source_sim.run(jobs_for_simulation)

    ps_sim = finite_source_sim.get_p()
    v_sim = finite_source_sim.v
    w_sim = finite_source_sim.w

    # Comparison of the results from the simulation and the analytical model

    probs_print(ps_sim, ps_ch)

    times_print(w_sim, w_ch, is_w=True)
    times_print(v_sim, v_ch, is_w=False)


if __name__ == "__main__":

    test_engset()
