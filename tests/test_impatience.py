"""
Test for M/M/1 queue with exponential impatience.
"""
from most_queue.sim.queueing_systems.impatient import ImpatientQueueSim
from most_queue.theory.queueing_systems.impatience.mm1 import MM1Impatience
from most_queue.general.tables import times_print, probs_print


def test_impatience():
    """
    Test for M/M/1 queue with exponential impatience.
    """
    n = 1  # number of servers
    l = 1.0  # arrival rate
    ro = 0.8  # load factor
    n_jobs = 300000  # number of jobs to simulate
    mu = l / (ro * n)  # service rate
    gamma = 0.2  # impatience rate

    # Calculate theoretical results
    imp_calc = MM1Impatience(l, mu, gamma)
    v1 = imp_calc.get_v1()
    probs = imp_calc.probs

    # Simulate the queue
    qs = ImpatientQueueSim(n)

    qs.set_sources(l, 'M')
    qs.set_servers(mu, 'M')
    qs.set_impatience(gamma, 'M')

    qs.run(n_jobs)

    v1_im = qs.v[0]
    probs_sim = qs.get_p()

    # Print results

    times_print(v1_im, v1, is_w=False)
    probs_print(probs_sim, probs)

    assert abs(v1 - v1_im) < 1e-2


if __name__ == "__main__":
    test_impatience()
