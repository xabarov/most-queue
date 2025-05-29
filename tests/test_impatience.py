"""
Test for M/M/1 queue with exponential impatience.
"""
from most_queue.general.tables import probs_print, times_print
from most_queue.sim.impatient import ImpatientQueueSim
from most_queue.theory.impatience.mm1 import MM1Impatience

NUM_OF_SERVERS = 1
ARRIVAL_RATE = 1.0
IMPATIENCE_RATE = 0.2
UTILIZATION = 0.8
NUM_OF_JOBS = 300000


def test_impatience():
    """
    Test for M/M/1 queue with exponential impatience.
    """
    mu = ARRIVAL_RATE / (UTILIZATION * NUM_OF_SERVERS)  # service rate

    # Calculate theoretical results
    imp_calc = MM1Impatience(ARRIVAL_RATE, mu, IMPATIENCE_RATE)
    v1 = imp_calc.get_v1()
    probs = imp_calc.probs

    # Simulate the queue
    qs = ImpatientQueueSim(NUM_OF_SERVERS)

    qs.set_sources(ARRIVAL_RATE, 'M')
    qs.set_servers(mu, 'M')
    qs.set_impatience(IMPATIENCE_RATE, 'M')

    qs.run(NUM_OF_JOBS)

    v1_im = qs.v[0]
    probs_sim = qs.get_p()

    # Print results

    times_print(v1_im, v1, is_w=False)
    probs_print(probs_sim, probs)

    assert abs(v1 - v1_im) < 1e-2


if __name__ == "__main__":
    test_impatience()
