"""
Test module for batch arrival queueing systems. 
Includes tests for the following models:
- Mx/M/1/infinite
The main function is test_batch_mm1, which tests the Mx/M/1/infinite model.
It compares the results of the simulation and the analytical solution.
"""
from most_queue.general.tables import times_print
from most_queue.sim.queueing_systems.batch import QueueingSystemBatchSim
from most_queue.theory.queueing_systems.batch.mm1 import BatchMM1


def calc_mean_batch_size(batch_probs):
    """
    Calc mean batch size 
    batch_probs - probs of batch size 1, 2, .. len(batch_probs)
    """
    mean = 0
    for i, prob in enumerate(batch_probs):
        mean += (i + 1)*prob
    return mean


def test_batch_mm1():
    """
    Test QS Mx/M/1/infinite with batch arrivals
    """

    # probs of batch size 1, 2, .. 5
    batch_probs = [0.2, 0.3, 0.1, 0.2, 0.2]
    mean_batch_size = calc_mean_batch_size(batch_probs)

    n = 1   # one channel
    lam = 0.7  # arrival intensity
    ro = 0.7  # QS utilization factor
    mu = lam * mean_batch_size / ro  # serving intensity
    n_jobs = 500000  # jobs to serve in QS simulation

    batch_calc = BatchMM1(lam, mu, batch_probs)

    v1 = batch_calc.get_v1()

    qs = QueueingSystemBatchSim(n, batch_probs)

    qs.set_sources(lam, 'M')
    qs.set_servers(mu, 'M')

    qs.run(n_jobs, is_real_served=True)

    v1_im = qs.v[0]

    times_print(v1_im, v1, False)  # prints 2.6556 and approx 2.5-2.7

    assert v1 - v1_im < 0.2


if __name__ == "__main__":

    test_batch_mm1()
