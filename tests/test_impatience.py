import numpy as np

from most_queue.sim.impatient_queue_sim import ImpatientQueueSim
from most_queue.theory import impatience_calc


def test_impatience():
    n = 1
    l = 1.0
    ro = 0.8
    n_jobs = 300000
    mu = l / (ro * n)
    gamma = 0.2

    v1 = impatience_calc.get_v1(l, mu, gamma)

    qs = ImpatientQueueSim(n)

    qs.set_sources(l, 'M')
    qs.set_servers(mu, 'M')
    qs.set_impatiens(gamma, 'M')

    qs.run(n_jobs)

    v1_im = qs.v[0]

    print("\nЗначения среднего времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(1, v1, v1_im))

    assert abs(v1 - v1_im) < 1e-2


if __name__ == "__main__":
    test_impatience()
