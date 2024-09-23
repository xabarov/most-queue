from theory.batch_mm1 import BatchMM1
from sim.batch_sim import QueueingSystemBatchSim
import numpy as np


def test_batch_mm1():
    ls = [0.2, 0.3, 0.1, 0.2, 0.2]
    moments = [0.0, 0.0]
    for j in range(len(moments)):
        for i in range(len(ls)):
            moments[j] += pow(i + 1, j + 1)
        moments[j] /= len(ls)

    n = 1
    lam = 0.7
    ro = 0.8
    mu = lam * moments[0] / ro
    n_jobs = 300000

    batch_calc = BatchMM1(lam, mu, ls)

    v1 = batch_calc.get_v1()

    qs = QueueingSystemBatchSim(n, ls)

    qs.set_sources(lam, 'M')
    qs.set_servers(mu, 'M')

    qs.run(n_jobs)

    v1_im = qs.v[0]

    print("\nЗначения среднего времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(1, v1, v1_im))
    
    assert 100*abs(v1 - v1_im)/max(v1, v1_im) < 5.0