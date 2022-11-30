import most_queue.sim.rand_destribution as rd
import math
from tqdm import tqdm
from most_queue.sim.smo_im import SmoIm, SetSmoException, Task
import numpy as np


class SmoBatchSim(SmoIm):
    def __init__(self, num_of_channels, batch_prob, buffer=None, verbose=True):
        super().__init__(num_of_channels, buffer, verbose)

        self.batch_prob = batch_prob
        self.calc_cdf_prob()

    def calc_cdf_prob(self):
        summ = 0
        self.batch_cdf = []
        for p in self.batch_prob:
            summ += p
            self.batch_cdf.append(summ)

    def arrival(self):

        """
        Действия по прибытию заявки в СМО.
        """

        p = np.random.random()
        batch_size = 0
        for i, batch_prob in enumerate(self.batch_cdf):
            if p < batch_prob:
                batch_size = i + 1
                break

        self.ttek = self.arrival_time
        self.t_old = self.ttek

        if self.cuda:
            self.arrival_time = self.ttek + self.source_random_vars[self.tek_source_num]
            self.tek_source_num += 1
            if self.tek_source_num == len(self.source_random_vars):
                self.tek_source_num = 0
        else:
            self.arrival_time = self.ttek + self.source.generate()

        for tsk in range(batch_size):
            self.arrived += 1
            self.p[self.in_sys] += self.arrival_time - self.t_old

            self.in_sys += 1

            if self.free_channels == 0:
                if self.buffer == None:  # не задана длина очередиб т.е бесконечная очередь
                    new_tsk = Task(self.ttek)
                    new_tsk.start_waiting_time = self.ttek
                    self.queue.append(new_tsk)
                else:
                    if len(self.queue) < self.buffer:
                        new_tsk = Task(self.ttek)
                        new_tsk.start_waiting_time = self.ttek
                        self.queue.append(new_tsk)
                    else:
                        self.dropped += 1
                        self.in_sys -= 1

            else:  # there are free channels:

                # check if its a warm phase:
                is_warm_start = False
                if len(self.queue) == 0 and self.free_channels == self.n and self.is_set_warm:
                    is_warm_start = True

                for s in self.servers:
                    if s.is_free:
                        self.taked += 1
                        s.start_service(Task(self.ttek), self.ttek, is_warm_start)
                        self.free_channels -= 1

                        # Проверям, не наступил ли ПНЗ:
                        if self.free_channels == 0:
                            if self.in_sys == self.n:
                                self.start_ppnz = self.ttek
                        break


if __name__ == '__main__':
    from most_queue.theory.batch_mm1 import BatchMM1

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
    n_jobs = 1000000

    batch_calc = BatchMM1(lam, mu, ls)

    v1 = batch_calc.get_v1()

    smo = SmoBatchSim(n, ls)

    smo.set_sources(lam, 'M')
    smo.set_servers(mu, 'M')

    smo.run(n_jobs)

    v1_im = smo.v[0]

    print("\nЗначения среднего времени пребывания заявок в системе:\n")

    print("{0:^15s}|{1:^15s}|{2:^15s}".format("№ момента", "Числ", "ИМ"))
    print("-" * 45)
    print("{0:^16d}|{1:^15.5g}|{2:^15.5g}".format(1, v1, v1_im))
