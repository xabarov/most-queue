"""
QS with batch arrival
"""
import numpy as np

from most_queue.sim.qs_sim import QueueingSystemSimulator, Task


class QueueingSystemBatchSim(QueueingSystemSimulator):
    """
    QS with batch arrivals
    """

    def __init__(self, num_of_channels,
                 batch_prob,
                 buffer=None, verbose=True):
        """
        Args:
            num_of_channels: 
            batch_prob: list[float], probs of batch size 1, 2, .. len(batch_probs)
            buffer: Optional(int, None) : length of queueu
        """
        super().__init__(num_of_channels, buffer, verbose)

        self.batch_prob = batch_prob
        self.calc_cdf_prob()

    def calc_cdf_prob(self):
        """
        Calcs CDF of batch probs distribution
        """
        summ = 0
        self.batch_cdf = []
        for p in self.batch_prob:
            summ += p
            self.batch_cdf.append(summ)

    def arrival(self):
        """
        Actions upon arrival of job to QS
        """

        p = np.random.random()
        batch_size = 0
        for i, batch_prob in enumerate(self.batch_cdf):
            if p < batch_prob:
                batch_size = i + 1
                break

        self.ttek = self.arrival_time
        self.t_old = self.ttek

        self.arrival_time = self.ttek + self.source.generate()

        for tsk in range(batch_size):
            self.arrived += 1
            self.p[self.in_sys] += self.arrival_time - self.t_old

            self.in_sys += 1

            if self.free_channels == 0:
                if self.buffer is None:  # не задана длина очередиб т.е бесконечная очередь
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
                if self.queue.size() == 0 and self.free_channels == self.n and self.warm_phase.is_set:
                    is_warm_start = True

                for s in self.servers:
                    if s.is_free:
                        self.taked += 1
                        s.start_service(Task(self.ttek),
                                        self.ttek, is_warm_start)
                        self.free_channels -= 1

                        # Проверям, не наступил ли ПНЗ:
                        if self.free_channels == 0:
                            if self.in_sys == self.n:
                                self.start_ppnz = self.ttek
                        break
