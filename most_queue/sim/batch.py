"""
Queueing system with batch arrivals.
"""

import numpy as np

from most_queue.sim.base import QsSim, Task


class QueueingSystemBatchSim(QsSim):
    """
    Queueing system with batch arrivals GI[x]/G/c/m

    """

    def __init__(
        self, num_of_channels, batch_prob, buffer=None, verbose=True, buffer_type="list"
    ):  # pylint: disable=too-many-positional-arguments, too-many-arguments
        """
        :param num_of_channels: int : number of channels (servers)
        :param batch_prob: list : probabilities for different batch sizes
        :param buffer: Optional(int, None) : length of queueu
        :param verbose: bool : if True prints info about simulation process
        """
        super().__init__(num_of_channels, buffer, verbose, buffer_type)

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

    def _handle_queueing(self, ttek):
        """Helper to manage task queuing when no free channels."""
        if self.buffer is None:  # infinite buffer
            new_tsk = Task(ttek)
            new_tsk.start_waiting_time = ttek
            self.queue.append(new_tsk)
        else:
            if len(self.queue) < self.buffer:
                new_tsk = Task(ttek)
                new_tsk.start_waiting_time = ttek
                self.queue.append(new_tsk)
            else:
                self.dropped += 1
                self.in_sys -= 1

    def _handle_service_start(self, ttek):
        """Helper to manage task service start when channels are free."""
        for s in self.servers:
            if s.is_free:
                self.taked += 1
                s.start_service(Task(ttek), ttek, False)
                self.free_channels -= 1

                # Check if busy period has started:
                if self.free_channels == 0:
                    if self.in_sys == self.n:
                        self.start_busy = ttek
                break

    def arrival(self, moment=None, ts=None):
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
        self.arrival_time = self.ttek + self.source.generate()

        for _tsk in range(batch_size):
            self.arrived += 1
            self.p[self.in_sys] += self.arrival_time - self.ttek
            self.in_sys += 1

            if self.free_channels == 0:
                self._handle_queueing(self.ttek)
            else:
                self._handle_service_start(self.ttek)
