"""
Simulation model of QS GI/G/n/r and GI/G/n with negative jobs
"""
import random
from enum import Enum

import numpy as np

from most_queue.sim.base import QsSim
from most_queue.sim.utils.distribution_utils import calc_qs_load, create_distribution
from most_queue.sim.utils.stats_update import refresh_moments_stat
from most_queue.theory.negative.structs import NegativeArrivalsResults


class NegativeServiceType(Enum):
    """
    Negative service types
    """
    DISASTER = 1  # remove all customers
    RCS = 2  # remove customer in service
    RCH = 3  # remove customer at the Head
    RCE = 4  # remove customer at the End


class QsSimNegatives(QsSim):
    """
    Simulation model of QS GI/G/n/r and GI/G/n with negative jobs
    """

    def __init__(self, num_of_channels: int,
                 type_of_negatives: NegativeServiceType = NegativeServiceType.DISASTER,
                 buffer: int | None = None,
                 verbose: bool = True,
                 buffer_type="list"):
        """
        Initialize the queueing system with GI/G/n/r or GI/G/n model.
        :param num_of_channels: int : number of channels in the system
        :param buffer: Optional(int, None) : maximum length of the queue, None if infinite
        :param verbose: bool : whether to print detailed information during simulation
        :param buffer_type: str : type of the buffer, "list" or "deque"
        """

        super().__init__(num_of_channels, buffer, verbose, buffer_type)

        self.type_of_negatives = type_of_negatives

        # initial moments of sojourn time of successfully served
        self.v_served = [0, 0, 0]
        # initial moments of sojourn time of broken by negative jobs
        self.v_broken = [0, 0, 0]

        self.served = 0  # number of job serviced by the system without negative job breaks
        self.broken = 0  # number of job broken by negatives
        self.total = 0  # number of job broken by negatives and served without breaks

        # Positive arrivals
        self.positive_arrived = 0  # number of job received
        self.positive_arrival_time = 0  # time of arrival of the next job
        self.positive_source = None
        self.positive_source_params = None
        self.positive_source_kendall_notation = None
        self.is_set_positive_source_params = False

        # Negative arrivals
        self.negative_arrived = 0  # number of job received
        self.negative_arrival_time = 0  # time of arrival of the next job
        self.negative_source = None
        self.negative_source_params = None
        self.negative_source_kendall_notation = None
        self.is_set_negative_source_params = False

    def set_positive_sources(self, params, kendall_notation: str = 'M'):
        """
        Specifies the type and parameters of positives source time distribution.
        :param params: dataclass : parameters for the positives source time distribution
            for example: H2Params for hyper-exponential distribution 
            (see most_queue.general.distribution_params) 
            For 'M' (exponential) params is a float number, that represent single parameter
        :param kendall_notation: str : types of positives source time distribution ,
           for example: 'H' for hyper-exponential, 'M' for exponential, 'C' for Coxian
        """
        self.positive_source_params = params
        self.positive_source_kendall_notation = kendall_notation

        self.is_set_positive_source_params = True

        self.positive_source = create_distribution(
            params, kendall_notation, self.generator)

        self.positive_arrival_time = self.positive_source.generate()

    def set_negative_sources(self, params, kendall_notation: str = 'M'):
        """
        Specifies the type and parameters of negative source time distribution.
        :param params: dataclass : parameters for the negative source time distribution
            for example: H2Params for hyper-exponential distribution 
            (see most_queue.general.distribution_params)
            For 'M' (exponential) params is a float number, that represent single parameter
        :param kendall_notation: str : types of negative source time distribution ,
           for example: 'H' for hyper-exponential, 'M' for exponential, 'C' for Coxian
        """
        self.negative_source_params = params
        self.negative_source_kendall_notation = kendall_notation

        self.is_set_negative_source_params = True

        self.negative_source = create_distribution(
            params, kendall_notation, self.generator)

        self.negative_arrival_time = self.negative_source.generate()

    def calc_positive_load(self):
        """
        Calculates the load factor of the QS if has no disatsers
        """

        return calc_qs_load(self.positive_source_kendall_notation,
                            self.positive_source_params,
                            self.server_kendall_notation,
                            self.server_params, self.n)

    def positive_arrival(self):
        """
        Actions upon arrival of positive job by the QS.
        """

        self.positive_arrived += 1
        self.p[self.in_sys] += self.positive_arrival_time - self.ttek

        self.in_sys += 1
        self.ttek = self.positive_arrival_time
        self.positive_arrival_time = self.ttek + self.positive_source.generate()

        if self.free_channels == 0:
            self.send_task_to_queue()
        else:
            self.send_task_to_channel()

    def negative_arrival(self):
        """
        Actions upon arrival of negative job by the QS.
        """

        self.negative_arrived += 1
        self.p[self.in_sys] += self.negative_arrival_time - self.ttek
        self.ttek = self.negative_arrival_time
        self.negative_arrival_time = self.ttek + self.negative_source.generate()

        if self.in_sys == 0:
            # If no jobs in system, negatives has no effect
            return

        if self.type_of_negatives == NegativeServiceType.DISASTER:

            not_free_servers = [c for c in range(
                self.n) if not self.servers[c].is_free]
            for c in not_free_servers:
                end_ts = self.servers[c].end_service()
                self.broken += 1
                self.total += 1
                sojourn_time = self.ttek - end_ts.arr_time
                self.refresh_v_stat(sojourn_time)
                self.refresh_v_stat_broken(sojourn_time)

            self.in_sys = 0
            self.free_channels = self.n

            while self.queue:
                ts = self.queue.pop()
                ts.wait_time += self.ttek - ts.start_waiting_time
                self.taked += 1
                self.total += 1
                self.broken += 1
                self.refresh_w_stat(ts.wait_time)

                sojourn_time = self.ttek - ts.arr_time
                self.refresh_v_stat(sojourn_time)
                self.refresh_v_stat_broken(sojourn_time)

        elif self.type_of_negatives == NegativeServiceType.RCE:
            self.in_sys -= 1
            ts = self.queue.tail()

            ts.wait_time += self.ttek - ts.start_waiting_time

            self.taked += 1
            self.total += 1
            self.broken += 1

            self.refresh_w_stat(ts.wait_time)
            sojourn_time = self.ttek - ts.arr_time
            self.refresh_v_stat(sojourn_time)
            self.refresh_v_stat_broken(sojourn_time)

        elif self.type_of_negatives == NegativeServiceType.RCH:
            self.in_sys -= 1
            ts = self.queue.pop()
            ts.wait_time += self.ttek - ts.start_waiting_time

            self.taked += 1
            self.total += 1
            self.broken += 1

            self.refresh_w_stat(ts.wait_time)
            sojourn_time = self.ttek - ts.arr_time
            self.refresh_v_stat(sojourn_time)
            self.refresh_v_stat_broken(sojourn_time)

        elif self.type_of_negatives == NegativeServiceType.RCS:

            not_free_servers = [c for c in range(
                self.n) if not self.servers[c].is_free]
            c = random.choice(not_free_servers)
            end_ts = self.servers[c].end_service()
            self.total += 1
            self.broken += 1
            self.free_channels += 1
            sojourn_time = self.ttek - end_ts.arr_time
            self.refresh_v_stat(sojourn_time)
            self.refresh_v_stat_broken(sojourn_time)

            self.in_sys -= 1

            if self.queue.size() != 0:
                self.send_head_of_queue_to_channel(c)

    def serving(self, c):
        """
        Actions upon receipt of a service job with - channel number
        """
        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()

        self.p[self.in_sys] += time_to_end - self.ttek

        self.ttek = time_to_end
        self.free_channels += 1

        self.served += 1
        self.total += 1
        sojourn_time = self.ttek - end_ts.arr_time
        self.refresh_v_stat(sojourn_time)
        self.refresh_v_stat_served(sojourn_time)

        self.in_sys -= 1

        if self.queue.size() != 0:
            self.send_head_of_queue_to_channel(c)

    def run_one_step(self):
        """
        Run Open step of simulation
        """

        num_of_server_earlier = -1
        serv_earl = 1e16

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        # Global warm-up is set. Need to track
        # including the moment of warm-up end
        times = [serv_earl, self.positive_arrival_time,
                 self.negative_arrival_time]
        min_time_num = np.argmin(times)
        if min_time_num == 0:
            # Serving
            self.serving(num_of_server_earlier)
        elif min_time_num == 1:
            # Arrival positive
            self.positive_arrival()
        else:
            # Arrival negative
            self.negative_arrival()

    def refresh_v_stat(self, new_a):
        """
        Updating statistics of sojourn times (all, broken and successfully served)
        """
        self.v = refresh_moments_stat(self.v, new_a, self.total)

    def refresh_v_stat_broken(self, new_a):
        """
        Updating statistics of sojourn times of broken jobs
        """
        self.v_broken = refresh_moments_stat(
            self.v_broken, new_a, self.broken)

    def refresh_v_stat_served(self, new_a):
        """
        Updating statistics of sojourn times of successfully served jobs
        """
        self.v_served = refresh_moments_stat(self.v_served, new_a, self.served)

    def get_v_served(self):
        """
        Returns initial moments of sojourn time (only for successfully served jobs)
        """
        return self.v_served

    def get_v_broken(self):
        """
        Returns initial moments of sojourn time  (only for broken by negative arrivals)
        """
        return self.v_broken

    def get_results(self, max_p: int = 100) -> NegativeArrivalsResults:
        """
        Returns results as a NegativeArrivalsResults object
        max_p: Maximum number of probabilities to return
        """
        return NegativeArrivalsResults(
            p=self.get_p()[:max_p],
            v=self.v,
            w=self.w,
            v_served=self.v_served,
            v_broken=self.v_broken)

    def __str__(self, is_short=False):

        type_of_neg_str = ''
        if self.type_of_negatives == NegativeServiceType.DISASTER:
            type_of_neg_str = 'Disaster'
        elif self.type_of_negatives == NegativeServiceType.RCE:
            type_of_neg_str = 'Remove customer from the End of the Queue'
        elif self.type_of_negatives == NegativeServiceType.RCH:
            type_of_neg_str = 'Remove customer from the Head of the Queue'
        elif self.type_of_negatives == NegativeServiceType.RCS:
            type_of_neg_str = 'Remove customer from the Service'
        else:
            type_of_neg_str = 'Unknown'

        res = "Queueing system " + self.positive_source_kendall_notation + \
            "/" + self.server_kendall_notation + \
            "/" + str(self.n) + type_of_neg_str
        if self.buffer is not None:
            res += "/" + str(self.buffer)
        res += f"\nLoad: {self.calc_positive_load():4.3f}\n"
        res += f"Current Time {self.ttek:8.3f}\n"
        res += f"Positive arrival Time: {self.positive_arrival_time:8.3f}\n"
        res += f"Negative arrival Time: {self.negative_arrival_time:8.3f}\n"

        res += "Sojourn moments of served jobs:\n"
        for i in range(3):
            res += f"\t{self.v[i]:8.4f}"
        res += "\n"

        res += "Wait moments:\n"
        for i in range(3):
            res += f"\t{self.w[i]:8.4f}"
        res += "\n"

        if not is_short:
            res += "Stationary prob:\n"
            res += "\t"
            for i in range(10):
                res += f"{self.p[i] / self.ttek:6.5f}\t"
            res += "\n"
            res += f"Positive arrived: {self.positive_arrived}\n"
            if self.buffer is not None:
                res += f"Positive dropped: {self.dropped}\n"
            res += f"Negative arrived: {self.negative_arrived}\n"
            res += f"Taken: {self.taked}\n"
            res += f"Served: {self.served}\n"
            res += f"In System: {self.in_sys}\n"
            res += "\n"
            for c in range(self.n):
                res += str(self.servers[c])
            res += f"\nQueue Count {self.queue.size()}\n"

        return res
