"""
Simulation model of QS GI/G/n/r and GI/G/n
"""
import time

import numpy as np
from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.sim.utils.distribution_utils import calc_qs_load, create_distribution
from most_queue.sim.utils.exceptions import (
    QsWrongQueueTypeException,
)
from most_queue.sim.utils.qs_queue import QsQueueDeque, QsQueueList
from most_queue.sim.utils.servers import Server
from most_queue.sim.utils.stats_update import refresh_moments_stat
from most_queue.sim.utils.tasks import Task

init()


class QsSim:
    """
    Base class for Queueing System Simulator
    """

    def __init__(self, num_of_channels,
                 buffer=None,
                 verbose=True,
                 buffer_type="list"):
        """
        Initialize the queueing system with GI/G/n/r or GI/G/n model.
        :param num_of_channels: int : number of channels in the system
        :param buffer: Optional(int, None) : maximum length of the queue, None if infinite
        :param verbose: bool : whether to print detailed information during simulation
        :param buffer_type: str : type of the buffer, "list" or "deque"
        """
        self.n = num_of_channels
        self.buffer = buffer
        self.verbose = verbose

        self.generator = np.random.default_rng()

        self.free_channels = self.n
        self.num_of_states = 100000
        self.load = 0  # utilization factor

        # to track the length of the continuous channel occupancy period:
        self.start_busy = 0
        self.busy = [0, 0, 0]
        self.busy_moments = 0

        self.ttek = 0  # current simulation time
        self.total = 0

        self.w = [0, 0, 0]  # initial moments of waiting time in the QS
        self.v = [0, 0, 0]  # initial moments of sojourn time in the QS

        # probabilities of the QS states:
        self.p = [0.0] * self.num_of_states

        self.taked = 0  # number of job accepted for service
        self.served = 0  # number of job serviced by the system
        self.in_sys = 0  # number of job in the system
        self.arrived = 0  # number of job received
        self.dropped = 0  # number of job denied service
        self.arrival_time = 0  # time of arrival of the next job

        # queue of jobs: class - Task
        if buffer_type == "list":
            self.queue = QsQueueList()
        elif buffer_type == "deque":
            self.queue = QsQueueDeque()
        else:
            raise QsWrongQueueTypeException("Unknown queue type")

        self.servers = []  # service channels, list of Server's

        self.source = None
        self.source_params = None
        self.source_kendall_notation = None

        self.server_params = None
        self.server_kendall_notation = None

        self.is_set_source_params = False
        self.is_set_server_params = False

        self.time_spent = 0

        self.zero_wait_arrivals_num = 0

    def set_sources(self, params, kendall_notation: str = 'M'):
        """
        Specifies the type and parameters of source time distribution.
        :param params: dataclass : parameters for the source time distribution
            for example: H2Params for hyper-exponential distribution 
            (see most_queue.general.distribution_params) 
            For 'M' (exponential) params is a float number, that represent single parameter
        :param kendall_notation: str : types of source time distribution ,
           for example: 'H' for hyper-exponential, 'M' for exponential, 'C' for Coxian
        """
        self.source_params = params
        self.source_kendall_notation = kendall_notation

        self.is_set_source_params = True

        self.source = create_distribution(
            params, kendall_notation, self.generator)

        self.arrival_time = self.source.generate()

    def set_servers(self, params, kendall_notation: str = 'M'):
        """
        Specifies the type and parameters of service time distribution.
        :param params: dataclass : parameters for the service time distribution
            for example: H2Params for hyper-exponential distribution 
            (see most_queue.general.distribution_params) 
            For 'M' (exponential) params is a float number, that represent single parameter
        :param kendall_notation: str : types of source time distribution ,
           for example: 'H' for hyper-exponential, 'M' for exponential, 'C' for Coxian
        """
        self.server_params = params
        self.server_kendall_notation = kendall_notation

        self.is_set_server_params = True

        self.servers = [Server(self.server_params, self.server_kendall_notation,
                               generator=self.generator) for _i in range(self.n)]

    def calc_load(self):
        """
        Calculates the load factor of the QS
        """

        return calc_qs_load(self.source_kendall_notation,
                            self.source_params,
                            self.server_kendall_notation,
                            self.server_params, self.n)

    def send_task_to_channel(self, is_warm_start=False, tsk=None):
        """
        Sends a job to the service channel
        """

        if tsk is None:
            tsk = Task(self.ttek)
            tsk.wait_time = 0

        for s in self.servers:
            if s.is_free:
                self.taked += 1
                self.refresh_w_stat(tsk.wait_time)
                self.zero_wait_arrivals_num += 1

                s.start_service(tsk, self.ttek, is_warm_start)
                self.free_channels -= 1

                # Проверям, не наступил ли ПНЗ:
                if self.free_channels == 0:
                    if self.in_sys == self.n:
                        self.start_busy = self.ttek

                break

    def send_task_to_queue(self, new_tsk=None):
        """
        Send Task to Queue
        """

        if new_tsk is None:
            new_tsk = Task(self.ttek)
            new_tsk.start_waiting_time = self.ttek

        if self.buffer is None:  # queue length is not specified, i.e. infinite queue

            self.queue.append(new_tsk)
        else:
            if self.queue.size() < self.buffer:
                self.queue.append(new_tsk)
            else:
                self.dropped += 1
                self.in_sys -= 1

    def arrival(self, moment=None, ts=None):
        """
        Actions upon arrival of the job by the QS.
        """
        
        self.arrived += 1
        self.p[self.in_sys] += self.arrival_time - self.ttek

        if moment:
            self.ttek = moment
            ts.arr_time = moment
            ts.wait_time = 0
            ts.start_waiting_time = -1
            ts.time_to_end_service = 0
            
        else:
            self.ttek = self.arrival_time
            self.arrival_time = self.ttek + self.source.generate()

        self.in_sys += 1
        
        if self.free_channels == 0:
            self.send_task_to_queue(new_tsk=ts)

        else:  # there are free channels:
            self.send_task_to_channel(tsk=ts)

    def serving(self, c, is_network=False):
        """
        Actions upon receipt of a service job with - channel number
        """
        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()

        self.p[self.in_sys] += time_to_end - self.ttek

        self.ttek = time_to_end
        self.served += 1
        self.total += 1
        self.free_channels += 1
        self.refresh_v_stat(self.ttek - end_ts.arr_time)

        self.in_sys -= 1

        # BUSY PERIOD
        if self.queue.size() == 0 and self.free_channels == 1:
            if self.in_sys == self.n - 1:
                # busy period ends
                self.busy_moments += 1
                self.refresh_busy_stat(self.ttek - self.start_busy)

        if self.queue.size() != 0:
            self.send_head_of_queue_to_channel(c, is_network=is_network)
            
        if is_network:
            return end_ts

    def send_head_of_queue_to_channel(self, channel_num, is_network=False):
        """
        Send first Task (head of queue) to Channel
        """
        que_ts = self.queue.pop()

        if self.free_channels == 1:
            self.start_busy = self.ttek

        self.taked += 1
        que_ts.wait_time += self.ttek - que_ts.start_waiting_time
        if is_network:
            que_ts.wait_network += self.ttek - que_ts.start_waiting_time
        self.refresh_w_stat(que_ts.wait_time)

        self.servers[channel_num].start_service(que_ts, self.ttek)
        self.free_channels -= 1

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
        times = [serv_earl, self.arrival_time]
        min_time_num = np.argmin(times)
        if min_time_num == 0:
            # Serving
            self.serving(num_of_server_earlier)
        else:
            # Arrival
            self.arrival()

    def run(self, total_served, is_real_served=True):
        """
        Run simulation process
        """
        start = time.process_time()

        print(Fore.GREEN + '\rStart simulation')

        if is_real_served:

            last_percent = 0

            with tqdm(total=100) as pbar:
                while self.served < total_served:
                    self.run_one_step()
                    percent = int(100*(self.served/total_served))
                    if last_percent != percent:
                        last_percent = percent
                        pbar.update(1)
                        pbar.set_description(Fore.MAGENTA + '\rJob served: ' +
                                             Fore.YELLOW + f'{self.served}/{total_served}' +
                                             Fore.LIGHTGREEN_EX)

        else:
            for _ in tqdm(range(total_served)):
                self.run_one_step()

        print(Fore.GREEN + '\rSimulation is finished')
        print(Style.RESET_ALL)

        self.time_spent = time.process_time() - start

    def refresh_busy_stat(self, new_a):
        """
        Updating statistics of the busy period 
        """
        self.busy = refresh_moments_stat(self.busy, new_a, self.busy_moments)

    def refresh_v_stat(self, new_a):
        """
        Updating statistics of sojourn times
        """
        self.v = refresh_moments_stat(self.v, new_a, self.served)

    def refresh_w_stat(self, new_a):
        """
        Updating statistics of wait times
        """
        self.w = refresh_moments_stat(self.w, new_a, self.taked)

    def get_p(self):
        """
        Returns a list with probabilities of QS states
        p[j] - the probability that there will be exactly j jobs 
        in the QS at a random moment in time
        """
        res = [0.0] * len(self.p)
        for j in range(0, self.num_of_states):
            res[j] = self.p[j] / self.ttek
        return res

    def get_v(self) -> list[float]:
        """
        Returns a list with moments of sojourn times
        v[j] - the j-th moment of sojourn time
        """
        return self.v

    def get_w(self) -> list[float]:
        """
        Returns a list with moments of wait times
        w[j] - the j-th moment of wait time
        """
        return self.w

    def __str__(self, is_short=False):

        res = "Queueing system " + self.source_kendall_notation + \
            "/" + self.server_kendall_notation + "/" + str(self.n)
        if self.buffer is not None:
            res += "/" + str(self.buffer)
        res += f"\nLoad: {self.calc_load():4.3f}\n"
        res += f"Current Time {self.ttek:8.3f}\n"
        res += f"Arrival Time: {self.arrival_time:8.3f}\n"

        res += "Sojourn moments:\n"
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
            res += f"Arrived: {self.arrived}\n"
            if self.buffer is not None:
                res += f"Dropped: {self.dropped}\n"
            res += f"Taken: {self.taked}\n"
            res += f"Served: {self.served}\n"
            res += f"In System: {self.in_sys}\n"
            res += "Busy moments:\n"
            for j in range(3):
                res += f"\t{self.busy[j]:8.4f}"
            res += "\n"
            for c in range(self.n):
                res += str(self.servers[c])
            res += f"\nQueue Count {self.queue.size()}\n"

        return res
