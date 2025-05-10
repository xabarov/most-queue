"""
Simulation model of QS GI/G/n/r and GI/G/n with negative jobs
"""
import random
import time
from enum import Enum

import numpy as np
from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.sim.utils.distribution_utils import calc_qs_load, create_distribution
from most_queue.sim.utils.exceptions import QsWrongQueueTypeException
from most_queue.sim.utils.qs_queue import QsQueueDeque, QsQueueList
from most_queue.sim.utils.servers import Server
from most_queue.sim.utils.stats_update import refresh_moments_stat
from most_queue.sim.utils.tasks import Task

init()


class NegativeServiceType(Enum):
    """
    Negative service types
    """
    DISASTER = 1  # remove all customers
    RCS = 2  # remove customer in service
    RCH = 3  # remove customer at the Head
    RCE = 4  # remove customer at the End


class QueueingSystemSimulatorWithNegatives:
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
        self.n = num_of_channels
        self.buffer = buffer
        self.verbose = verbose
        self.type_of_negatives = type_of_negatives

        self.generator = np.random.default_rng()

        self.free_channels = self.n
        self.num_of_states = 100000
        self.load = 0  # utilization factor

        self.ttek = 0  # current simulation time

        self.w = [0, 0, 0]  # initial moments of waiting time in the QS
        self.v_served = [0, 0, 0]  # initial moments of sojourn time of served job in the QS
        self.v_breaked = [0, 0, 0]  # initial moments of sojourn time of breaked job in the QS

        # probabilities of the QS states:
        self.p = [0.0] * self.num_of_states

        self.taked = 0  # number of job accepted for service
        self.served = 0  # number of job serviced by the system
        self.breaked = 0  # number of job breaked by negative arrivals in the system
        self.in_sys = 0  # number of job in the system

        # Positive arrivals
        self.positive_arrived = 0  # number of job received
        self.positive_dropped = 0  # number of job denied service
        self.positive_arrival_time = 0  # time of arrival of the next job
        self.positive_source = None
        self.positive_source_params = None
        self.positive_source_types = None
        self.is_set_positive_source_params = False

        # Negative arrivals
        self.negative_arrived = 0  # number of job received
        self.negative_arrival_time = 0  # time of arrival of the next job
        self.negative_source = None
        self.negative_source_params = None
        self.negative_source_types = None
        self.is_set_negative_source_params = False

        # queue of jobs: class - Task
        if buffer_type == "list":
            self.queue = QsQueueList()
        elif buffer_type == "deque":
            self.queue = QsQueueDeque()
        else:
            raise QsWrongQueueTypeException("Unknown queue type")

        self.servers = []  # service channels, list of Server's

        self.server_params = None
        self.server_types = None

        self.is_set_server_params = False

        self.time_spent = 0

    def set_positive_sources(self, params, types):
        """
        Specifies the type and parameters of positives source time distribution.
        :param params: list : parameters for the positives source time distribution
        :param types: list : types of positives source time distribution 

        """
        self.positive_source_params = params
        self.positive_source_types = types

        self.is_set_positive_source_params = True

        self.positive_source = create_distribution(
            params, types, self.generator)

        self.positive_arrival_time = self.positive_source.generate()

    def set_negative_sources(self, params, types):
        """
        Specifies the type and parameters of negative source time distribution.
        :param params: list : parameters for the negative source time distribution
        :param types: list : types of negative source time distribution 

        """
        self.negative_source_params = params
        self.negative_source_types = types

        self.is_set_negative_source_params = True

        self.negative_source = create_distribution(
            params, types, self.generator)

        self.negative_arrival_time = self.negative_source.generate()

    def set_servers(self, params, types):
        """
        Specifies the type and parameters of service time distribution.
        :param params: list : parameters for the service time distribution
        :param types: list : types of service time distribution 
        """
        self.server_params = params
        self.server_types = types

        self.is_set_server_params = True

        self.servers = [Server(self.server_params, self.server_types,
                               generator=self.generator) for i in range(self.n)]

    def calc_positive_load(self):
        """
        Calculates the load factor of the QS if has no disatsers
        """

        return calc_qs_load(self.positive_source_types,
                            self.positive_source_params,
                            self.server_types,
                            self.server_params, self.n)

    def send_task_to_channel(self):
        """
        Sends a job to the service channel
        """
        for s in self.servers:
            if s.is_free:
                tsk = Task(self.ttek)
                tsk.wait_time = 0
                self.taked += 1
                self.refresh_w_stat(tsk.wait_time)

                s.start_service(tsk, self.ttek, False)
                self.free_channels -= 1

                break

    def send_task_to_queue(self):
        """
        Send Task to Queue
        """
        if self.buffer is None:  # queue length is not specified, i.e. infinite queue
            new_tsk = Task(self.ttek)
            new_tsk.start_waiting_time = self.ttek
            self.queue.append(new_tsk)
        else:
            if self.queue.size() < self.buffer:
                new_tsk = Task(self.ttek)
                new_tsk.start_waiting_time = self.ttek
                self.queue.append(new_tsk)
            else:
                self.positive_dropped += 1
                self.in_sys -= 1

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
            self.in_sys = 0
            self.free_channels = self.n
            while self.queue:
                ts = self.queue.pop()
                ts.wait_time += self.ttek - ts.start_waiting_time
                self.taked += 1
                self.breaked += 1
                self.refresh_w_stat(ts.wait_time)
                self.refresh_v_stat_breaked(self.ttek - ts.arr_time)

        elif self.type_of_negatives == NegativeServiceType.RCE:
            self.in_sys -= 1
            ts = self.queue.tail()
            ts.wait_time += self.ttek - ts.start_waiting_time
            self.taked += 1
            self.breaked += 1
            self.refresh_w_stat(ts.wait_time)
            self.refresh_v_stat_breaked(self.ttek - ts.arr_time)

        elif self.type_of_negatives == NegativeServiceType.RCH:
            self.in_sys -= 1
            ts = self.queue.pop()
            ts.wait_time += self.ttek - ts.start_waiting_time
            self.taked += 1
            self.breaked += 1
            self.refresh_w_stat(ts.wait_time)
            self.refresh_v_stat_breaked(self.ttek - ts.arr_time)

        elif self.type_of_negatives == NegativeServiceType.RCS:

            not_free_servers = [c for c in range(self.n) if not self.servers[c].is_free]
            c = random.choice(not_free_servers)
            
            end_ts = self.servers[c].end_service()
            self.breaked += 1
            self.free_channels += 1
            self.refresh_v_stat_breaked(self.ttek - end_ts.arr_time)

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
        self.served += 1
        self.free_channels += 1
        self.refresh_v_stat_served(self.ttek - end_ts.arr_time)

        self.in_sys -= 1

        if self.queue.size() != 0:
            self.send_head_of_queue_to_channel(c)

    def send_head_of_queue_to_channel(self, channel_num):
        """
        Send first Task (head of queue) to Channel
        """
        que_ts = self.queue.pop()

        self.taked += 1
        que_ts.wait_time += self.ttek - que_ts.start_waiting_time
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

    def run(self, total_served, is_real_served=True):
        """
        Run simulation process
        """

        # Check if is_set_server_params is set
        if not self.is_set_server_params:
            raise ValueError('Server parameters are not set. Please call set_servers() first.')
        # Check if is_set_positive_source_params
        if not self.is_set_positive_source_params:
            raise ValueError('Positive source parameters are not set. Please call set_positive_sources() first.')
        # Check if is_set_negative_source_params
        if not self.is_set_negative_source_params:
            raise ValueError('Negative source parameters are not set. Please call set_negative_sources() first.')
        
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

    def refresh_v_stat_served(self, new_a):
        """
        Updating statistics of sojourn times of served jobs
        """
        self.v_served = refresh_moments_stat(self.v_served, new_a, self.served)
        
        
    def refresh_v_stat_breaked(self, new_a):
        """
        Updating statistics of sojourn times of breaked jobs
        """
        self.v_breaked = refresh_moments_stat(self.v_breaked, new_a, self.breaked)

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
    
    def get_w(self):
        """
        Returns waiting time initial moments
        """
        return self.w
    
    def get_v_served(self):
        """
        Returns initial moments of soujourn time served jobs
        """
        return self.v_served
    
    def get_v_breaked(self):
        """
        Returns initial moments of soujourn time breaked jobs
        """
        return self.v_breaked

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

        res = "Queueing system " + self.positive_source_types + \
            "/" + self.server_types + "/" + str(self.n) + type_of_neg_str
        if self.buffer is not None:
            res += "/" + str(self.buffer)
        res += f"\nLoad: {self.calc_positive_load():4.3f}\n"
        res += f"Current Time {self.ttek:8.3f}\n"
        res += f"Positive arrival Time: {self.positive_arrival_time:8.3f}\n"
        res += f"Negative arrival Time: {self.negative_arrival_time:8.3f}\n"

        res += "Sojourn moments of served jobs:\n"
        for i in range(3):
            res += f"\t{self.v_served[i]:8.4f}"
        res += "\n"
        
        res += "Sojourn moments of breaked jobs:\n"
        for i in range(3):
            res += f"\t{self.v_breaked[i]:8.4f}"
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
                res += f"Positive dropped: {self.positive_dropped}\n"
            res += f"Negative arrived: {self.negative_arrived}\n"
            res += f"Taken: {self.taked}\n"
            res += f"Served: {self.served}\n"
            res += f"In System: {self.in_sys}\n"
            res += "\n"
            for c in range(self.n):
                res += str(self.servers[c])
            res += f"\nQueue Count {self.queue.size()}\n"

        return res
