"""
Simulation model of QS GI/G/n/r and GI/G/n
"""
import time

import numpy as np
from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.sim.utils.distribution_utils import calc_qs_load, create_distribution
from most_queue.sim.utils.exceptions import (
    QsSourseSettingException,
    QsWrongQueueTypeException,
)
from most_queue.sim.utils.phase import QsPhase
from most_queue.sim.utils.qs_queue import QsQueueDeque, QsQueueList
from most_queue.sim.utils.servers import Server
from most_queue.sim.utils.stats_update import refresh_moments_stat
from most_queue.sim.utils.tasks import Task

init()


class QueueingSystemSimulator:
    """
    Simulation model of QS GI/G/n/r and GI/G/n
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
        self.t_old = 0  # time of the previous event
        self.arrived = 0  # number of job received
        self.dropped = 0  # number of job denied service
        self.arrival_time = 0  # time of arrival of the next job

        self.time_to_next_event = 0

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
        self.source_types = None

        self.server_params = None
        self.server_types = None

        self.is_set_source_params = False
        self.is_set_server_params = False

        self.warm_phase = QsPhase("WarmUp")
        self.cold_phase = QsPhase("Cold")
        self.cold_delay_phase = QsPhase("ColdDelay")

        self.warm_after_cold_starts = 0
        self.zero_wait_arrivals_num = 0

        self.time_spent = 0

    def set_warm(self, params, kendall_notation):
        """
        Set the type and parameters of the warm-up time distribution

        :param params: list : parameters for the warm-up time distribution
        :param kendall_notation: str : Kendall notation for the warm-up time distribution

        """
        dist = create_distribution(params, kendall_notation, self.generator)
        self.warm_phase.set_dist(dist)

    def set_cold(self, params, kendall_notation):
        """
        Set the type and parameters of the cooling time distribution
        :param params: list : parameters for the cooling time distribution
        :param kendall_notation: str : Kendall notation for the cooling time distribution
        """
        dist = create_distribution(params, kendall_notation, self.generator)
        self.cold_phase.set_dist(dist)

    def set_cold_delay(self, params, kendall_notation):
        """
        Set the type and parameters of the cooling start delay time distribution
        :param params: list : parameters for the cooling start delay time distribution
        :param kendall_notation: str : Kendall notation for the cooling start delay time distribution
        """

        if not self.cold_phase.is_set:
            raise QsSourseSettingException(
                "You must first set the cooling time. Use the set_cold() method.")

        dist = create_distribution(params, kendall_notation, self.generator)
        self.cold_delay_phase.set_dist(dist)

    def set_sources(self, params, types):
        """
        Specifies the type and parameters of source time distribution.
        :param params: list : parameters for the source time distribution
        :param types: list : types of source time distribution 

        """
        self.source_params = params
        self.source_types = types

        self.is_set_source_params = True

        self.source = create_distribution(params, types, self.generator)

        self.arrival_time = self.source.generate()

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

    def calc_load(self):
        """
        Calculates the load factor of the QS
        """

        return calc_qs_load(self.source_types,
                            self.source_params,
                            self.server_types,
                            self.server_params, self.n)

    def send_task_to_channel(self, is_warm_start=False):
        """
        Sends a job to the service channel
        is_warm_start - is warming up needed
        """
        for s in self.servers:
            if s.is_free:
                tsk = Task(self.ttek)
                tsk.wait_time = 0
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
                self.dropped += 1
                self.in_sys -= 1

    def arrival(self):
        """
        Actions upon arrival of the job by the QS.
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_time - self.t_old

        self.in_sys += 1
        self.ttek = self.arrival_time
        self.t_old = self.ttek
        self.arrival_time = self.ttek + self.source.generate()

        if self.free_channels == 0:
            self.send_task_to_queue()

        else:  # there are free channels:

            if self.cold_phase.is_set:
                if self.cold_phase.is_start:
                    # Cooling is not finished yet. In queue
                    self.send_task_to_queue()
                    return

            if self.cold_delay_phase.is_set:
                if self.cold_delay_phase.is_start:
                    # The job was received before the end of the cooling start delay time
                    self.cold_delay_phase.is_start = False
                    self.cold_delay_phase.end_time = 1e16
                    self.cold_delay_phase.prob += self.ttek - self.cold_delay_phase.start_mom
                    self.send_task_to_channel()
                    return

            if self.warm_phase.is_set:
                # Warm-up set

                # Check warm-up. By this point the system is definitely not in cooling mode
                # and not in the cooling start delay state.
                # So either:
                # 1. In warm-up mode -> send the job to the queue
                # 2. It is empty and was turned off after cooling. Start warm-up
                # 3. Not empty and warmed up -> then send it for maintenance
                if self.warm_phase.is_start:
                    # 1. In warm-up mode -> send the job to the queue
                    self.send_task_to_queue()
                else:
                    if self.free_channels == self.n:
                        # 2. It is empty and was turned off after cooling. We start warm-up
                        self.warm_phase.start(self.ttek)
                        # to queue
                        self.send_task_to_queue()
                    else:
                        # 3. Not empty and warmed up -> then we send it for servicing
                        self.send_task_to_channel()

            else:
                # No warm-up. Send a job to the service channel
                self.send_task_to_channel()

    def serving(self, c):
        """
        Actions upon receipt of a service job with - channel number
        """
        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()

        self.p[self.in_sys] += time_to_end - self.t_old

        self.ttek = time_to_end
        self.t_old = self.ttek
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

        # COLD
        if self.cold_phase.is_set:
            if self.queue.size() == 0 and self.free_channels == self.n:
                # The system has become empty.
                # 1. If a delay in the start of cooling is set, we play the time of its end
                # 2. If not, we start cooling
                if self.cold_delay_phase.is_set:
                    self.cold_delay_phase.start(self.ttek)
                else:
                    self.cold_phase.start(self.ttek)

        if self.queue.size() != 0:
            self.send_head_of_queue_to_channel(c)

    def send_head_of_queue_to_channel(self, channel_num):
        """
        Send first Task (head of queue) to Channel
        """
        que_ts = self.queue.pop()

        if self.free_channels == 1:
            self.start_busy = self.ttek

        self.taked += 1
        que_ts.wait_time += self.ttek - que_ts.start_waiting_time
        self.refresh_w_stat(que_ts.wait_time)

        self.servers[channel_num].start_service(que_ts, self.ttek)
        self.free_channels -= 1

    def on_end_warming(self):
        """
        Job that has to be done after WarmUp Period Ends
        """

        self.p[self.in_sys] += self.warm_phase.end_time - self.t_old

        self.ttek = self.warm_phase.end_time
        self.t_old = self.ttek

        self.warm_phase.end(self.ttek)

        # Send n jobs from the queue to the channels
        for i in range(self.n):
            if self.queue.size() != 0:
                self.send_head_of_queue_to_channel(i)

    def on_end_cold(self):
        """
        Job that has to be done after Cold Period Ends
        """
        self.p[self.in_sys] += self.cold_phase.end_time - self.t_old

        self.ttek = self.cold_phase.end_time
        self.t_old = self.ttek

        self.cold_phase.end(self.ttek)

        if self.warm_phase.is_set:
            if self.queue.size() != 0:
                # We start warming up only if there are jobs accumulated in the queue.
                self.warm_after_cold_starts += 1
                self.warm_phase.start(self.ttek)

        else:
            # Send n jobs from the queue to the channels
            for i in range(self.n):
                if self.queue.size() != 0:
                    self.send_head_of_queue_to_channel(i)

    def on_end_cold_delay(self):
        """
        Job that has to be done after Cold Delay Period Ends
        """
        self.p[self.in_sys] += self.cold_delay_phase.end_time - self.t_old

        self.ttek = self.cold_delay_phase.end_time
        self.t_old = self.ttek

        self.cold_delay_phase.end(self.ttek)

        # Start the cooling process
        self.cold_phase.start(self.ttek)

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
        times = [serv_earl, self.arrival_time, self.warm_phase.end_time,
                 self.cold_phase.end_time, self.cold_delay_phase.end_time]
        min_time_num = np.argmin(times)
        if min_time_num == 0:
            # Serving
            self.serving(num_of_server_earlier)
        elif min_time_num == 1:
            # Arrival
            self.arrival()
        elif min_time_num == 2:
            # Warm-up ends
            self.on_end_warming()
        elif min_time_num == 3:
            # Cold ends
            self.on_end_cold()
        else:
            # Delay cold ends
            self.on_end_cold_delay()

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

    def get_warmup_prob(self):
        """
        Returns probability of the system being in the warm-up phase
        """
        return self.warm_phase.get_prob(self.ttek)

    def get_cold_prob(self):
        """
        Returns probability of the system being in the cold phase
        """
        return self.cold_phase.get_prob(self.ttek)

    def get_cold_delay_prob(self):
        """
        Returns probability of the system being in the delay cold phase
        """
        return self.cold_delay_phase.get_prob(self.ttek)

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

    def __str__(self, is_short=False):

        res = "Queueing system " + self.source_types + \
            "/" + self.server_types + "/" + str(self.n)
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
