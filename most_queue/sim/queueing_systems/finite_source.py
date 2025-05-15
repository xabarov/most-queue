"""
Simulating a queueing system with finite number of sources.
"""
import time

import numpy as np
from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.sim.queueing_systems.base import QsSim, Task
from most_queue.sim.utils.distribution_utils import create_distribution

init()

class QueueingFiniteSourceSim(QsSim):
    """
    Simulating a queueing system with finite number of sources.
    """

    def __init__(self, num_of_channels: int, m: int, buffer=None, verbose=True, buffer_type="list"):
        """
        num_of_channels - number of channels in the system.
        m - number of sources.
        buffer - maximum length of the queue.
        verbose - print comments during simulation.
        
        To start the simulation, you need to:
        - call the constructor with parameters
        - set the input arrival distribution using the set_sorces() method
        - set the service distribution using the set_servers() method
        - start the simulation using the run() method
        to which you need to pass the number of job required for servicing

        See supported distributions params in the README.md file or use
            ``` 
            from most_queue.sim.utils.distribution_utils import print_supported_distributions
            print_supported_distributions()
            ```
            
        """
        self.m = m
        self.sources_left = m  # how many sources are ready to send requests

        super().__init__(num_of_channels, buffer=buffer, verbose=verbose, buffer_type=buffer_type)

        self.arrival_times = []
        self.arrived_num = -1
        self.p = [0.0] * (m+1)

    def set_sources(self, params, types):
        """
        Set source parameters and types
        :param params: list of lists, where each sublist contains parameters for a source distribution
        :param types: list of strings, where each string specifies the type of distribution for a source
        :return: None

        See supported distributions params in the README.md file or use
            ``` 
            from most_queue.sim.utils.distribution_utils import print_supported_distributions
            print_supported_distributions()
            ```
        """
        self.source_params = params
        self.source_types = types

        self.is_set_source_params = True

        self.source = create_distribution(params, types, self.generator)

        self.arrival_times = [self.source.generate() for i in range(self.m)]

    def arrival(self):
        """
        Action on arrival of a request to the queue.
        :return: None
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_times[self.arrived_num] - self.t_old
        self.sources_left -= 1
        self.in_sys += 1
        self.ttek = self.arrival_times[self.arrived_num]
        self.t_old = self.ttek
        self.arrival_times[self.arrived_num] = 1e10

        if self.free_channels == 0:
            if self.buffer is None:  # infinite queue

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
            for s in self.servers:
                if s.is_free:
                    self.taked += 1
                    s.start_service(Task(self.ttek), self.ttek, False)
                    self.free_channels -= 1

                    # Проверям, не наступил ли ПНЗ:
                    if self.free_channels == 0:
                        if self.in_sys == self.n:
                            self.start_busy = self.ttek
                    break

    def serving(self, c):
        """
        Action when the service is completed.
        c - channel number.
        """
        self.sources_left += 1

        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()

        self.p[self.in_sys] += time_to_end - self.t_old

        self.ttek = time_to_end

        for i, a in enumerate(self.arrival_times):
            if a > 9.9e9:
                self.arrival_times[i] = self.ttek + self.source.generate()
                break

        self.t_old = self.ttek
        self.served += 1
        self.total += 1
        self.free_channels += 1
        self.refresh_v_stat(self.ttek - end_ts.arr_time)
        self.refresh_w_stat(end_ts.wait_time)
        self.in_sys -= 1

        if len(self.queue) == 0 and self.free_channels == 1:
            if self.in_sys == self.n - 1:
                # Конец ПНЗ
                self.busy_moments += 1
                self.refresh_busy_stat(self.ttek - self.start_busy)

        if len(self.queue) != 0:

            que_ts = self.queue.pop()

            if self.free_channels == 1:
                self.start_busy = self.ttek

            self.taked += 1
            que_ts.wait_time += self.ttek - que_ts.start_waiting_time
            self.servers[c].start_service(que_ts, self.ttek)
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
        times = [serv_earl]
        for t in self.arrival_times:  # Arrival times
            times.append(t)

        min_time_num = np.argmin(times)
        if min_time_num == 0:
            # Serving
            self.serving(num_of_server_earlier)
        else:
            self.arrived_num = min_time_num - 1
            # Arrival of a new customer
            self.arrival()


    def get_p(self):
        """
        Get probabilities of states.
        Returns list with probabilities of states.
        p[j] - probability that in random moment of time there will be exactly j requests in the system
        """
        res = [0.0] * len(self.p)
        for j in range(0, self.m+1):
            res[j] = self.p[j] / self.ttek
        return res

    def __str__(self, is_short=False):
        """
        Representation of the queueing system
        :param is_short: if True then return short representation
        :return: string with information about the queueing system
        """

        res = f"{Fore.GREEN}Queueing system {self.source_types}/{self.server_types}/{self.n}{Style.RESET_ALL}\n"
        if self.buffer is not None:
            res += f"{Fore.GREEN}/ {self.buffer}{Style.RESET_ALL}\n"
        res += f"{Fore.YELLOW}Load: {self.calc_load():4.3f}{Style.RESET_ALL}\n"
        res += f"{Fore.YELLOW}Current Time {self.ttek:8.3f}{Style.RESET_ALL}\n"
        for i, a in enumerate(self.arrival_times):
            res += f"Arrival Time of Source {i + 1}: {a:8.3f}\n"

        res += f"{Fore.CYAN}Sojourn moments:{Style.RESET_ALL}\n"
        for i in range(3):
            res += f"\t{self.v[i]:8.4f}\n"

        res += f"{Fore.CYAN}Wait moments:{Style.RESET_ALL}\n"
        for i in range(3):
            res += f"\t{self.w[i]:8.4f}\n"

        if not is_short:
            res += f"{Fore.MAGENTA}Stationary prob:{Style.RESET_ALL}\n"
            res += "\t"
            for i in range(10):
                res += f"{self.p[i] / self.ttek:6.5f}{Style.RESET_ALL}   "
            res += "\n"
            res += f"{Fore.MAGENTA}Arrived: {self.arrived}{Style.RESET_ALL}\n"
            if self.buffer is not None:
                res += f"{Fore.MAGENTA}Dropped: {self.dropped}{Style.RESET_ALL}\n"
            res += f"{Fore.MAGENTA}Taken: {self.taked}{Style.RESET_ALL}\n"
            res += f"{Fore.MAGENTA}Served: {self.served}{Style.RESET_ALL}\n"
            res += f"{Fore.MAGENTA}In System: {self.in_sys}{Style.RESET_ALL}\n"
            res += f"{Fore.MAGENTA}Busy moments:{Style.RESET_ALL}\n"
            for j in range(3):
                res += f"\t{self.busy[j]:8.4f}{Style.RESET_ALL}    "
            res += "\n"
            for c in range(self.n):
                res += str(self.servers[c])
            res += "\nQueue Count {self.queue.size()}{Style.RESET_ALL}\n"

        return res
