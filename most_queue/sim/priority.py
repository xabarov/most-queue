"""
Simulation of a priority queue system (GI/G/n/r and GI/G/n systems)
"""

import math

import numpy as np
from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.sim.utils.distribution_utils import create_distribution
from most_queue.sim.utils.servers import ServerPriority
from most_queue.sim.utils.tasks import TaskPriority

init()


class PriorityQueueSimulator:
    """
    Simulation of a priority queue system (GI/G/n/r and GI/G/n systems)
    """

    def __init__(self, num_of_channels, num_of_classes, prty_type='No', buffer=None):
        """
        num_of_channels - number of channels (servers)
        num_of_classes - number of classes of requests
        prty_type - type of priority:
            No  - no priorities, FIFO
            PR  - preemptive resume, with resuming interrupted request
            RS  - preemptive repeat with resampling, re-sampling duration for new service
            RW  - preemptive repeat without resampling, repeating service with previous duration
            NP  - non preemptive, relative priority
        buffer - maximum queue length

        To start the simulation, you need to:
        - call the constructor with parameters
        - set the input arrival distribution using the set_sorces() method
        - set the service distribution using the set_servers() method
        - start the simulation using the run() method
        to which you need to pass the number of job required for servicing

        """

        self.n = num_of_channels
        self.k = num_of_classes
        self.buffer = buffer
        self.prty_type = prty_type
        self.free_channels = self.n
        self.num_of_states = 100000
        self.load = 0  # load factor of the system

        # for tracking the duration of continuous busy periods of channels:
        self.start_busy = -1
        self.busy_moments = [0] * self.k
        self.busy = []
        self.queue = []
        self.class_busy_started = -1
        self.w = []  # wait moments of the system
        self.v = []  # sojourn moments of the system

        # probability of states of the system (number of requests in it j):
        self.p = []
        
        for _ in range(self.k):
            self.queue.append([])

        for _ in range(self.k):
            self.busy.append([0, 0, 0])
            self.w.append([0, 0, 0])
            self.v.append([0, 0, 0])
            self.p.append([0.0] * self.num_of_states)

        self.ttek = 0  # current time of simulation
        self.total = 0

        self.taked = [0] * self.k  # number of tasks taken for service
        self.served = [0] * self.k  # number of tasks served
        self.in_sys = [0] * self.k  # number of tasks in system
        self.t_old = [0.0] * self.k  # moment of the previous event

        self.arrived = [0] * self.k  # number of tasks arrived

        # number of tasks dropped due to lack of resources
        self.dropped = [0] * self.k
        self.arrival_time = [0.0] * self.k  # moment of the next arrival

        self.servers = []  # channels of service, list with Server classes
        self.sources = []  # sources of tasks, list with Source classes

        self.sources_params = None
        self.servers_params = None

        self.is_set_source_params = False
        self.is_set_server_params = False

        self.warm_up = None
        self.is_warm_up_set = False

        self.generator = np.random.default_rng()

    def set_sources(self, sources: list[dict]):
        """
        Set sources of tasks. Each source is a dictionary with the following keys:
        - type: str, distribution type (e.g., 'M', 'E', 'Pa')
        - params: list, distribution parameters (e.g., [mu], [y1, mu1, mu2])

        See supported distributions params in the README.md file or use
            ``` 
            from most_queue.sim.utils.distribution_utils import print_supported_distributions
            print_supported_distributions()
            ```
        """
        self.sources_params = sources
        self.is_set_source_params = True

        for i, source in enumerate(sources):
            source_type = source['type']
            params = source['params']

            self.sources.append(create_distribution(
                params, source_type, self.generator))

            self.arrival_time[i] = self.sources[i].generate()

    def set_servers(self, servers_params):
        """
        Set the parameters of the servers. Each server is represented 
        as a dictionary with the following keys:
        - type: a string representing the distribution type (e.g., 'М', 'Н', 'E', etc.)
        - params: a list of parameters for the distribution

        See supported distributions params in the README.md file or use
            ``` 
            from most_queue.sim.utils.distribution_utils import print_supported_distributions
            print_supported_distributions()
            ```

        """
        self.servers_params = servers_params

        self.is_set_server_params = True

        for _ in range(self.n):
            self.servers.append(
                ServerPriority(self.servers_params, self.prty_type, self.generator))

    def set_warm_up(self, warm_up_params):
        """
        Set the parameters of the warm-up period. Each warm-up period is represented 
        as a dictionary with the following keys:
        - type: a string representing the distribution type (e.g., 'М', 'Н', 'E', etc.)
        - params: a list of parameters for the distribution

        The warm-up period is used to initialize the system before starting the simulation. 
        It helps to avoid transient effects and provides a more accurate estimate 
        of the steady-state behavior of the system.

        See supported distributions params in the README.md file or use
            ``` 
            from most_queue.sim.utils.distribution_utils import print_supported_distributions
            print_supported_distributions()
            ```
        """

        self.is_warm_up_set = True
        self.warm_up = []

        for params in warm_up_params:
            warm_up_type = params['type']
            params = params['params']

            self.warm_up.append(create_distribution(
                params, warm_up_type, self.generator))

    def calc_load(self):
        """
        Calc the load of the system
        """
        l_sum = 0
        b1_sr = 0

        for i in range(self.k):

            if self.sources_params[i]['type'] == "M":
                l_sum += self.sources_params[i]['params']
            elif self.sources_params[i]['type'] == "H":
                y1 = self.sources_params[i]['params'][0]
                y2 = 1.0 - y1
                mu1 = self.sources_params[i]['type'][1]
                mu2 = self.sources_params[i]['type'][2]

                f1 = y1 / mu1 + y2 / mu2
                l_sum += 1.0 / f1

            elif self.sources_params[i]['type'] == "E":
                r = self.sources_params[i]['params'][0]
                mu = self.sources_params[i]['params'][1]
                l_sum += mu / r

            elif self.sources_params[i]['type'] == "Gamma":
                mu = self.sources_params[i]['params'][0]
                alpha = self.sources_params[i]['params'][1]
                l_sum += mu / alpha

            elif self.sources_params[i]['type'] == "C":
                y1 = self.sources_params[i]['params'][0]
                y2 = 1.0 - y1
                mu1 = self.sources_params[i]['params'][1]
                mu2 = self.sources_params[i]['params'][2]

                f1 = y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
                l_sum += 1.0 / f1
            elif self.sources_params[i]['type'] == "Pa":
                if self.sources_params[i]['params'][0] < 1:
                    return None
                else:
                    a = self.sources_params[i]['params'][0]
                    k = self.sources_params[i]['params'][1]
                    f1 = a * k / (a - 1)
                    l_sum += 1.0 / f1
            elif self.sources_params[i]['type'] == "Uniform":
                f1 = self.sources_params[i]['type'][0]
                l_sum += 1.0 / f1

            elif self.sources_params[i]['type'] == "D":
                f1 = self.sources_params[i]['type']
                l_sum += 1.0 / f1

            if self.servers_params[i]['type'] == "M":
                mu = self.servers_params[i]['params']
                b1_sr += 1.0 / mu

            elif self.servers_params[i]['type'] == "H":
                y1 = self.servers_params[i]['params'][0]
                y2 = 1.0 - y1
                mu1 = self.servers_params[i]['params'][1]
                mu2 = self.servers_params[i]['params'][2]

                b1_sr += y1 / mu1 + y2 / mu2

            elif self.servers_params[i]['type'] == "Gamma":
                mu = self.servers_params[i]['params'][0]
                alpha = self.servers_params[i]['params'][1]
                b1_sr += alpha / mu

            elif self.servers_params[i]['type'] == "E":
                r = self.servers_params[i]['params'][0]
                mu = self.servers_params[i]['params'][1]
                b1_sr += r / mu

            elif self.servers_params[i]['type'] == "Uniform":
                f1 = self.servers_params[i]['params'][0]
                b1_sr += 1.0 / f1

            elif self.servers_params[i]['type'] == "D":
                f1 = self.servers_params[i]['type']
                b1_sr += 1.0 / f1

            elif self.servers_params[i]['type'] == "C":
                y1 = self.servers_params[i]['params'][0]
                y2 = 1.0 - y1
                mu1 = self.servers_params[i]['params'][1]
                mu2 = self.servers_params[i]['params'][2]

                b1_sr += y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
            elif self.servers_params[i]['type'] == "Pa":
                if self.servers_params[i]['params'][0] < 1:
                    return math.inf
                else:
                    a = self.servers_params[i]['params'][0]
                    k = self.servers_params[i]['params'][1]
                    b1_sr += a * k / (a - 1)

        return l_sum * b1_sr / (self.n * self.k)

    def arrival(self, k, moment=None, ts=None):
        """
        Action of arrival of a job to the system
        :param k: class of job
        :param moment: time of arrival
        :param ts: task 
        """
        if moment:
            self.ttek = moment
            self.p[k][self.in_sys[k]] += moment - self.t_old[k]
            new_tsk = ts
            new_tsk.in_node_class_num = k
            new_tsk.arr_time = moment
            # в текущем узле обнуляем время ожидания. Общее время ожидания - ts.wait_network
            new_tsk.wait_time = 0
            new_tsk.is_pr = False
            new_tsk.start_waiting_time = -1
            new_tsk.time_to_end_service = 0

        else:
            self.ttek = self.arrival_time[k]
            self.p[k][self.in_sys[k]] += self.arrival_time[k] - self.t_old[k]
            self.arrival_time[k] = self.ttek + self.sources[k].generate()
            new_tsk = TaskPriority(k, self.ttek)

        self.arrived[k] += 1
        self.in_sys[k] += 1
        self.t_old[k] = self.ttek

        if self.free_channels != 0:
            self.arrive_on_free_channels(moment, k, new_tsk)
            return

        # All servers are busy. Check priority type and add task to queue or start service.

        if self.prty_type == 'No':

            self.arrive_with_no_prty(new_tsk, k)
            return

        if self.prty_type in ("PR", "RS", "RW"):

            self.arrive_with_absolute_prty(moment, new_tsk, k)
            return

        # Priority type is "NP", check if there are any free channels

        new_tsk.start_waiting_time = self.ttek
        self.queue[k].append(new_tsk)

    def arrive_on_free_channels(self, moment, k, new_tsk):
        """
        Arrive on free channels.
         If there are no free channels and the system is warm up, then start service on the first server.
         Otherwise, start service on the first available server.
         If there are no free channels and the system is not warm up, then start service on the first available server. 
        """
        if self.free_channels == self.n and self.is_warm_up_set is True:
            self.taked[k] += 1
            if moment:
                self.servers[0].start_service(
                    new_tsk, self.ttek, self.warm_up[k], is_network=True)
            else:
                self.servers[0].start_service(
                    new_tsk, self.ttek, self.warm_up[k])
            self.free_channels -= 1
        else:
            for s in self.servers:
                if s.is_free:
                    self.taked[k] += 1
                    if moment:
                        s.start_service(new_tsk, self.ttek,
                                        is_network=True)
                    else:
                        s.start_service(new_tsk, self.ttek)
                    self.free_channels -= 1
                    break

        #  check for busy periods
        if self.free_channels == 0:
            if self.in_sys[k] == self.n:
                self.start_busy = self.ttek
                self.class_busy_started = k

    def arrive_with_no_prty(self, new_tsk, k):
        """
        Action of arriving a new task with no priority.
        new_tsk - new task to be added to the queue.
        k - class of the task.

        """
        if not self.buffer:  # no buffer, infinite queue length
            new_tsk.start_waiting_time = self.ttek
            self.queue[k].append(new_tsk)
        else:

            if len(self.queue) < self.buffer:
                new_tsk.start_waiting_time = self.ttek
                self.queue[k].append(new_tsk)
            else:
                self.dropped[k] += 1
                self.in_sys[k] -= 1

    def arrive_with_absolute_prty(self, moment, new_tsk, k):
        """
        Action of arriving a new task with absolute priority.
        moment - current simulation moment.
        new_tsk - new task to be added to the queue.
        k - class of the task.

        """

        # look for the task with lower priority
        is_found_weekier = False
        for c in self.servers:
            if moment:
                class_on_service = c.tsk_on_service.in_node_class_num
            else:
                class_on_service = c.tsk_on_service.k

            if class_on_service > k:
                time_to_end = c.time_to_end_service
                total_time = c.total_time_to_serve

                dropped_tsk = c.end_service()
                self.taked[k] += 1

                if k != self.class_busy_started and self.class_busy_started != -1 and self.in_sys[
                        k] == self.n:
                    self.busy_moments[self.class_busy_started] += 1
                    self.refresh_busy_stat(
                        self.class_busy_started, self.ttek - self.start_busy)
                    self.start_busy = self.ttek
                    self.class_busy_started = k

                dropped_tsk.start_waiting_time = self.ttek
                dropped_tsk.is_pr = True
                if self.prty_type == 'PR':
                    dropped_tsk.time_to_end_service = time_to_end - self.ttek
                elif self.prty_type == "RS":
                    dropped_tsk.time_to_end_service = c.dist[k].generate(
                    )
                elif self.prty_type == "RW":
                    dropped_tsk.time_to_end_service = total_time

                is_found_weekier = True
                if moment:
                    self.queue[dropped_tsk.in_node_class_num].append(
                        dropped_tsk)
                    c.start_service(
                        new_tsk, self.ttek, is_network=True)
                else:
                    self.queue[dropped_tsk.k].append(dropped_tsk)
                    c.start_service(new_tsk, self.ttek)

                break
        if not is_found_weekier:
            if not self.buffer:  # infinite buffer

                new_tsk.start_waiting_time = self.ttek
                self.queue[k].append(new_tsk)
            else:
                total_queue_length = 0
                for q in self.queue:
                    total_queue_length += len(q)

                if total_queue_length < self.buffer:
                    new_tsk.start_waiting_time = self.ttek
                    self.queue[k].append(new_tsk)
                else:
                    self.dropped[k] += 1
                    self.in_sys[k] -= 1

    def serving(self, c, is_network=False):
        """
        Action of serving a task in the server c.
        c - number of server
        is_network - is the system part of a network or not.
        """
        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()
        if is_network:
            k = end_ts.in_node_class_num
        else:
            k = end_ts.k

        self.p[k][self.in_sys[k]] += time_to_end - self.t_old[k]

        self.ttek = time_to_end
        self.t_old[k] = self.ttek
        self.served[k] += 1
        self.total += 1
        self.free_channels += 1
        self.refresh_v_stat(k, self.ttek - end_ts.arr_time)
        self.refresh_w_stat(k, end_ts.wait_time)
        self.in_sys[k] -= 1


        if self.prty_type != "No":
            if len(self.queue[k]) == 0 and self.free_channels == 1:
                if self.in_sys[k] == self.n - 1 and self.class_busy_started != -1:
                    # End of busy period
                    self.busy_moments[k] += 1
                    self.refresh_busy_stat(k, self.ttek - self.start_busy)


            start_number = 0
            if self.prty_type == "PR" or self.prty_type == "RS" or self.prty_type == "RW":
                # we can only look at the queue starting from the current class number
                start_number = k

            for kk in range(start_number, self.k):
                if len(self.queue[kk]) != 0:

                    que_ts = self.queue[kk].pop(0)

                    if self.free_channels == 1 and kk != end_ts.k:
                        self.start_busy = self.ttek
                        self.class_busy_started = kk

                    self.taked[kk] += 1
                    que_ts.wait_time += self.ttek - que_ts.start_waiting_time
                    if is_network:
                        que_ts.wait_network += self.ttek - que_ts.start_waiting_time
                        self.servers[c].start_service(
                            que_ts, self.ttek, is_network=True)
                    else:
                        self.servers[c].start_service(que_ts, self.ttek)

                    self.free_channels -= 1
                    break
        else:
            if len(self.queue[0]) == 0 and self.free_channels == 1:
                if self.in_sys == self.n - 1 and self.class_busy_started != -1:
                    # End of busy period
                    self.busy_moments += 1
                    self.refresh_busy_stat(0, self.ttek - self.start_busy)
            # one queue
            if len(self.queue[0]) != 0:

                que_ts = self.queue[0].pop(0)

                if self.free_channels == 1 and k != end_ts.k:
                    self.start_busy[k] = self.ttek
                    self.class_busy_started = k

                self.taked[k] += 1
                que_ts.wait_time += self.ttek - que_ts.start_waiting_time
                self.servers[c].start_service(que_ts, self.ttek)
                self.free_channels -= 1
        if is_network:
            return end_ts

    def swap_queue(self, last_class, new_class):
        """
        Swap two queues based on class priority.
        """
        buf = self.queue[last_class]
        self.queue[last_class] = self.queue[new_class]
        self.queue[new_class] = buf

    def run_one_step(self):
        """
        Run one step of the simulation.
        """

        num_of_server_earlier = -1
        serv_earl = 1e10

        k_earlier = -1
        arrival_earlier = 1e10

        for kk in range(self.k):
            if self.arrival_time[kk] < arrival_earlier:
                arrival_earlier = self.arrival_time[kk]
                k_earlier = kk

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        # Key moment:

        if arrival_earlier < serv_earl:
            self.arrival(k_earlier)

        else:
            self.serving(num_of_server_earlier)

    def run(self, total_served, is_real_served=True):
        """
        Run simulation for total_served jobs.
        :param total_served: int, number of jobs to serve
        :param is_real_served: bool, if True then show progress bar and print job served percentage.
        :return: None
        """

        print(Fore.GREEN + '\rStart simulation')
        if is_real_served:

            last_percent = 0

            with tqdm(total=100) as pbar:
                while sum(self.served) < total_served:
                    self.run_one_step()
                    percent = int(100*(sum(self.served)/total_served))
                    if last_percent != percent:
                        last_percent = percent
                        pbar.update(1)
                        pbar.set_description(Fore.MAGENTA + '\rJob served: ' +
                                             Fore.YELLOW + f'{sum(self.served)}/{total_served}' + Fore.LIGHTGREEN_EX)

        else:
            for i in tqdm(range(total_served)):
                self.run_one_step()

        print(Fore.GREEN + '\rSimulation is finished')
        print(Style.RESET_ALL)

    def refresh_busy_stat(self, k, new_a):
        """
        Refresh statistics of busy time for class k and new arrival rate new_a.
        :param k: class number
        :param new_a: new arrival rate
        :return: None
        """
        for i in range(3):
            self.busy[k][i] = self.busy[k][i] * (1.0 - (1.0 / self.busy_moments[k])) + \
                math.pow(new_a, i + 1) / self.busy_moments[k]

    def refresh_v_stat(self, k, new_a):
        """
        Refresh statistics of soujorn time for class k and new arrival rate new_a.
        :param k: class number
        :param new_a: new arrival rate
        """
        for i in range(3):
            self.v[k][i] = self.v[k][i] * \
                (1.0 - (1.0 / self.served[k])) + \
                math.pow(new_a, i + 1) / self.served[k]

    def refresh_w_stat(self, k, new_a):
        """
        Refresh statistics of waiting time for class k and new arrival rate new_a.
        :param k: class number
        :param new_a: new arrival rate

        """
        for i in range(3):
            self.w[k][i] = self.w[k][i] * \
                (1.0 - (1.0 / self.served[k])) + \
                math.pow(new_a, i + 1) / self.served[k]

    def get_p(self):
        """
        Get the probability distribution of states for each class.
        Returns a list with probabilities of states in the queueing system.
        p[k][j] - the probability that there are exactly j requests of class k at random time

        """
        res = []
        for kk in range(self.k):
            res.append([0.0] * len(self.p[kk]))
            for j in range(0, self.num_of_states):
                res[kk][j] = self.p[kk][j] / self.ttek
        return res

    def __str__(self, is_short=False):
        """
        Representation of the queueing system with priorities.
        :param is_short: if True, then only the number of channels and classes are shown.
        :return: string representation of the queueing system.
        """

        res = f"{Fore.GREEN}Queueing system {Style.RESET_ALL}"
        is_the_same_source = True
        first_source_type = self.sources_params[0]['type']
        for kk in range(1, self.k):
            if self.sources_params[kk]['type'] != first_source_type:
                is_the_same_source = False
        if is_the_same_source:
            res += f"{Fore.GREEN}{first_source_type}*/{Style.RESET_ALL}"
        else:
            for kk in range(self.k - 1):
                res += f"{Fore.GREEN}{self.sources_params[kk]['type']},{Style.RESET_ALL}"
            res += f"{Fore.GREEN}{self.sources_params[self.k - 1]['type']}/{Style.RESET_ALL}"

        is_the_same_serving_type = True
        first_serv_type = self.servers_params[0]['type']
        for kk in range(1, self.k):
            if self.servers_params[kk]['type'] != first_serv_type:
                is_the_same_serving_type = False
        if is_the_same_serving_type:
            res += f"{Fore.GREEN}{first_serv_type}/{Style.RESET_ALL}"
        else:
            for kk in range(self.k - 1):
                res += f"{Fore.GREEN}{self.servers_params[kk]['type']},{Style.RESET_ALL}"
            res += f"{Fore.GREEN}{self.servers_params[self.k - 1]['type']}/{Style.RESET_ALL}"

        res += f"{Fore.BLUE}{str(self.n)}{Style.RESET_ALL}"

        if self.buffer is not None:
            res += f"/{Fore.YELLOW}{str(self.buffer)}{Style.RESET_ALL}"
        if self.prty_type != 'No':
            res += f"/{Fore.CYAN}{self.prty_type}{Style.RESET_ALL}"

        res += f"\n{Fore.MAGENTA}Load: {self.calc_load():.3f}{Style.RESET_ALL}\n"
        if not is_short:
            res += f"{Fore.LIGHTGREEN_EX}Current Time {self.ttek:.3f}{Style.RESET_ALL}\n"
        for kk in range(self.k):
            res += f"\n{Fore.CYAN}Class {kk + 1}{Style.RESET_ALL}\n"
            if not is_short:
                res += f"{Fore.LIGHTBLUE_EX}\tArrival time: {self.arrival_time[kk]:.3f}{Style.RESET_ALL}\n"
            res += "\tSojourn moments:\n"
            for i in range(3):
                res += f"\t{Fore.GREEN}{self.v[kk][i]:8.4g}\t"

            res += "\n\tWait moments:\n"
            for i in range(3):
                res += f"\t{Fore.YELLOW}{self.w[kk][i]:8.4g}\t"
            res += "\n"
            if not is_short:
                res += "\tStationary prob:\n\t"
                for i in range(10):
                    res += f"{Fore.CYAN}{self.p[kk][i] / self.ttek:8.4g}   "
                res += "\n"
                res += f"\tArrived: {self.arrived[kk]}{Style.RESET_ALL}\n"
                if self.buffer is not None:
                    res += f"\tDropped: {self.dropped[kk]}{Style.RESET_ALL}\n"
                res += f"\tTaken: {self.taked[kk]}{Style.RESET_ALL}\n"
                res += f"\tServed: {self.served[kk]}{Style.RESET_ALL}\n"
                res += f"\tIn System: {self.in_sys[kk]}{Style.RESET_ALL}\n"
                res += "\tBusy moments:\n\t"
                for j in range(3):
                    res += f"{Fore.MAGENTA}{self.busy[kk][j]:8.4g}    "
                res += "\n"
        for c in range(self.n):
            if self.servers[c].is_free:
                res += f"Server {c + 1}: Free\n"
            else:
                res += str(self.servers[c]) + "\n"

        for kk in range(self.k):
            res += f"{Fore.GREEN}Queue #{kk + 1}{Style.RESET_ALL} count {len(self.queue[kk])}\n"

        return res
