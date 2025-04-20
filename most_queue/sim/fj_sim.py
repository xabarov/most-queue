"""
ForkJoin Queue
"""
import math

from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.sim.qs_sim import QueueingSystemSimulator
from most_queue.sim.utils.tasks import ForkJoinTask

init()


class ForkJoinSim(QueueingSystemSimulator):
    """
    Simulation of a ForkJoin Queueing System (Fork-Join, Split-Join)
    """

    def __init__(self, num_of_channels, k, is_sj=False, is_purge=False, buffer=None):
        """
        :param num_of_channels: int : number of channels in the system
        :param k: int : number of parts a task is split into
        :param is_sj: bool : if True, then Split-Join model is used, otherwise Fork-Join model is used
        :param buffer: Optional(int, None) : maximum length of the queue
        :param is_purge: bool : if True, then Purge model is used

        """
        QueueingSystemSimulator.__init__(self, num_of_channels, buffer)
        self.k = k
        self.is_sj = is_sj
        self.is_purge = is_purge
        self.served_subtask_in_task = {}
        self.sub_task_in_sys = 0

        self.queues = []
        for _ in range(num_of_channels):
            self.queues.append([])

    def calc_load(self):
        """
        calculates the load of the system.
        """

        pass

    def arrival(self):
        """
        action on arrival of a task in the system.
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_time - self.t_old
        self.ttek = self.arrival_time
        self.t_old = self.ttek
        self.arrival_time = self.ttek + self.source.generate()

        is_dropped = False

        if self.buffer:  # ограниченная длина очереди
            if not self.is_sj:

                if self.queue.size() + self.k - 1 > self.buffer + self.free_channels:
                    self.dropped += 1
                    is_dropped = True
            else:
                if self.free_channels == 0 and self.queue.size() + self.k - 1 > self.buffer:
                    self.dropped += 1
                    is_dropped = True

        if not is_dropped:
            self.served_subtask_in_task[ForkJoinTask.task_id] = 0
            t = ForkJoinTask(self.n, self.ttek)
            self.in_sys += 1
            self.sub_task_in_sys += self.n

            if not self.is_sj:  # Fork-Join discipline

                for i in range(self.n):
                    if self.free_channels == 0:
                        self.queues[i].append(t.subtasks[i])
                    else:  # there are free channels:
                        if self.servers[i].is_free:
                            self.servers[i].start_service(
                                t.subtasks[i], self.ttek)
                            self.free_channels -= 1
                        else:
                            self.queues[i].append(t.subtasks[i])

            else:  # Split-Join discipline

                if self.free_channels < self.n:
                    for i in range(self.n):
                        self.queue.append(t.subtasks[i])
                else:
                    for i in range(self.n):
                        self.servers[i].start_service(t.subtasks[i], self.ttek)
                        self.free_channels -= 1

    def serving(self, c):
        """
        Action when a service is completed on channel c.
        :param c: int : number of channel where service is completed.
        """
        time_to_end = self.servers[c].time_to_end_service
        self.p[self.in_sys] += time_to_end - self.t_old
        end_ts = self.servers[c].end_service()
        self.ttek = time_to_end
        self.t_old = self.ttek
        self.served_subtask_in_task[end_ts.task_id] += 1
        self.total += 1
        self.free_channels += 1
        self.sub_task_in_sys -= 1

        if not self.is_sj:

            if self.served_subtask_in_task[end_ts.task_id] == self.k:

                if self.is_purge:
                    # найти все остальные подзадачи в СМО и выкинуть
                    task_id = end_ts.task_id
                    for i in range(self.n):
                        if self.servers[i].tsk_on_service.task_id == task_id:
                            self.servers[c].end_service()
                    for i in range(self.n):
                        for j in range(len(self.queues[i])):
                            if self.queues[i][j].task_id == task_id:
                                self.queues[i].pop(j)

                self.served += 1
                self.refresh_v_stat(self.ttek - end_ts.arr_time)
                self.in_sys -= 1

            if len(self.queues[c]) != 0:
                que_ts = self.queues[c].pop(0)
                self.servers[c].start_service(que_ts, self.ttek)
                self.free_channels -= 1

        else:
            if self.served_subtask_in_task[end_ts.task_id] == self.n:

                self.served += 1
                self.refresh_v_stat(self.ttek - end_ts.arr_time)
                self.in_sys -= 1

                if self.queue.size() != 0:
                    for i in range(self.n):
                        que_ts = self.queue.pop()
                        self.servers[i].start_service(que_ts, self.ttek)
                        self.free_channels -= 1

    def run_one_step(self):
        """
        Run one step of the simulation.
         If there is an arrival before the earliest service completion time,
         then process the arrival. Otherwise, process the earliest service completion.
        """

        num_of_server_earlier = -1
        serv_earl = 1e10

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        # Key moment:

        if self.arrival_time < serv_earl:
            self.arrival()
        else:
            self.serving(num_of_server_earlier)

    def run(self, total_served, is_real_served=True):
        """
        Run simulation until total_served tasks are served.
        """

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
                                             Fore.YELLOW + f'{self.served}/{total_served}' + Fore.LIGHTGREEN_EX)

        else:
            print(Fore.GREEN + '\rStart simulation')
            print(Style.RESET_ALL)
            # print(Back.YELLOW + 'на желтом фоне')

            for i in tqdm(range(total_served)):
                self.run_one_step()

            print(Fore.GREEN + '\rSimulation is finished')
            print(Style.RESET_ALL)

    def refresh_v_stat(self, new_a):
        for i in range(3):
            self.v[i] = self.v[i] * (1.0 - (1.0 / self.served)) + \
                math.pow(new_a, i + 1) / self.served

    def get_p(self):
        """
        Get probabilities of states for Fork-Join model
        """
        res = [0.0] * len(self.p)
        for j in range(0, self.num_of_states):
            res[j] = self.p[j] / self.ttek
        return res

    def __str__(self, is_short=False):
        """
        Prints the model of the queueing system
        :param is_short: bool : if True, then short information about the model of the queueing system is returned
        :return: str : string representation of the model of the queueing system
        """

        res = "Queueing system " + self.source_types + \
            "/" + self.server_types + "/" + str(self.n)
        if self.buffer is not None:
            res += "/" + str(self.buffer)
        if self.is_sj:
            res += '| Split-Join'
        else:
            res += '| Fork-Join'

        res += "\n"
        # res += "Load: " + "{0:4.3f}".format(self.calc_load()) + "\n"
        res += f"Current Time {self.ttek:8.3f}\n"
        res += f"Arrival Time: {self.arrival_time:8.3f}\n"

        res += "Sojourn moments:\n"
        for i in range(3):
            res += f"\t{self.v[i]:8.4f}"
        res += "\n"

        if not is_short:
            res += "Stationary prob:\n"
            res += "\t"
            for i in range(10):
                res += f"{self.p[i] / self.ttek:6.5f}   "
            res += "\n"
            res += f"Arrived: {self.arrived}\n"
            if self.buffer is not None:
                res += f"Dropped: {self.dropped}\n"
            res += f"Served: {self.served}\n"
            res += f"In System: {self.in_sys}\n"

            for c in range(self.n):
                res += str(self.servers[c])
            res += "\nQueue Count " + str(self.queue.size()) + "\n"

        return res
