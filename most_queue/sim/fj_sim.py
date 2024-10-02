"""
ForkJoin Queue
"""
import math
import sys

from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.sim.qs_sim import QueueingSystemSimulator
from most_queue.sim.utils.tasks import ForkJoinTask

init()


class ForkJoinSim(QueueingSystemSimulator):
    """
    Имитационная модель СМО Fork-Join, Split-Join
    """

    def __init__(self, num_of_channels, k, is_SJ=False, is_Purge=False, buffer=None):
        """
        num_of_channels - количество каналов СМО
        buffer - максимальная длина очереди
        """
        QueueingSystemSimulator.__init__(self, num_of_channels, buffer)
        self.k = k
        self.is_SJ = is_SJ
        self.is_Purge = is_Purge
        self.served_subtask_in_task = {}
        self.sub_task_in_sys = 0

        self.queues = []
        for i in range(num_of_channels):
            self.queues.append([])

    def calc_load(self):
        """
        вычисляет коэффициент загрузки СМО
        """

        pass

    def arrival(self):
        """
        Действия по прибытию заявки в СМО.
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_time - self.t_old
        self.ttek = self.arrival_time
        self.t_old = self.ttek
        self.arrival_time = self.ttek + self.source.generate()

        is_dropped = False

        if self.buffer:  # ограниченная длина очереди
            if not self.is_SJ:

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

            if not self.is_SJ:  # Fork-Join discipline

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
        Дейтсвия по поступлению заявки на обслуживание
        с - номер канала
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

        if not self.is_SJ:

            if self.served_subtask_in_task[end_ts.task_id] == self.k:

                if self.is_Purge:
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

        if is_real_served:
            
            last_percent = 0

            with tqdm(total=100, unit='jobs') as pbar:
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
        Возвращает список с вероятностями состояний СМО
        p[j] - вероятность того, что в СМО в случайный момент времени будет ровно j заявок
        """
        res = [0.0] * len(self.p)
        for j in range(0, self.num_of_states):
            res[j] = self.p[j] / self.ttek
        return res

    def __str__(self, is_short=False):

        res = "Queueing system " + self.source_types + \
            "/" + self.server_types + "/" + str(self.n)
        if self.buffer != None:
            res += "/" + str(self.buffer)
        if self.is_SJ:
            res += '| Split-Join'
        else:
            res += '| Fork-Join'

        res += "\n"
        # res += "Load: " + "{0:4.3f}".format(self.calc_load()) + "\n"
        res += "Current Time " + "{0:8.3f}".format(self.ttek) + "\n"
        res += "Arrival Time: " + "{0:8.3f}".format(self.arrival_time) + "\n"

        res += "Sojourn moments:\n"
        for i in range(3):
            res += "\t" + "{0:8.4f}".format(self.v[i])
        res += "\n"

        if not is_short:
            res += "Stationary prob:\n"
            res += "\t"
            for i in range(10):
                res += "{0:6.5f}".format(self.p[i] / self.ttek) + "   "
            res += "\n"
            res += "Arrived: " + str(self.arrived) + "\n"
            if self.buffer != None:
                res += "Dropped: " + str(self.dropped) + "\n"
            res += "Served: " + str(self.served) + "\n"
            res += "In System:" + str(self.in_sys) + "\n"

            for c in range(self.n):
                res += str(self.servers[c])
            res += "\nQueue Count " + str(self.queue.size()) + "\n"

        return res
