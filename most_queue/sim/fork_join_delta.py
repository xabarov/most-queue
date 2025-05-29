"""
ForkJoin Queue with delta
"""
from colorama import Fore, Style, init

from most_queue.theory.utils.conv import get_self_conv_moments
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.fork_join import ForkJoinSim
from most_queue.sim.utils.tasks import ForkJoinTask

init()


class ForkJoinSimDelta(ForkJoinSim):
    """
    Simulation of ForkJoin queue with delta.
    """

    def __init__(self, num_of_channels, num_of_parts, delta: list[float] | float, 
                 is_sj=False, buffer=None, buffer_type="list", verbose=True):
        """
        :param num_of_channels: int : number of channels (servers)
        :param num_of_parts: int : number of parts on which the task is divided
        :param delta: list[float] | float : If delta is a list, 
            then it should contain the moments of time delay
          caused by reception and restoration operations for each part. 
          If delta is a float, delay is determistic and equal to delta.
        :param is_sj: bool : if True, that means that the model is Split-Join, 
                             otherwise it's Fork-Join.
        :param buffer: Optional(int, None) : maximum length of the queue
        """
        super().__init__(num_of_channels, num_of_parts, is_sj,
                         buffer, buffer_type=buffer_type, verbose=verbose)
        self.delta = delta
        self.subtask_arr_queue = []
        self.serv_task_id = -1
        self.first_subtask_arr_time = {}

    def task_arrival(self):
        """
        actions on task arrival in the system.
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_time - self.ttek
        self.ttek = self.arrival_time
        self.arrival_time = self.ttek + self.source.generate()
        is_dropped = False

        if self.buffer:  # length of the queue is limited
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
            t = ForkJoinTask(self.k, self.ttek)
            self.first_subtask_arr_time[t.id] = self.ttek

            self.in_sys += 1
            self.sub_task_in_sys += 1

            if not self.is_sj:  # Fork-Join discipline

                if self.free_channels == 0:
                    self.queue.append(t.subtasks[0])
                else:  # there are free channels:
                    for s in self.servers:
                        if s.is_free:
                            s.start_service(t.subtasks[0], self.ttek)
                            self.free_channels -= 1
                            break

            else:  # Split-Join discipline

                if self.free_channels < self.k:
                    self.queue.append(t.subtasks[0])
                else:
                    self.servers[0].start_service(t.subtasks[0], self.ttek)
                    self.free_channels -= 1
                    self.serv_task_id = t.id

            for i in range(1, self.k):
                if not isinstance(self.delta, list):
                    t.subtasks[i].future_arr_time = self.ttek + i * self.delta
                else:
                    b_delta = get_self_conv_moments(self.delta, i)
                    params_delta = GammaDistribution.get_params(b_delta)
                    t.subtasks[i].future_arr_time = self.ttek + \
                        GammaDistribution.generate_static(params_delta)
                self.subtask_arr_queue.append(t.subtasks[i])

    def subtask_arrival(self, subtask_num):
        """
        Action to be taken when a subtask arrives in the system.
        """

        subtsk = self.subtask_arr_queue.pop(subtask_num)
        self.p[self.in_sys] += subtsk.future_arr_time - self.ttek
        self.ttek = subtsk.future_arr_time

        is_dropped = False

        if self.buffer:  # length of the queue is limited
            pass

        if not is_dropped:
            self.sub_task_in_sys += 1

            if not self.is_sj:  # Fork-Join discipline

                if self.free_channels == 0:
                    self.queue.append(subtsk)
                else:  # there are free channels:
                    for s in self.servers:
                        if s.is_free:
                            s.start_service(subtsk)
                            self.free_channels -= 1
                            break

            else:  # Split-Join discipline

                if self.free_channels != 0 and self.serv_task_id == subtsk.task_id:
                    for free_c in self.servers:
                        if free_c.is_free:
                            free_c.start_service(subtsk, self.ttek)
                            self.free_channels -= 1
                            break
                else:
                    self.queue.append(subtsk)

    def serving(self, c):
        """
        Action on the arrival of a request for service.
        c - channel number
        """
        time_to_end = self.servers[c].time_to_end_service
        self.p[self.in_sys] += time_to_end - self.ttek
        end_ts = self.servers[c].end_service()
        self.serv_task_id = end_ts.task_id
        self.ttek = time_to_end
        self.served_subtask_in_task[end_ts.task_id] += 1
        self.total += 1
        self.free_channels += 1

        if not self.is_sj:

            if self.served_subtask_in_task[end_ts.task_id] == self.k:
                self.served += 1
                self.refresh_v_stat(
                    self.ttek - self.first_subtask_arr_time[end_ts.task_id])
                self.in_sys -= 1

            if self.queue.size() != 0:
                que_ts = self.queue.pop()
                self.servers[c].start_service(que_ts, self.ttek)
                self.free_channels -= 1

        else:
            if self.served_subtask_in_task[end_ts.task_id] == self.k:

                self.served += 1
                self.refresh_v_stat(self.ttek - end_ts.arr_time)
                self.in_sys -= 1

                if self.queue.size() != 0:

                    new_task_id = self.queue.queue[0].task_id

                    brothers = [q for q in range(
                        self.queue.size()) if self.queue.queue[q].task_id == new_task_id]
                    for q in brothers:
                        que_ts = self.queue.queue[q]
                        for serv in self.servers:
                            if serv.is_free:
                                serv.start_service(que_ts, self.ttek)
                                self.free_channels -= 1
                                break
                    self.queue.queue = [
                        q for q in self.queue.queue if q.task_id != new_task_id]
            else:
                if self.queue.size() != 0:
                    brothers = [q for q in range(
                        self.queue.size()) if self.queue.queue[q].task_id == end_ts.task_id]
                    for q in brothers:
                        que_ts = self.queue.queue[q]
                        for serv in self.servers:
                            if serv.is_free:
                                serv.start_service(que_ts, self.ttek)
                                self.free_channels -= 1
                                break
                    self.queue.queue = [
                        q for q in self.queue.queue if q.task_id != end_ts.task_id]

    def run_one_step(self):
        """
        Runs one step of simulation.
        """

        num_of_server_earlier = -1
        serv_earl = 1e10

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        subtask_arr_time_earl = 1e10
        num_of_subtask_earl = -1

        for c, subtask in enumerate(self.subtask_arr_queue):
            if subtask.future_arr_time < subtask_arr_time_earl:
                subtask_arr_time_earl = subtask.future_arr_time
                num_of_subtask_earl = c

        is_task_arr = False

        if subtask_arr_time_earl > self.arrival_time:
            is_task_arr = True

        arr_time = min(self.arrival_time, subtask_arr_time_earl)

        if arr_time < serv_earl:
            if is_task_arr:
                self.task_arrival()
            else:
                self.subtask_arrival(num_of_subtask_earl)
        else:
            self.serving(num_of_server_earlier)

    def __str__(self, is_short=False):
        """
        :return: string representation of the system
        :param is_short: if True, return short representation of the system
        """
        res = f"{Fore.GREEN}Queueing system {self.source_kendall_notation}/"
        res += f"{self.server_kendall_notation}/{self.n}{Style.RESET_ALL}\n"
        if self.buffer is not None:
            res += f"{Fore.YELLOW}/ {self.buffer}{Style.RESET_ALL}\n"
        if self.is_sj:
            res += f"{Fore.CYAN}| Split-Join{Style.RESET_ALL}\n"
        else:
            res += f"{Fore.CYAN}| Fork-Join{Style.RESET_ALL}\n"

        res += f"{Fore.MAGENTA}Current Time {self.ttek:8.3f}{Style.RESET_ALL}\n"
        res += f"{Fore.MAGENTA}Arrival Time: {self.arrival_time:8.3f}{Style.RESET_ALL}\n"

        res += f"{Fore.CYAN}Sojourn moments:{Style.RESET_ALL}\n"
        for i in range(3):
            res += f"\t{self.v[i]:8.4f}"
        res += "\n"

        if not is_short:
            res += f"{Fore.BLUE}Stationary prob:{Style.RESET_ALL}\n"
            res += "\t"
            for i in range(10):
                res += f"{self.p[i] / self.ttek:6.5f}   "
            res += "\n"
            res += f"{Fore.GREEN}Arrived: {self.arrived}{Style.RESET_ALL}\n"
            if self.buffer is not None:
                res += f"{Fore.RED}Dropped: {self.dropped}{Style.RESET_ALL}\n"
            res += f"{Fore.GREEN}Served: {self.served}{Style.RESET_ALL}\n"
            res += f"{Fore.YELLOW}In System: {self.in_sys}{Style.RESET_ALL}\n"

            for c in range(self.n):
                res += str(self.servers[c])
            res += "\n{Fore.CYAN}Queue Count {self.queue.size()}{Style.RESET_ALL}\n"

        return res
