"""
Simulation model for queueing systems with impatient tasks
"""
from colorama import Fore, Style, init

from most_queue.sim.queueing_systems.fifo import QueueingSystemSimulator
from most_queue.sim.utils.distribution_utils import create_distribution
from most_queue.sim.utils.tasks import ImpatientTask

init()

class ImpatientQueueSim(QueueingSystemSimulator):
    """
    Queueing system with impatient tasks
    """

    def __init__(self, num_of_channels, buffer=None, verbose=True):
        """
        Initialize the queueing system with impatient tasks.
        The system can have a finite or infinite buffer. 

        :param num_of_channels: int : number of channels in the system
        :param buffer: Optional(int, None) : maximum length of the queue, None if infinite
        :param verbose: bool : whether to print detailed information during simulation  

        """
        super().__init__(num_of_channels, buffer, verbose)

        self.impatience_params = None
        self.impatience_types = None

        self.impatience = None
        self.is_set_impatience_params = False

    def set_impatience(self, params, types):
        """
        Set the impatience distribution for tasks.

        :param params: list : parameters for the impatience distribution
        :param types: list : types of the impatience distribution (e.g. 'exp', 'norm')

        """
        self.impatience_params = params
        self.impatience_types = types

        self.is_set_impatience_params = True

        self.impatience = create_distribution(params, types, self.generator)

    def arrival(self):
        """
        Actions on arrival of a task.
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_time - self.t_old

        self.in_sys += 1
        self.ttek = self.arrival_time
        self.t_old = self.ttek
        self.arrival_time = self.ttek + self.source.generate()

        moment_to_leave = self.ttek + self.impatience.generate()

        if self.free_channels == 0:
            if self.buffer is None:  # не задана длина очереди, т.е бесконечная очередь
                new_tsk = ImpatientTask(self.ttek, moment_to_leave)
                new_tsk.start_waiting_time = self.ttek
                self.queue.append(new_tsk)
            else:
                if self.queue.size() < self.buffer:
                    new_tsk = ImpatientTask(self.ttek, moment_to_leave)
                    new_tsk.start_waiting_time = self.ttek
                    self.queue.append(new_tsk)
                else:
                    self.dropped += 1
                    self.in_sys -= 1

        else:  # there are free channels:

            # check if its a warm phase:
            is_warm_start = False
            if self.queue.size() == 0 and self.free_channels == self.n and self.warm_phase.is_set:
                is_warm_start = True

            for s in self.servers:
                if s.is_free:
                    self.taked += 1
                    s.start_service(ImpatientTask(
                        self.ttek, moment_to_leave), self.ttek, is_warm_start)
                    self.free_channels -= 1

                    # Проверям, не наступил ли ПНЗ:
                    if self.free_channels == 0:
                        if self.in_sys == self.n:
                            self.start_busy = self.ttek
                    break

    def drop_task(self, num_of_task_earlier, moment_to_leave_earlier):
        """
        Drop a task from the queue and update statistics.
        """
        self.ttek = moment_to_leave_earlier
        new_queue = []
        end_ts = None
        for i, tsk in enumerate(self.queue.queue):
            if i != num_of_task_earlier:
                new_queue.append(tsk)
            else:
                end_ts = self.queue.queue[i]

        self.queue.queue = new_queue
        self.in_sys -= 1
        self.dropped += 1
        self.served += 1
        
        if end_ts is not None:
            self.refresh_v_stat(self.ttek - end_ts.arr_time)
            end_ts.wait_time = self.ttek - end_ts.arr_time
            self.refresh_w_stat(end_ts.wait_time)

    def run_one_step(self):
        """
        Run one step of the simulation.
        """

        num_of_server_earlier = -1
        serv_earl = 1e10

        for c in range(self.n):
            if self.servers[c].time_to_end_service < serv_earl:
                serv_earl = self.servers[c].time_to_end_service
                num_of_server_earlier = c

        num_of_task_earlier = -1
        moment_to_leave_earlier = 1e10
        for i, tsk in enumerate(self.queue.queue):
            if tsk.moment_to_leave < moment_to_leave_earlier:
                moment_to_leave_earlier = tsk.moment_to_leave
                num_of_task_earlier = i

        # Key moment:

        if self.arrival_time < serv_earl:
            if self.arrival_time < moment_to_leave_earlier:
                self.arrival()
            else:
                self.drop_task(num_of_task_earlier, moment_to_leave_earlier)
        else:
            if serv_earl < moment_to_leave_earlier:
                self.serving(num_of_server_earlier)
            else:
                self.drop_task(num_of_task_earlier, moment_to_leave_earlier)

    def __str__(self, is_short=False):
        """
        Returns a string representation of the system.
         If `is_short` is True, it returns a short version of the string.
         Otherwise, it returns a detailed version of the string.
        """
        super().__str__(is_short)
        num_of_task_earlier = -1
        moment_to_leave_earlier = 1e10
        for i, tsk in enumerate(self.queue.queue):
            if tsk.moment_to_leave < moment_to_leave_earlier:
                moment_to_leave_earlier = tsk.moment_to_leave
                num_of_task_earlier = i
        return f'{Fore.GREEN}Task {num_of_task_earlier}{Style.RESET_ALL} is earlier to leave at {Fore.YELLOW}{moment_to_leave_earlier:8.3f}{Style.RESET_ALL}'