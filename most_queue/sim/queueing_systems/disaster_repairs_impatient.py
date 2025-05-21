"""
Simulation model of QS GI/G/n/r and GI/G/n with negative jobs
"""
import numpy as np

from most_queue.sim.queueing_systems.negative import (NegativeServiceType,
                                                      QsSimNegatives)
from most_queue.sim.utils.distribution_utils import create_distribution
from most_queue.sim.utils.tasks import ImpatientTaskWithRepairs
from most_queue.sim.utils.stats_update import refresh_moments_stat


class QsSimNegativeImpatience(QsSimNegatives):
    """
    Simulation model of QS GI/G/n/r and GI/G/n with negative jobs and impatience

        Consider a system operating as an M/G/1 queue. The system as a whole suffers
    random disastrous failures (catastrophes) such that, when a failure occurs, all present
    customers are flushed out of the system and lost. The system then goes through a
    repair process whose duration is random. 

        Meanwhile, while the system is down and inoperative, the stream of arrivals continues. 

        However, the new arrivals become impatient: each customer, upon arrival, 
    activates his own timer, with random duration T, such that if the system is still down 
    when the timer expires, the customer abandons the system never to return. 
    """

    def __init__(self, num_of_channels: int,
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

        super().__init__(num_of_channels, type_of_negatives=NegativeServiceType.DISASTER, buffer=buffer,
                         verbose=verbose, buffer_type=buffer_type)

        self.is_leave_time_set = False
        self.leave_time_params = None
        self.leave_time_types = None
        self.leave_num = 0
        self.leave_time_distribution = None

        self.is_repair_time_set = False
        self.repair_time_params = None
        self.repair_time_types = None
        self.repair_time_distribution = None
        self.repair_end_time = 1e16
        self.repair_num = 0

        self.is_system_in_repair = False

        self.served_in_up = 0  # number of job serviced by the system in up state
        self.served_in_down = 0  # number of job serviced by the system in down state
        self.v_up = [0, 0, 0]
        self.v_down = [0, 0, 0]

    def set_leave_time(self, params, types):
        """
        Specifies the type and parameters of impatience (leave) time distribution.
        :param params: list : parameters for the impatience (leave) time  distribution
        :param types: list : types of distribution 

        """
        self.leave_time_params = params
        self.leave_time_types = types

        self.is_leave_time_set = True

        self.leave_time_distribution = create_distribution(
            params, types, self.generator)

    def set_repair_time(self, params, types):
        """
        Specifies the type and parameters of repair time distribution.
        :param params: list : parameters for the repair time distribution
        :param types: list : types of distribution 

        """
        self.repair_time_params = params
        self.repair_time_types = types

        self.is_repair_time_set = True

        self.repair_time_distribution = create_distribution(
            params, types, self.generator)

    def send_task_to_queue(self):
        """
        Send Task to Queue
        """

        if self.is_system_in_repair:
            moment_to_leave = self.ttek + self.leave_time_distribution.generate()
        else:
            moment_to_leave = self.ttek + 1e16

        new_tsk = ImpatientTaskWithRepairs(
            self.ttek, moment_to_leave, self.is_system_in_repair)

        if self.buffer is None:  # queue length is not specified, i.e. infinite queue
            new_tsk.start_waiting_time = self.ttek
            self.queue.append(new_tsk)
        else:
            if self.queue.size() < self.buffer:
                new_tsk.start_waiting_time = self.ttek
                self.queue.append(new_tsk)
            else:
                self.dropped += 1
                self.in_sys -= 1

    def positive_arrival(self):
        """
        Actions upon arrival of positive job by the QS.
        """

        self.positive_arrived += 1
        self.p[self.in_sys] += self.positive_arrival_time - self.ttek

        self.ttek = self.positive_arrival_time
        self.positive_arrival_time = self.ttek + self.positive_source.generate()

        self.in_sys += 1

        if self.free_channels == 0:
            self.send_task_to_queue()
        else:
            if self.is_system_in_repair:
                self.send_task_to_queue()
            else:
                self.send_task_to_channel()

    def negative_arrival(self):
        """
        Actions upon arrival of disaster 
        After disaster start repair process
        """

        self.negative_arrived += 1
        self.p[self.in_sys] += self.negative_arrival_time - self.ttek
        self.ttek = self.negative_arrival_time
        self.negative_arrival_time = self.ttek + self.negative_source.generate()

        not_free_servers = [c for c in range(
            self.n) if not self.servers[c].is_free]
        for c in not_free_servers:
            end_ts = self.servers[c].end_service()
            self.broken += 1
            self.total += 1

            self.refresh_v_stat(self.ttek - end_ts.arr_time)
            self.refresh_v_stat_broken(self.ttek - end_ts.arr_time)

            self.choose_mode_and_update_v(end_ts)

        while self.queue:
            ts = self.queue.pop()
            ts.wait_time += self.ttek - ts.start_waiting_time
            self.taked += 1
            self.total += 1
            self.broken += 1
            self.refresh_w_stat(ts.wait_time)

            self.refresh_v_stat(self.ttek - ts.arr_time)
            self.refresh_v_stat_broken(self.ttek - ts.arr_time)

            self.choose_mode_and_update_v(ts)

        self.in_sys = 0
        self.free_channels = self.n
        self.is_system_in_repair = True
        self.repair_num += 1
        self.repair_end_time = self.ttek + self.repair_time_distribution.generate()

    def choose_mode_and_update_v(self, ts: ImpatientTaskWithRepairs):
        """
        Choose the mode of operation and update the virtual statistics.
        """
        if ts.arrive_in_repair_mode and ts.is_end_repair:
            sojourn_time = self.ttek - ts.end_repair_time
        else:
            sojourn_time = self.ttek - ts.arr_time

        if self.is_system_in_repair:
            self.served_in_down += 1
            self.refresh_v_stat_down(sojourn_time)
        else:
            self.served_in_up += 1
            self.refresh_v_stat_up(sojourn_time)

    def send_task_to_channel(self, is_warm_start=False):
        """
        Sends a job to the service channel
        """
        for s in self.servers:
            if s.is_free:
                tsk = ImpatientTaskWithRepairs(
                    self.ttek, self.ttek+1e16, self.is_system_in_repair)
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

        self.refresh_v_stat(self.ttek - end_ts.arr_time)
        self.refresh_v_stat_served(self.ttek - end_ts.arr_time)

        self.choose_mode_and_update_v(end_ts)

        self.in_sys -= 1

        if self.queue.size() != 0:
            self.send_head_of_queue_to_channel(c)

    def repair_end(self):
        """
        Actions upon repair process end
        """

        self.p[self.in_sys] += self.repair_end_time - self.ttek
        self.ttek = self.repair_end_time

        for tsk in self.queue.queue:

            tsk.moment_to_leave = self.ttek + 1e12
            tsk.end_repair_time = self.ttek
            tsk.is_end_repair = True

            self.choose_mode_and_update_v(tsk)

        # send task to channel if there are free channels
        free_servers = [c for c in range(self.n) if self.servers[c].is_free]
        for c in free_servers:
            if self.queue.size() != 0:
                self.send_head_of_queue_to_channel(c)

        self.is_system_in_repair = False
        self.repair_end_time = 1e16

    def drop_task(self, num_of_task_earlier, moment_to_leave_earlier):
        """
        Drop a task from the queue and update statistics.
        Only happens if system is in repair mode.
        """
        self.p[self.in_sys] += moment_to_leave_earlier - self.ttek
        self.ttek = moment_to_leave_earlier

        # drop task from queue and update statistics
        new_queue = []
        end_ts = None
        for i, tsk in enumerate(self.queue.queue):
            if i != num_of_task_earlier:
                new_queue.append(tsk)
            else:
                end_ts = self.queue.queue[i]

        self.queue.queue = new_queue
        self.in_sys -= 1
        self.leave_num += 1

        self.total += 1
        self.taked += 1

        sojourn_time = self.ttek - end_ts.arr_time
        self.refresh_v_stat(sojourn_time)
        end_ts.wait_time = sojourn_time
        self.refresh_w_stat(end_ts.wait_time)

        self.choose_mode_and_update_v(end_ts)

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

        moment_to_leave_earlier = 1e10
        num_of_task_earlier = -1
        if self.is_system_in_repair:
            for i, tsk in enumerate(self.queue.queue):
                if tsk.moment_to_leave < moment_to_leave_earlier:
                    moment_to_leave_earlier = tsk.moment_to_leave
                    num_of_task_earlier = i

        # Global warm-up is set. Need to track
        # including the moment of warm-up end
        times = [serv_earl, self.positive_arrival_time,
                 self.negative_arrival_time, self.repair_end_time, moment_to_leave_earlier]
        min_time_num = np.argmin(times)
        if min_time_num == 0:
            # Serving
            self.serving(num_of_server_earlier)
        elif min_time_num == 1:
            # Arrival positive
            self.positive_arrival()
        elif min_time_num == 2:
            # Arrival negative
            self.negative_arrival()
        elif min_time_num == 3:
            # Repair end time
            self.repair_end()
        else:
            # drop imaptient task
            self.drop_task(num_of_task_earlier, moment_to_leave_earlier)

    def get_v_up(self):
        """
        Returns the mean sojourn time in up state
        """
        return self.v_up

    def get_v_down(self):
        """
        Returns the mean sojourn time in down state
        """
        return self.v_down

    def refresh_v_stat_down(self, new_a):
        """
        Updating statistics of sojourn times in down state 
        """
        self.v_down = refresh_moments_stat(
            self.v_down, new_a, self.served_in_down)

    def refresh_v_stat_up(self, new_a):
        """
        Updating statistics of sojourn times in up state 
        """
        self.v_up = refresh_moments_stat(self.v_up, new_a, self.served_in_up)

    def __str__(self, is_short=False):

        type_of_neg_str = 'Disaster'

        res = "Queueing system " + self.positive_source_types + \
            "/" + self.server_types + "/" + str(self.n) + type_of_neg_str
        if self.buffer is not None:
            res += "/" + str(self.buffer)
        res += f"\nLoad: {self.calc_positive_load():4.3f}\n"
        res += f"Current Time {self.ttek:8.3f}\n"
        res += f"Positive arrival Time: {self.positive_arrival_time:8.3f}\n"
        res += f"Negative arrival Time: {self.negative_arrival_time:8.3f}\n"
        res += f'Repair and time: {self.repair_end_time:8.3f}\n'
        res += f'Repairs num: {self.repair_num}\n'

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
                res += f"Positive dropped (by buffer): {self.dropped}\n"
            res += f"Negative arrived: {self.negative_arrived}\n"
            res += f"Taken: {self.taked}\n"
            res += f"Served: {self.served}\n"
            res += f"In System: {self.in_sys}\n"
            res += "\n"
            for c in range(self.n):
                res += str(self.servers[c])
            res += f"\nQueue Count {self.queue.size()}\n"

        return res
