"""
Simulation model of QS GI/G/n/r and GI/G/n
"""

import numpy as np

from most_queue.sim.base import QsSim
from most_queue.sim.utils.distribution_utils import create_distribution
from most_queue.sim.utils.exceptions import QsSourseSettingException
from most_queue.sim.utils.phase import QsPhase
from most_queue.sim.utils.servers import Server


class VacationQueueingSystemSimulator(QsSim):
    """
    Simulation model of QS GI/G/n/r and GI/G/n
    """

    def __init__(self, num_of_channels,
                 buffer=None,
                 verbose=True,
                 buffer_type="list", is_service_on_warm_up=False):
        """
        Initialize the queueing system with GI/G/n/r or GI/G/n model.
        :param num_of_channels: int : number of channels in the system
        :param buffer: Optional(int, None) : maximum length of the queue, None if infinite
        :param verbose: bool : whether to print detailed information during simulation
        :param buffer_type: str : type of the buffer, "list" or "deque"
        :param is_service_on_warm_up: bool : 
            if is True, jobs arrived in empty system will be served with warm_phase distribution
            if is False, jobs arrived in empty system will be send to queue, 
                and warm_phase starts with warm_phase distribution
        """
        super().__init__(num_of_channels, buffer, verbose, buffer_type)

        self.warm_phase = QsPhase("WarmUp")
        self.cold_phase = QsPhase("Cold")
        self.cold_delay_phase = QsPhase("ColdDelay")

        self.warm_after_cold_starts = 0
        self.is_service_on_warm_up = is_service_on_warm_up

    def set_warm(self, params, kendall_notation):
        """
        Set the type and parameters of the warm-up time distribution

        :param params: list : parameters for the warm-up time distribution
        :param kendall_notation: str : Kendall notation for the warm-up time distribution

        """
        if not self.is_set_server_params:
            raise ValueError(
                "Server parameters are not set. Please call set_servers() first.")

        dist = create_distribution(params, kendall_notation, self.generator)
        self.warm_phase.set_dist(dist)

        if self.is_service_on_warm_up:
            self.servers = []
            for _i in range(self.n):
                server = Server(self.server_params, self.server_kendall_notation,
                                generator=self.generator)
                server.set_warm(params, kendall_notation, self.generator)
                self.servers.append(server)

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

    def arrival(self):
        """
        Actions upon arrival of the job by the QS.
        """

        self.arrived += 1
        self.p[self.in_sys] += self.arrival_time - self.ttek

        self.in_sys += 1
        self.ttek = self.arrival_time
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
                        # to queue
                        if self.is_service_on_warm_up:
                            self.send_task_to_channel(is_warm_start=True)
                        else:
                            # 2. It is empty and was turned off after cooling. We start warm-up
                            self.warm_phase.start(self.ttek)
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

    def on_end_warming(self):
        """
        Job that has to be done after WarmUp Period Ends
        """

        self.p[self.in_sys] += self.warm_phase.end_time - self.ttek

        self.ttek = self.warm_phase.end_time

        self.warm_phase.end(self.ttek)

        # Send n jobs from the queue to the channels
        for i in range(self.n):
            if self.queue.size() != 0:
                self.send_head_of_queue_to_channel(i)

    def on_end_cold(self):
        """
        Job that has to be done after Cold Period Ends
        """
        self.p[self.in_sys] += self.cold_phase.end_time - self.ttek

        self.ttek = self.cold_phase.end_time

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
        self.p[self.in_sys] += self.cold_delay_phase.end_time - self.ttek

        self.ttek = self.cold_delay_phase.end_time

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
