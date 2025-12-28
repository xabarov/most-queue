"""
Simulation model for queueing systems with impatient tasks
"""

from colorama import Fore, Style, init

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base import QsSim
from most_queue.sim.utils.tasks import ImpatientTask

init()


class ImpatientQueueSim(QsSim):
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
        self.impatience_kendall_notation = None

        self.impatience = None
        self.is_set_impatience_params = False

    def set_impatience(self, params, kendall_notation: str = "M"):
        """
        Set the impatience distribution for tasks.
        :param params: dataclass : parameters for the impatience time distribution
            for example: H2Params for hyper-exponential distribution
            (see most_queue.general.distribution_params)
             For 'M' (exponential) params is a float number, that represent single parameter
        :param kendall_notation: str : types of source time distribution ,
           for example: 'H' for hyper-exponential, 'M' for exponential, 'C' for Coxian

        """
        self.impatience_params = params
        self.impatience_kendall_notation = kendall_notation

        self.is_set_impatience_params = True

        self.impatience = create_distribution(params, kendall_notation, self.generator)

    def _get_custom_events(self):
        """
        Get custom events (impatient task drops).
        """
        events = {}
        if self.queue.size() > 0:
            min_leave_time = float("inf")
            for tsk in self.queue.queue:
                if tsk.moment_to_leave < min_leave_time:
                    min_leave_time = tsk.moment_to_leave
            if min_leave_time < float("inf"):
                events["task_drop"] = min_leave_time
        return events

    def _handle_custom_event(self, event_type: str):
        """
        Handle custom events (impatient task drops).
        """
        if event_type == "task_drop":
            # Find the task with minimum leave time
            num_of_task_earlier = -1
            moment_to_leave_earlier = float("inf")
            for i, tsk in enumerate(self.queue.queue):
                if tsk.moment_to_leave < moment_to_leave_earlier:
                    moment_to_leave_earlier = tsk.moment_to_leave
                    num_of_task_earlier = i

            if num_of_task_earlier >= 0:
                self.drop_task(num_of_task_earlier, moment_to_leave_earlier)
        else:
            super()._handle_custom_event(event_type)

    def arrival(self, moment=None, ts=None):
        """
        Actions on arrival of a task.
        """

        self.arrived += 1
        # Update state probabilities
        self._update_state_probs(self.ttek, self.arrival_time, self.in_sys)

        self.in_sys += 1
        self.ttek = self.arrival_time
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
            # Use free servers set for O(1) access (optimized in base class)
            if self._free_servers:
                server_idx = next(iter(self._free_servers))
                self.taked += 1
                self.servers[server_idx].start_service(ImpatientTask(self.ttek, moment_to_leave), self.ttek, False)
                self._free_servers.remove(server_idx)
                self.free_channels -= 1
                self._mark_servers_time_changed()

                # Проверям, не наступил ли ПНЗ:
                if self.free_channels == 0:
                    if self.in_sys == self.n:
                        self.start_busy = self.ttek

    def drop_task(self, num_of_task_earlier, moment_to_leave_earlier):
        """
        Drop a task from the queue and update statistics.
        """
        # Update state probabilities
        self._update_state_probs(self.ttek, moment_to_leave_earlier, self.in_sys)
        self.ttek = moment_to_leave_earlier

        # Remove task from queue - need to handle deque properly
        end_ts = None
        queue_list = list(self.queue.queue)
        for i, tsk in enumerate(queue_list):
            if i == num_of_task_earlier:
                end_ts = tsk
                break

        # Rebuild queue without the dropped task
        self.queue.queue.clear()
        for i, tsk in enumerate(queue_list):
            if i != num_of_task_earlier:
                self.queue.queue.append(tsk)
        self.in_sys -= 1
        self.dropped += 1
        self.served += 1

        if end_ts is not None:
            self.refresh_v_stat(self.ttek - end_ts.arr_time)
            end_ts.wait_time = self.ttek - end_ts.arr_time
            self.refresh_w_stat(end_ts.wait_time)

    # run_one_step is now inherited from base class and uses event-based approach
    # Custom events (task drops) are handled via _get_custom_events() and _handle_custom_event()

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

        result = f"{Fore.GREEN}Task {num_of_task_earlier}{Style.RESET_ALL}\n"
        result += f"leave at {Fore.YELLOW}{moment_to_leave_earlier:8.3f}{Style.RESET_ALL}"

        return result
