"""
Simulation model of QS GI/G/n/r and GI/G/n
"""

import time

from colorama import Fore, Style, init
from tqdm import tqdm

from most_queue.constants import DEFAULT_NUM_STATES
from most_queue.random.utils.create import create_distribution
from most_queue.random.utils.load import calc_qs_load
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.sim.utils.events import EventScheduler
from most_queue.sim.utils.qs_queue import QsQueueDeque, QsQueueList
from most_queue.sim.utils.servers import Server
from most_queue.sim.utils.stats_update import refresh_moments_stat
from most_queue.sim.utils.tasks import Task
from most_queue.structs import QueueResults

init()


class QsSim(BaseSimulationCore):
    """
    Base class for Queueing System Simulator
    """

    def __init__(self, num_of_channels, buffer=None, verbose=True, buffer_type="list"):
        """
        Initialize the queueing system with GI/G/n/r or GI/G/n model.
        :param num_of_channels: int : number of channels in the system
        :param buffer: Optional(int, None) : maximum length of the queue, None if infinite
        :param verbose: bool : whether to print detailed information during simulation
        :param buffer_type: str : type of the buffer, "list" or "deque"
        """
        super().__init__()

        self.n = num_of_channels
        self.buffer = buffer
        self.verbose = verbose

        self.free_channels = self.n
        self.num_of_states = DEFAULT_NUM_STATES
        self.load = 0  # utilization factor

        # to track the length of the continuous channel occupancy period:
        self.start_busy = 0

        self.total = 0

        self.w = [0, 0, 0, 0]  # raw moments of waiting time in the QS
        self.v = [0, 0, 0, 0]  # raw moments of sojourn time in the QS

        # probabilities of the QS states:
        self.p = [0.0] * self.num_of_states

        self.taked = 0  # number of job accepted for service
        self.served = 0  # number of job serviced by the system
        self.in_sys = 0  # number of job in the system
        self.arrived = 0  # number of job received
        self.dropped = 0  # number of job denied service

        self.arrival_time = 0  # time of arrival of the next job

        # queue of jobs: class - Task
        if buffer_type == "list":
            self.queue = QsQueueList()
        elif buffer_type == "deque":
            self.queue = QsQueueDeque()
        else:
            raise ValueError("Unknown queue type")

        self.servers = []  # service channels, list of Server's

        self.source = None
        self.source_params = None
        self.source_kendall_notation = None

        self.server_params = None
        self.server_kendall_notation = None

        self.is_set_source_params = False
        self.is_set_server_params = False

        self.zero_wait_arrivals_num = 0

        # Set of free server indices for O(1) access
        self._free_servers = set(range(self.n))

        # Event scheduler for managing simulation events
        self.event_scheduler = EventScheduler()

    def set_sources(self, params, kendall_notation: str = "M"):
        """
        Specifies the type and parameters of source time distribution.
        :param params: dataclass : parameters for the source time distribution
            for example: H2Params for hyper-exponential distribution
            (see most_queue.general.distribution_params)
            For 'M' (exponential) params is a float number, that represent single parameter
        :param kendall_notation: str : types of source time distribution ,
           for example: 'H' for hyper-exponential, 'M' for exponential, 'C' for Coxian
        """
        self.source_params = params
        self.source_kendall_notation = kendall_notation

        self.is_set_source_params = True

        self.source = create_distribution(params, kendall_notation, self.generator)

        self.arrival_time = self.source.generate()

    def set_servers(self, params, kendall_notation: str = "M"):
        """
        Specifies the type and parameters of service time distribution.
        :param params: dataclass : parameters for the service time distribution
            for example: H2Params for hyper-exponential distribution
            (see most_queue.general.distribution_params)
            For 'M' (exponential) params is a float number, that represent single parameter
        :param kendall_notation: str : types of source time distribution ,
           for example: 'H' for hyper-exponential, 'M' for exponential, 'C' for Coxian
        """
        self.server_params = params
        self.server_kendall_notation = kendall_notation

        self.is_set_server_params = True

        self.servers = [
            Server(
                self.server_params,
                self.server_kendall_notation,
                generator=self.generator,
            )
            for _i in range(self.n)
        ]
        self._mark_servers_time_changed()
        self._free_servers = set(range(self.n))

    def calc_load(self) -> float:
        """
        Calculates the load factor (utilization) of the queueing system.

        Returns:
            float: The utilization factor (load) of the system, between 0 and 1.
        """
        return calc_qs_load(
            self.source_kendall_notation,
            self.source_params,
            self.server_kendall_notation,
            self.server_params,
            self.n,
        )

    def send_task_to_channel(self, is_warm_start: bool = False, tsk=None) -> None:
        """
        Sends a job to an available service channel.

        Args:
            is_warm_start: Whether this is a warm start scenario.
            tsk: Task to send. If None, a new task is created.
        """

        if tsk is None:
            tsk = Task(self.ttek)
            tsk.wait_time = 0

        # Use free servers set for O(1) access
        if self._free_servers:
            server_idx = next(iter(self._free_servers))
            self.taked += 1
            self.refresh_w_stat(tsk.wait_time)
            self.zero_wait_arrivals_num += 1

            self.servers[server_idx].start_service(tsk, self.ttek, is_warm_start)
            self._free_servers.remove(server_idx)
            self.free_channels -= 1
            self._mark_servers_time_changed()

            # Проверям, не наступил ли ПНЗ:
            if self.free_channels == 0:
                if self.in_sys == self.n:
                    self.start_busy = self.ttek

    def send_task_to_queue(self, new_tsk=None) -> None:
        """
        Send a task to the queue.

        Args:
            new_tsk: Task to add to queue. If None, a new task is created.
                    If buffer is full, the task will be dropped.
        """

        if new_tsk is None:
            new_tsk = Task(self.ttek)
            new_tsk.start_waiting_time = self.ttek
        else:
            # If task already exists but start_waiting_time is not set, set it now
            if new_tsk.start_waiting_time < 0:
                new_tsk.start_waiting_time = self.ttek

        if self.buffer is None:  # queue length is not specified, i.e. infinite queue

            self.queue.append(new_tsk)
        else:
            if self.queue.size() < self.buffer:
                self.queue.append(new_tsk)
            else:
                self.dropped += 1
                self.in_sys -= 1

    def arrival(self, moment: float | None = None, ts=None) -> None:
        """
        Handle job arrival event in the queueing system.

        Args:
            moment: Optional specific arrival moment. If None, uses current arrival_time.
            ts: Optional task object. If None, a new task is created.
        """

        self.arrived += 1
        # Update state probabilities
        self._update_state_probs(self.ttek, self.arrival_time, self.in_sys)

        if moment:
            self.ttek = moment
            ts.arr_time = moment
            ts.wait_time = 0
            ts.start_waiting_time = -1
            ts.time_to_end_service = 0

        else:
            self.ttek = self.arrival_time
            self.arrival_time = self.ttek + self.source.generate()

        self.in_sys += 1

        if self.free_channels == 0:
            self.send_task_to_queue(new_tsk=ts)

        else:  # there are free channels:
            self.send_task_to_channel(tsk=ts)

    def serving(self, c: int, is_network: bool = False):
        """
        Handle service completion event for a channel.

        Args:
            c: Channel number that completed service.
            is_network: Whether this is part of a network simulation.

        Returns:
            Task: The task that completed service.
        """
        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()
        self._mark_servers_time_changed()
        self._free_servers.add(c)  # Server is now free

        # Update state probabilities
        self._update_state_probs(self.ttek, time_to_end, self.in_sys)

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

        if self.queue.size() != 0:
            self.send_head_of_queue_to_channel(c, is_network=is_network)

        return end_ts

    def send_head_of_queue_to_channel(self, channel_num: int, is_network: bool = False) -> None:
        """
        Send the first task from the queue (head) to an available channel.

        Args:
            channel_num: Channel number to send the task to.
            is_network: Whether this is part of a network simulation.
        """
        que_ts = self.queue.pop()

        if self.free_channels == 1:
            self.start_busy = self.ttek

        self.taked += 1
        que_ts.wait_time += self.ttek - que_ts.start_waiting_time
        if is_network:
            que_ts.wait_network += self.ttek - que_ts.start_waiting_time
        self.refresh_w_stat(que_ts.wait_time)

        self.servers[channel_num].start_service(que_ts, self.ttek)
        self._free_servers.remove(channel_num)  # Server is now busy
        self.free_channels -= 1
        self._servers_time_changed = True

    def _get_min_server_time(self):
        """
        Get server with minimum time to end service.
        Uses caching to avoid O(n) search on every call.
        """
        if self._servers_time_changed:
            self._min_server_time = 1e16
            self._min_server_idx = -1
            for c in range(self.n):
                if self.servers[c].time_to_end_service < self._min_server_time:
                    self._min_server_time = self.servers[c].time_to_end_service
                    self._min_server_idx = c
            self._servers_time_changed = False
        return self._min_server_idx, self._min_server_time

    def _update_state_probs(self, old_time: float, new_time: float, state: int):
        """
        Update state probabilities array.

        Args:
            old_time: Previous time
            new_time: New time
            state: Current system state
        """
        # Ensure p array is large enough
        while state >= len(self.p):
            # Extend p array in chunks to avoid frequent reallocations
            self.p.extend([0.0] * min(1000, max(1, len(self.p) // 10)))
        self.p[state] += new_time - old_time

    # Hook methods for customization
    def _before_arrival(self, moment: float = None, ts=None):
        """
        Hook called before arrival event is processed.
        Override this to add custom logic before arrival.

        Args:
            moment: Optional specific arrival moment
            ts: Optional task object
        """
        # Override in subclasses if needed

    def _after_arrival(self, moment: float = None, ts=None):
        """
        Hook called after arrival event is processed.
        Override this to add custom logic after arrival.

        Args:
            moment: Optional specific arrival moment
            ts: Optional task object
        """
        # Override in subclasses if needed

    def _before_serving(self, channel: int, is_network: bool = False):
        """
        Hook called before serving event is processed.
        Override this to add custom logic before serving.

        Args:
            channel: Channel number that completed service
            is_network: Whether this is part of a network simulation
        """
        # Override in subclasses if needed

    def _after_serving(self, channel: int, task=None, is_network: bool = False):
        """
        Hook called after serving event is processed.
        Override this to add custom logic after serving.

        Args:
            channel: Channel number that completed service
            task: Task that completed service
            is_network: Whether this is part of a network simulation
        """
        # Override in subclasses if needed

    def _get_custom_events(self):
        """
        Get custom events for this simulator.
        Override this method to register custom event types.

        Returns:
            dict: Dictionary mapping event_type -> event_time
        """
        return {}

    def _handle_custom_event(self, event_type: str):
        """
        Handle a custom event.
        Override this method to process custom events.

        Args:
            event_type: Type of the custom event
        """
        raise NotImplementedError(f"Custom event '{event_type}' not handled")

    def _get_available_events(self):
        """
        Collect all available events (arrival, serving, custom).
        Returns dictionary of event_type -> event_time.
        """
        events = {}

        # Standard arrival event
        if hasattr(self, "arrival_time") and self.arrival_time < float("inf"):
            events["arrival"] = self.arrival_time

        # Standard serving event (next service completion)
        server_idx, serv_time = self._get_min_server_time()
        if server_idx >= 0 and serv_time < float("inf"):
            events["serving"] = serv_time

        # Custom events from subclass
        custom_events = self._get_custom_events()
        events.update(custom_events)

        return events

    def _select_next_event(self):
        """
        Select the next event to process based on minimum time.

        Returns:
            tuple: (event_type, event_time) or (None, None) if no events
        """
        events = self._get_available_events()
        if not events:
            return None, None

        event_type = min(events.keys(), key=lambda k: events[k])
        return event_type, events[event_type]

    def _execute_event(self, event_type: str):
        """
        Execute the specified event.

        Args:
            event_type: Type of event to execute
        """
        if event_type == "arrival":
            self._before_arrival()
            self.arrival()
            self._after_arrival()
        elif event_type == "serving":
            server_idx, _ = self._get_min_server_time()
            self._before_serving(server_idx)
            task = self.serving(server_idx)
            self._after_serving(server_idx, task)
        else:
            # Custom event
            self._handle_custom_event(event_type)

    def run_one_step(self) -> None:
        """
        Execute one step of the simulation using event-based approach.
        Automatically handles arrival, serving, and custom events.
        """
        event_type, _event_time = self._select_next_event()

        if event_type is None:
            raise RuntimeError("No events available to process")

        self._execute_event(event_type)

    def run(self, total_served: int) -> QueueResults:
        """
        Run the complete simulation process.

        Args:
            total_served: Number of jobs to serve before stopping the simulation.

        Returns:
            QueueResults: Results object containing statistics from the simulation.
        """
        start = time.process_time()

        print(Fore.GREEN + "\rStart simulation")

        last_percent = 0

        with tqdm(total=100) as pbar:
            while self.served < total_served:
                self.run_one_step()
                percent = int(100 * (self.served / total_served))
                if last_percent != percent:
                    last_percent = percent
                    pbar.update(1)
                    pbar.set_description(
                        Fore.MAGENTA
                        + "\rJob served: "
                        + Fore.YELLOW
                        + f"{self.served}/{total_served}"
                        + Fore.LIGHTGREEN_EX
                    )

        print(Fore.GREEN + "\rSimulation is finished")
        print(Style.RESET_ALL)

        v = self.get_v()
        w = self.get_w()
        p = self.get_p()

        self.time_spent = time.process_time() - start

        return QueueResults(v=v, w=w, p=p, duration=self.time_spent, utilization=self.calc_load())

    def refresh_busy_stat(self, new_a: float, count: int = None) -> None:
        """
        Update statistics of the busy period.

        Args:
            new_a: New busy period duration to include in statistics.
            count: Number of busy periods (if None, uses self.busy_moments)
        """
        if count is None:
            count = self.busy_moments
        super().refresh_busy_stat(new_a, count)

    def refresh_v_stat(self, new_a: float, count: int = None) -> None:
        """
        Update statistics of sojourn times.

        Args:
            new_a: New sojourn time value to include in statistics.
            count: Number of served tasks (if None, uses self.served)
        """
        if count is None:
            count = self.served
        self.v = refresh_moments_stat(self.v, new_a, count)

    def refresh_w_stat(self, new_a: float, count: int = None) -> None:
        """
        Update statistics of wait times.

        Args:
            new_a: New waiting time value to include in statistics.
            count: Number of taken tasks (if None, uses self.taked)
        """
        if count is None:
            count = self.taked
        self.w = refresh_moments_stat(self.w, new_a, count)

    def get_p(self) -> list[float]:
        """
        Returns a list with probabilities of queueing system states.

        Returns:
            list[float]: List where p[j] is the probability that there will be exactly j jobs
                        in the system at a random moment in time.
        """
        res = [0.0] * len(self.p)
        for j in range(0, self.num_of_states):
            res[j] = self.p[j] / self.ttek
        return res

    def get_v(self) -> list[float]:
        """
        Returns a list with moments of sojourn times
        v[j] - the j-th moment of sojourn time
        """
        return self.v

    def get_w(self) -> list[float]:
        """
        Returns a list with moments of wait times
        w[j] - the j-th moment of wait time
        """
        return self.w

    def __str__(self, is_short=False):
        buffer_str = f"/{self.buffer}" if self.buffer is not None else ""
        res = f"Queueing system {self.source_kendall_notation}/{self.server_kendall_notation}/{self.n}{buffer_str}\n"
        res += f"Load: {self.calc_load():4.3f}\n"
        res += f"Current Time {self.ttek:8.3f}\n"
        res += f"Arrival Time: {self.arrival_time:8.3f}\n"

        sojourn_moments = "\t".join(f"{self.v[i]:8.4f}" for i in range(3))
        res += f"Sojourn moments:\n\t{sojourn_moments}\n"

        wait_moments = "\t".join(f"{self.w[i]:8.4f}" for i in range(3))
        res += f"Wait moments:\n\t{wait_moments}\n"

        if not is_short:
            stationary_probs = "\t".join(f"{self.p[i] / self.ttek:6.5f}" for i in range(10))
            res += f"Stationary prob:\n\t{stationary_probs}\n"

            res += f"Arrived: {self.arrived}\n"
            if self.buffer is not None:
                res += f"Dropped: {self.dropped}\n"
            res += f"Taken: {self.taked}\n"
            res += f"Served: {self.served}\n"
            res += f"In System: {self.in_sys}\n"

            busy_moments = "\t".join(f"{self.busy[j]:8.4f}" for j in range(3))
            res += f"Busy moments:\n\t{busy_moments}\n"

            server_strs = "".join(str(self.servers[c]) for c in range(self.n))
            res += f"{server_strs}\nQueue Count {self.queue.size()}\n"

        return res
