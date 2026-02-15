"""
Simulation model of QS GI/G/n/r and GI/G/n with negative jobs
"""

import random
from enum import Enum

from most_queue.random.utils.create import create_distribution
from most_queue.random.utils.load import calc_qs_load
from most_queue.sim.base import QsSim
from most_queue.sim.utils.stats_update import refresh_moments_stat
from most_queue.structs import NegativeArrivalsResults


class NegativeServiceType(Enum):
    """
    Negative service types
    """

    DISASTER = 1  # remove all customers
    RCS = 2  # remove customer in service
    RCH = 3  # remove customer at the Head
    RCE = 4  # remove customer at the End


class DisasterScenario(Enum):
    """
    Scenario for DISASTER negative arrivals.

    - CLEAR_SYSTEM: negative arrival removes all positive jobs (current behavior)
    - REQUEUE_ALL: negative arrival interrupts service and requeues all jobs
      that were in service back to the queue (service restarts).
    """

    CLEAR_SYSTEM = 1
    REQUEUE_ALL = 2


class RcsScenario(Enum):
    """
    Scenario for RCS (Remove Customer in Service) negative arrivals.

    - REMOVE: negative arrival removes one random customer in service (current behavior)
    - REQUEUE: negative arrival interrupts one random customer in service and requeues it
      to the head of the queue (service restarts; no customer leaves the system).
    """

    REMOVE = 1
    REQUEUE = 2


class QsSimNegatives(QsSim):
    """
    Simulation model of QS GI/G/n/r and GI/G/n with negative jobs
    """

    def __init__(
        self,
        num_of_channels: int,
        type_of_negatives: NegativeServiceType = NegativeServiceType.DISASTER,
        buffer: int | None = None,
        verbose: bool = True,
        buffer_type: str = "list",
        disaster_scenario: DisasterScenario = DisasterScenario.CLEAR_SYSTEM,
        rcs_scenario: RcsScenario = RcsScenario.REMOVE,
    ):  # pylint: disable=too-many-positional-arguments, too-many-arguments
        """
        Initialize the queueing system with GI/G/n/r or GI/G/n model.
        :param num_of_channels: int : number of channels in the system
        :param buffer: Optional(int, None) : maximum length of the queue, None if infinite
        :param verbose: bool : whether to print detailed information during simulation
        :param buffer_type: str : type of the buffer, "list" or "deque"
        :param disaster_scenario: DisasterScenario : behavior for DISASTER negatives:
            - CLEAR_SYSTEM (default): remove all jobs from system
            - REQUEUE_ALL: interrupt service and requeue jobs in service to the head
        :param rcs_scenario: RcsScenario : behavior for RCS negatives:
            - REMOVE (default): remove one random job from service (broken)
            - REQUEUE: interrupt one random job in service and requeue it to the head
        """

        super().__init__(num_of_channels, buffer, verbose, buffer_type)

        self.type_of_negatives = type_of_negatives
        self.disaster_scenario = disaster_scenario
        self.rcs_scenario = rcs_scenario

        # raw moments of sojourn time of successfully served
        self.v_served = [0, 0, 0, 0]
        # raw moments of sojourn time of broken by negative jobs
        self.v_broken = [0, 0, 0, 0]

        self.served = 0  # number of job serviced by the system without negative job breaks
        self.broken = 0  # number of job broken by negatives
        self.total = 0  # number of job broken by negatives and served without breaks

        # Positive arrivals
        self.positive_arrived = 0  # number of job received
        self.positive_arrival_time = 0  # time of arrival of the next job
        self.positive_source = None
        self.positive_source_params = None
        self.positive_source_kendall_notation = None
        self.is_set_positive_source_params = False

        # Negative arrivals
        self.negative_arrived = 0  # number of job received
        self.negative_arrival_time = 0  # time of arrival of the next job
        self.negative_source = None
        self.negative_source_params = None
        self.negative_source_kendall_notation = None
        self.is_set_negative_source_params = False

    def set_positive_sources(self, params, kendall_notation: str = "M"):
        """
        Specifies the type and parameters of positives source time distribution.
        :param params: dataclass : parameters for the positives source time distribution
            for example: H2Params for hyper-exponential distribution
            (see most_queue.general.distribution_params)
            For 'M' (exponential) params is a float number, that represent single parameter
        :param kendall_notation: str : types of positives source time distribution ,
           for example: 'H' for hyper-exponential, 'M' for exponential, 'C' for Coxian
        """
        self.positive_source_params = params
        self.positive_source_kendall_notation = kendall_notation

        self.is_set_positive_source_params = True

        self.positive_source = create_distribution(params, kendall_notation, self.generator)

        self.positive_arrival_time = self.positive_source.generate()

    def set_negative_sources(self, params, kendall_notation: str = "M"):
        """
        Specifies the type and parameters of negative source time distribution.
        :param params: dataclass : parameters for the negative source time distribution
            for example: H2Params for hyper-exponential distribution
            (see most_queue.general.distribution_params)
            For 'M' (exponential) params is a float number, that represent single parameter
        :param kendall_notation: str : types of negative source time distribution ,
           for example: 'H' for hyper-exponential, 'M' for exponential, 'C' for Coxian
        """
        self.negative_source_params = params
        self.negative_source_kendall_notation = kendall_notation

        self.is_set_negative_source_params = True

        self.negative_source = create_distribution(params, kendall_notation, self.generator)

        self.negative_arrival_time = self.negative_source.generate()

    def _handle_custom_event(self, event_type: str) -> None:
        """No additional custom events in negative jobs simulator."""
        pass  # pylint: disable=unnecessary-pass

    def calc_positive_load(self):
        """
        Calculates the load factor of the QS if has no disatsers
        """

        return calc_qs_load(
            self.positive_source_kendall_notation,
            self.positive_source_params,
            self.server_kendall_notation,
            self.server_params,
            self.n,
        )

    def run(self, total_served) -> NegativeArrivalsResults:
        """
        Run simulation process
        """
        results = super().run(total_served=total_served)
        v_served = self.v_served
        v_broken = self.v_broken

        return NegativeArrivalsResults(
            v=results.v,
            w=results.w,
            p=results.p,
            v_served=v_served,
            v_broken=v_broken,
            duration=results.duration,
            utilization=results.utilization,
        )

    def positive_arrival(self):
        """
        Actions upon arrival of positive job by the QS.
        """

        self.positive_arrived += 1
        # Update state probabilities
        self._update_state_probs(self.ttek, self.positive_arrival_time, self.in_sys)

        self.in_sys += 1
        self.ttek = self.positive_arrival_time
        self.positive_arrival_time = self.ttek + self.positive_source.generate()

        if self.free_channels == 0:
            self.send_task_to_queue()
        else:
            self.send_task_to_channel()

    def negative_arrival(self):
        """
        Actions upon arrival of negative job by the QS.
        """

        self.negative_arrived += 1
        # Update state probabilities
        self._update_state_probs(self.ttek, self.negative_arrival_time, self.in_sys)
        self.ttek = self.negative_arrival_time
        self.negative_arrival_time = self.ttek + self.negative_source.generate()

        if self.in_sys == 0:
            # If no jobs in system, negatives has no effect
            return

        if self.type_of_negatives == NegativeServiceType.DISASTER:

            if self.disaster_scenario == DisasterScenario.REQUEUE_ALL:
                # Interrupt all services and requeue the tasks back to the head of the queue.
                # Semantics: service restarts (preemptive-repeat with resampling).
                tasks_to_requeue = []
                not_free_servers = [c for c in range(self.n) if not self.servers[c].is_free]
                for c in not_free_servers:
                    ts = self.servers[c].end_service()
                    self._free_servers.add(c)
                    ts.start_waiting_time = self.ttek
                    tasks_to_requeue.append(ts)

                # Put interrupted tasks in FCFS order (by original arrival moment) to the head of the queue.
                tasks_to_requeue.sort(key=lambda t: t.arr_time)
                for ts in reversed(tasks_to_requeue):
                    self.queue.append_left(ts)

                # All servers become free at this instant, then immediately take tasks from the queue.
                self.free_channels = self.n
                self._free_servers = set(range(self.n))
                self._mark_servers_time_changed()

                while self.free_channels > 0 and self.queue.size() > 0:
                    c = next(iter(self._free_servers))
                    self.send_head_of_queue_to_channel(c)

                # Note: in_sys does not change; no job leaves the system.
            else:
                # CLEAR_SYSTEM (current behavior): remove all jobs from service and from queue.
                not_free_servers = [c for c in range(self.n) if not self.servers[c].is_free]
                for c in not_free_servers:
                    end_ts = self.servers[c].end_service()
                    self._free_servers.add(c)  # Server is now free
                    self.broken += 1
                    self.total += 1
                    sojourn_time = self.ttek - end_ts.arr_time
                    self.refresh_v_stat(sojourn_time)
                    self.refresh_v_stat_broken(sojourn_time)

                self.in_sys = 0
                self.free_channels = self.n
                self._free_servers = set(range(self.n))  # All servers are free
                self._mark_servers_time_changed()

                while self.queue.size() > 0:
                    ts = self.queue.pop()
                    ts.wait_time += self.ttek - ts.start_waiting_time
                    self.taked += 1
                    self.total += 1
                    self.broken += 1
                    self.refresh_w_stat(ts.wait_time)

                    sojourn_time = self.ttek - ts.arr_time
                    self.refresh_v_stat(sojourn_time)
                    self.refresh_v_stat_broken(sojourn_time)

        elif self.type_of_negatives == NegativeServiceType.RCE:
            self.in_sys -= 1
            ts = self.queue.tail()

            ts.wait_time += self.ttek - ts.start_waiting_time

            self.taked += 1
            self.total += 1
            self.broken += 1

            self.refresh_w_stat(ts.wait_time)
            sojourn_time = self.ttek - ts.arr_time
            self.refresh_v_stat(sojourn_time)
            self.refresh_v_stat_broken(sojourn_time)

        elif self.type_of_negatives == NegativeServiceType.RCH:
            self.in_sys -= 1
            ts = self.queue.pop()
            ts.wait_time += self.ttek - ts.start_waiting_time

            self.taked += 1
            self.total += 1
            self.broken += 1

            self.refresh_w_stat(ts.wait_time)
            sojourn_time = self.ttek - ts.arr_time
            self.refresh_v_stat(sojourn_time)
            self.refresh_v_stat_broken(sojourn_time)

        elif self.type_of_negatives == NegativeServiceType.RCS:

            not_free_servers = [c for c in range(self.n) if not self.servers[c].is_free]
            if not not_free_servers:
                # No one is in service: RCS has no effect in this instant.
                return
            c = random.choice(not_free_servers)
            end_ts = self.servers[c].end_service()
            self._free_servers.add(c)  # Server is now free
            self._mark_servers_time_changed()
            self.free_channels += 1

            if self.rcs_scenario == RcsScenario.REQUEUE:
                # Interrupt service and requeue the task to the head.
                # Semantics: service restarts (preemptive-repeat with resampling).
                end_ts.start_waiting_time = self.ttek
                self.queue.append_left(end_ts)

                # No one leaves the system; restart service immediately if possible.
                if self.queue.size() != 0:
                    self.send_head_of_queue_to_channel(c)
            else:
                # REMOVE (current behavior): remove the job from system as broken.
                self.total += 1
                self.broken += 1
                sojourn_time = self.ttek - end_ts.arr_time
                self.refresh_v_stat(sojourn_time)
                self.refresh_v_stat_broken(sojourn_time)

                self.in_sys -= 1

                if self.queue.size() != 0:
                    self.send_head_of_queue_to_channel(c)

    def _get_available_events(self):
        """
        Collect all available events (positive arrival, negative arrival, serving, custom).
        Override to include negative arrivals.
        """
        events = {}

        # Positive arrival event
        if hasattr(self, "positive_arrival_time") and self.positive_arrival_time < float("inf"):
            events["positive_arrival"] = self.positive_arrival_time

        # Negative arrival event
        if hasattr(self, "negative_arrival_time") and self.negative_arrival_time < float("inf"):
            events["negative_arrival"] = self.negative_arrival_time

        # Standard serving event (next service completion)
        server_idx, serv_time = self._get_min_server_time()
        if server_idx >= 0 and serv_time < float("inf"):
            events["serving"] = serv_time

        # Custom events from subclass
        custom_events = self._get_custom_events()
        events.update(custom_events)

        return events

    def _execute_event(self, event_type: str):
        """
        Execute the specified event.
        Override to handle positive/negative arrivals.
        """
        if event_type == "positive_arrival":
            self._before_arrival()
            self.positive_arrival()
            self._after_arrival()
        elif event_type == "negative_arrival":
            self.negative_arrival()
        elif event_type == "serving":
            server_idx, _ = self._get_min_server_time()
            self._before_serving(server_idx)
            task = self.serving(server_idx)
            self._after_serving(server_idx, task)
        else:
            # Custom event
            self._handle_custom_event(event_type)

    def serving(self, c, is_network=False):
        """
        Actions upon receipt of a service job with - channel number
        """
        time_to_end = self.servers[c].time_to_end_service
        end_ts = self.servers[c].end_service()
        self._free_servers.add(c)  # Server is now free
        self._mark_servers_time_changed()

        # Update state probabilities
        self._update_state_probs(self.ttek, time_to_end, self.in_sys)

        self.ttek = time_to_end
        self.free_channels += 1

        self.served += 1
        self.total += 1
        sojourn_time = self.ttek - end_ts.arr_time
        self.refresh_v_stat(sojourn_time)
        self.refresh_v_stat_served(sojourn_time)

        self.in_sys -= 1

        if self.queue.size() != 0:
            self.send_head_of_queue_to_channel(c, is_network=is_network)

        return end_ts

    # run_one_step is now inherited from base class and uses event-based approach
    # Positive and negative arrivals are handled via overridden _get_available_events() and _execute_event()

    def refresh_v_stat(self, new_a):
        """
        Updating statistics of sojourn times (all, broken and successfully served)
        """
        self.v = refresh_moments_stat(self.v, new_a, self.total)

    def refresh_v_stat_broken(self, new_a):
        """
        Updating statistics of sojourn times of broken jobs
        """
        self.v_broken = refresh_moments_stat(self.v_broken, new_a, self.broken)

    def refresh_v_stat_served(self, new_a):
        """
        Updating statistics of sojourn times of successfully served jobs
        """
        self.v_served = refresh_moments_stat(self.v_served, new_a, self.served)

    def get_v_served(self):
        """
        Returns raw moments of sojourn time (only for successfully served jobs)
        """
        return self.v_served

    def get_v_broken(self):
        """
        Returns raw moments of sojourn time  (only for broken by negative arrivals)
        """
        return self.v_broken

    def __str__(self, is_short=False):

        type_of_neg_str = ""
        if self.type_of_negatives == NegativeServiceType.DISASTER:
            type_of_neg_str = "Disaster"
        elif self.type_of_negatives == NegativeServiceType.RCE:
            type_of_neg_str = "Remove customer from the End of the Queue"
        elif self.type_of_negatives == NegativeServiceType.RCH:
            type_of_neg_str = "Remove customer from the Head of the Queue"
        elif self.type_of_negatives == NegativeServiceType.RCS:
            type_of_neg_str = "Remove customer from the Service"
        else:
            type_of_neg_str = "Unknown"

        res = (
            "Queueing system "
            + self.positive_source_kendall_notation
            + "/"
            + self.server_kendall_notation
            + "/"
            + str(self.n)
            + type_of_neg_str
        )
        if self.buffer is not None:
            res += "/" + str(self.buffer)
        res += f"\nLoad: {self.calc_positive_load():4.3f}\n"
        res += f"Current Time {self.ttek:8.3f}\n"
        res += f"Positive arrival Time: {self.positive_arrival_time:8.3f}\n"
        res += f"Negative arrival Time: {self.negative_arrival_time:8.3f}\n"

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
                res += f"Positive dropped: {self.dropped}\n"
            res += f"Negative arrived: {self.negative_arrived}\n"
            res += f"Taken: {self.taked}\n"
            res += f"Served: {self.served}\n"
            res += f"In System: {self.in_sys}\n"
            res += "\n"
            for c in range(self.n):
                res += str(self.servers[c])
            res += f"\nQueue Count {self.queue.size()}\n"

        return res
