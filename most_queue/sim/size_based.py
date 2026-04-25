"""
Size-based single-server queue simulation (SJF, SRPT, SPJF, ...).

``SizeBasedQsSim`` samples job sizes at arrival and schedules by discipline.
Currently only ``num_of_channels == 1`` is supported.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base import QsSim
from most_queue.sim.utils.queue_priority_size import PrioritySizeQueue
from most_queue.sim.utils.tasks import Task

SizeDiscipline = Literal["FCFS", "SJF", "PSJF", "SRPT", "SPJF", "PSPJF", "SPRPT"]

__all__ = ["SizeDiscipline", "SizePredictor", "PerfectSimPredictor", "SizeBasedQsSim"]


@runtime_checkable
class SizePredictor(Protocol):
    """Predictor for SPJF / PSPJF / SPRPT: ``predict(true_size, rng) -> y``."""

    def predict(self, size: float, rng) -> float:  # noqa: ANN001
        """Return predicted job size ``y`` given true size ``size`` and simulator RNG."""
        ...  # pylint: disable=unnecessary-ellipsis


class PerfectSimPredictor:
    """Ideal predictions for simulation: ``Y = X`` (same as true job size)."""

    def predict(self, size: float, _rng) -> float:
        """Return ``y = size`` (perfect prediction)."""
        return float(size)


class _PriorityHeapBuffer:
    """Minimal buffer API used by ``QsSim`` (``append`` / ``pop`` / ``size``)."""

    __slots__ = ("_pq",)

    def __init__(self, pq: PrioritySizeQueue) -> None:
        self._pq = pq

    def append(self, task: Task) -> None:
        """Push *task* onto the priority heap."""
        self._pq.push(task)

    def append_left(self, task: Task) -> None:
        """Same as ``append`` (priority order; no special LIFO head)."""
        self._pq.push(task)

    def pop(self) -> Task:
        """Pop the highest-priority task."""
        return self._pq.pop()

    def size(self) -> int:
        """Return the number of tasks in the queue."""
        return len(self._pq)


class SizeBasedQsSim(QsSim):
    """
    M/G/1-style simulator with size-at-arrival and priority scheduling.

    :param discipline: Scheduling policy (see roadmap).
    :param buffer: ``None`` for infinite buffer (recommended).
    :param track_slowdown: If True, append ``sojourn / original_size`` on each completion (see ``get_slowdown``).
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        num_of_channels: int = 1,
        discipline: SizeDiscipline = "SRPT",
        buffer: int | None = None,
        verbose: bool = True,
        buffer_type: str = "list",
        track_slowdown: bool = False,
    ) -> None:
        if num_of_channels != 1:
            raise ValueError("SizeBasedQsSim currently supports only num_of_channels=1.")
        valid: tuple[str, ...] = ("FCFS", "SJF", "PSJF", "SRPT", "SPJF", "PSPJF", "SPRPT")
        if discipline not in valid:
            raise ValueError(f"Unknown discipline {discipline!r}; expected one of {valid}.")

        super().__init__(num_of_channels, buffer, verbose, buffer_type)

        self._discipline: SizeDiscipline = discipline
        self._size_dist = None
        self._predictor: SizePredictor = PerfectSimPredictor()
        self._track_slowdown = bool(track_slowdown)
        self._slowdown_samples: list[float] = []

        self._pq = PrioritySizeQueue(self._rank_value)
        self.queue = _PriorityHeapBuffer(self._pq)

    def _handle_custom_event(self, event_type: str) -> None:
        """No custom events are registered for this simulator."""
        raise NotImplementedError(f"SizeBasedQsSim has no custom events ({event_type!r})")

    def set_predictor(self, predictor: SizePredictor | None) -> None:
        """Set predictor for SPJF / PSPJF / SPRPT. ``None`` restores perfect predictions."""
        self._predictor = PerfectSimPredictor() if predictor is None else predictor

    def get_slowdown(self) -> list[float]:
        """Return a copy of slowdown samples ``T / original_size`` if ``track_slowdown`` was True."""
        return list(self._slowdown_samples)

    def _after_serving(self, channel: int, task=None, is_network: bool = False) -> None:
        """Record slowdown T/X on job completion when ``track_slowdown`` is enabled."""
        super()._after_serving(channel, task, is_network=is_network)
        if not self._track_slowdown or task is None:
            return
        x = getattr(task, "original_size", None)
        if x is None or float(x) <= 0.0:
            return
        sojourn = float(self.ttek) - float(task.arr_time)
        self._slowdown_samples.append(sojourn / float(x))

    def set_servers(self, params, kendall_notation: str = "M") -> None:
        """Create service channels and a separate size sampler (same distribution)."""
        super().set_servers(params, kendall_notation)
        self._size_dist = create_distribution(params, kendall_notation, self.generator)

    def _is_preemptive(self) -> bool:
        return self._discipline in ("SRPT", "PSJF", "PSPJF", "SPRPT")

    def _rank_value(self, ts: Task):
        d = self._discipline
        if d == "FCFS":
            return (float(ts.arr_time),)
        if d in ("SJF", "PSJF"):
            return (float(ts.original_size or 0.0),)
        if d == "SRPT":
            rem = ts.service_remaining
            if rem is None:
                rem = float(ts.original_size or 0.0)
            return (float(rem),)
        if d in ("SPJF", "PSPJF"):
            return (float(ts.predicted_size or 0.0),)
        if d == "SPRPT":
            sz = float(ts.original_size or 0.0)
            rem = ts.service_remaining
            if rem is None:
                rem = sz
            served = sz - float(rem)
            pred = float(ts.predicted_size if ts.predicted_size is not None else sz)
            pred_rem = max(0.0, pred - served)
            return (float(pred_rem),)
        raise RuntimeError(f"Unhandled discipline {d!r}")

    def _new_strictly_better_than_current(self, new: Task, cur: Task) -> bool:
        kn = self._pq.comparison_key(new)
        kc = self._pq.comparison_key(cur)
        return kn < kc

    def _sample_size(self, ts: Task) -> None:
        """Sample true size at arrival; FCFS defers service sampling to ``Server``."""
        if self._discipline == "FCFS":
            return
        if self._size_dist is None:
            return
        sz = float(self._size_dist.generate())
        ts.original_size = sz
        ts.service_remaining = sz
        ts.predicted_size = float(self._predictor.predict(sz, self.generator))

    def _update_in_service_rank_fields(self, server_idx: int) -> None:
        """Refresh ``service_remaining`` on the in-service task to its true current value.

        This must be called before a preemption decision for disciplines whose rank
        depends on the remaining work (SRPT, SPRPT): the task's ``service_remaining``
        field is only written at (re-)start of service and becomes stale as time passes.
        Reading ``server.time_to_end_service - ttek`` gives the true remaining work.
        """
        srv = self.servers[server_idx]
        ts = srv.tsk_on_service
        if ts is None:
            return
        if self._discipline in ("SRPT", "SPRPT"):
            true_rem = max(0.0, float(srv.time_to_end_service) - float(self.ttek))
            ts.service_remaining = true_rem

    def _try_preempt_on_arrival(self, ts: Task) -> bool:
        """If the arriving job should preempt the job in service, do it. Returns True if handled."""
        if not self._is_preemptive():
            return False
        busy_idx = next((i for i in range(self.n) if not self.servers[i].is_free), None)
        if busy_idx is None:
            return False
        cur = self.servers[busy_idx].tsk_on_service
        if cur is None:
            return False

        self._update_in_service_rank_fields(busy_idx)

        if not self._new_strictly_better_than_current(ts, cur):
            return False

        old = self.servers[busy_idx].preempt_service(self.ttek, preserve_remaining=True)
        self._free_servers.add(busy_idx)
        self.free_channels += 1

        old.start_waiting_time = self.ttek
        self.queue.append(old)

        self.taked += 1
        self.refresh_w_stat(ts.wait_time)
        if ts.wait_time == 0:
            self.zero_wait_arrivals_num += 1
        self.servers[busy_idx].start_service(ts, self.ttek, False)
        self._free_servers.remove(busy_idx)
        self.free_channels -= 1
        self._mark_servers_time_changed()
        return True

    def arrival(self, moment: float | None = None, ts=None) -> None:
        """Handle arrival; preemptive disciplines may preempt the job in service."""
        self.arrived += 1
        self._update_state_probs(self.ttek, self.arrival_time, self.in_sys)

        if moment:
            self.ttek = moment
            if ts is not None:
                ts.arr_time = moment
                ts.wait_time = 0
                ts.start_waiting_time = -1
        else:
            self.ttek = self.arrival_time
            self.arrival_time = self.ttek + self.source.generate()

        self.in_sys += 1

        if ts is not None:
            self._sample_size(ts)

        if self.free_channels == 0:
            if ts is None:
                ts = Task(self.ttek)
                ts.start_waiting_time = self.ttek
                self._sample_size(ts)
            if self._try_preempt_on_arrival(ts):
                return
            self.send_task_to_queue(new_tsk=ts)
        else:
            self.send_task_to_channel(tsk=ts)
