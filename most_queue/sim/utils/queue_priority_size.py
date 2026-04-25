"""
Priority queue for size-based scheduling (min-heap by rank).

Heap entries store a comparison key at push time. ``rank_fn`` may return a
``float`` / ``int`` or a ``tuple`` of numeric components (lexicographic order).
Tie-break: ``(arr_time, task.id)`` is always appended to the key.

If job ranks change while queued, callers should ``remove`` + ``push`` again.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from typing import Union

from most_queue.sim.utils.tasks import Task

RankValue = Union[float, int, tuple[Union[float, int], ...]]
RankFn = Callable[[Task], RankValue]


class PrioritySizeQueue:
    """
    Min-heap keyed by ``rank_fn(task)`` then ``(arr_time, task.id)``.

    Lazy deletion: ``remove`` marks a task id; stale heap entries are skipped on
    ``pop`` / ``peek``.
    """

    __slots__ = ("_heap", "_removed_ids", "_logical_len", "_rank_fn")

    def __init__(self, rank_fn: RankFn):
        """
        :param rank_fn: Callable mapping a Task to its scheduling rank.
            Smaller key ? higher priority.  Examples::

                SRPT: lambda t: (t.service_remaining, t.arr_time, t.id)
                SJF:  lambda t: (t.original_size, t.arr_time, t.id)
                SPJF: lambda t: (t.predicted_size, t.arr_time, t.id)

            A scalar rank is equivalent to ``(float(rank),)``.
        """
        self._rank_fn = rank_fn
        self._heap: list[tuple[tuple[Union[float, int], ...], Task]] = []
        self._removed_ids: set[int] = set()
        self._logical_len = 0

    def _comparison_key(self, task: Task) -> tuple[Union[float, int], ...]:
        rank = self._rank_fn(task)
        tail = (float(task.arr_time), int(task.id))
        if isinstance(rank, (int, float)):
            return (float(rank),) + tail
        if isinstance(rank, tuple):
            return tuple(float(x) if isinstance(x, (int, float)) else x for x in rank) + tail
        raise TypeError(f"rank_fn must return float, int, or tuple; got {type(rank)!r}")

    def comparison_key(self, task: Task) -> tuple[Union[float, int], ...]:
        """Full heap ordering key for *task* (for preemption / debugging)."""
        return self._comparison_key(task)

    def push(self, task: Task) -> None:
        """Enqueue *task*. Its rank is evaluated immediately via ``rank_fn``."""
        key = self._comparison_key(task)
        heapq.heappush(self._heap, (key, task))
        self._logical_len += 1

    def pop(self) -> Task:
        """Remove and return the task with the smallest rank.

        :raises IndexError: if the queue is empty.
        """
        while self._heap:
            _key, task = heapq.heappop(self._heap)
            tid = int(task.id)
            if tid in self._removed_ids:
                self._removed_ids.discard(tid)
                continue
            self._logical_len -= 1
            return task
        raise IndexError("pop from empty PrioritySizeQueue")

    def peek(self) -> Task | None:
        """Return the task with the smallest rank without removing it.

        Returns ``None`` if the queue is empty.
        """
        while self._heap:
            _key, task = self._heap[0]
            tid = int(task.id)
            if tid in self._removed_ids:
                heapq.heappop(self._heap)
                self._removed_ids.discard(tid)
                continue
            return task
        return None

    def remove(self, task: Task) -> None:
        """Logically remove *task* (lazy; physical entry may remain until popped)."""
        tid = int(task.id)
        if tid in self._removed_ids:
            return
        self._removed_ids.add(tid)
        self._logical_len -= 1

    def __len__(self) -> int:
        """Return the number of logically present tasks."""
        return self._logical_len

    def is_empty(self) -> bool:
        """Return ``True`` if the queue contains no tasks."""
        return self._logical_len == 0
