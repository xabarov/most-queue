"""Unit tests for PrioritySizeQueue (size-based scheduling heap)."""

import pytest

from most_queue.sim.utils.queue_priority_size import PrioritySizeQueue
from most_queue.sim.utils.tasks import Task


def _task(arr_time: float, job_size: float) -> Task:
    t = Task(arr_time)
    t.original_size = job_size
    return t


def test_pop_order_by_rank_not_fifo():
    q = PrioritySizeQueue(rank_fn=lambda t: float(t.original_size or 0.0))
    q.push(_task(0.0, 3.0))
    q.push(_task(0.0, 1.0))
    q.push(_task(0.0, 2.0))
    assert q.pop().original_size == 1.0
    assert q.pop().original_size == 2.0
    assert q.pop().original_size == 3.0
    assert q.is_empty()


def test_remove_lazy_deletion():
    q = PrioritySizeQueue(rank_fn=lambda t: float(t.original_size or 0.0))
    a = _task(0.0, 1.0)
    b = _task(0.0, 2.0)
    c = _task(0.0, 3.0)
    q.push(a)
    q.push(b)
    q.push(c)
    assert len(q) == 3
    q.remove(b)
    assert len(q) == 2
    out = [q.pop().id, q.pop().id]
    assert set(out) == {a.id, c.id}
    assert q.is_empty()


def test_tie_break_arrival_time_then_task_id():
    q = PrioritySizeQueue(rank_fn=lambda t: float(t.original_size or 0.0))
    t_late = _task(10.0, 1.0)
    t_early = _task(5.0, 1.0)
    q.push(t_late)
    q.push(t_early)
    # same rank -> smaller arr_time first
    assert q.pop().arr_time == 5.0
    assert q.pop().arr_time == 10.0


def test_tie_break_task_id_same_rank_and_arrival():
    q = PrioritySizeQueue(rank_fn=lambda t: float(t.original_size or 0.0))
    first = _task(1.0, 1.0)
    second = _task(1.0, 1.0)
    q.push(second)
    q.push(first)
    # same (rank, arr_time) -> smaller task.id first (heap tuple order)
    ids = [q.pop().id, q.pop().id]
    assert ids == sorted(ids)


def test_peek_returns_min_without_removing():
    q = PrioritySizeQueue(rank_fn=lambda t: float(t.original_size or 0.0))
    q.push(_task(0.0, 2.0))
    q.push(_task(0.0, 1.0))
    assert q.peek() is not None and q.peek().original_size == 1.0
    assert len(q) == 2
    q.pop()
    q.pop()
    assert q.peek() is None


def test_pop_empty_raises_index_error():
    q = PrioritySizeQueue(rank_fn=lambda t: 0.0)
    with pytest.raises(IndexError):
        q.pop()


def test_len_and_is_empty():
    q = PrioritySizeQueue(rank_fn=lambda t: float(t.original_size or 0.0))
    assert len(q) == 0
    assert q.is_empty()
    q.push(_task(0.0, 1.0))
    assert len(q) == 1
    assert not q.is_empty()
    q.pop()
    assert q.is_empty()
