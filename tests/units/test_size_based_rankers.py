"""Unit tests for SizeBasedQsSim rank ordering via PrioritySizeQueue."""

from most_queue.sim.size_based import SizeBasedQsSim
from most_queue.sim.utils.queue_priority_size import PrioritySizeQueue
from most_queue.sim.utils.tasks import Task


def _pq_for(discipline: str) -> tuple[SizeBasedQsSim, PrioritySizeQueue]:
    sim = SizeBasedQsSim(1, discipline=discipline)  # type: ignore[arg-type]
    pq = PrioritySizeQueue(sim._rank_value)
    return sim, pq


def test_srpt_preempt_order_smaller_remaining_first():
    _sim, pq = _pq_for("SRPT")
    a = Task(1.0)
    a.size = 10.0
    a.service_remaining = 5.0
    b = Task(2.0)
    b.size = 10.0
    b.service_remaining = 8.0
    assert pq.comparison_key(a) < pq.comparison_key(b)


def test_psjf_uses_original_size_not_remaining():
    _sim, pq = _pq_for("PSJF")
    a = Task(1.0)
    a.size = 3.0
    a.service_remaining = 0.1
    b = Task(2.0)
    b.size = 5.0
    b.service_remaining = 4.9
    assert pq.comparison_key(a) < pq.comparison_key(b)


def test_sprpt_predicted_remaining_rank():
    _sim, pq = _pq_for("SPRPT")
    t = Task(0.0)
    t.size = 10.0
    t.predicted_size = 8.0
    t.service_remaining = 4.0
    served = 6.0
    assert abs((t.size or 0) - (t.service_remaining or 0) - served) < 1e-9
    key = pq.comparison_key(t)
    assert key[0] == max(0.0, 8.0 - 6.0)


def test_srpt_preempt_uses_true_remaining_not_stale():
    """After time passes, preemption comparison must use actual remaining, not stale field."""
    import numpy as np

    from most_queue.random.distributions import ExpDistribution
    from most_queue.sim.size_based import SizeBasedQsSim

    rng = np.random.default_rng(99)
    sim = SizeBasedQsSim(1, discipline="SRPT")
    sim.generator = rng
    mu = 1.0
    sim.set_servers(mu, "M")
    sim.set_sources(0.5, "M")

    # Create task in service with large original size but near completion
    from most_queue.sim.utils.tasks import Task as T

    cur = T(0.0)
    cur.size = 10.0
    cur.service_remaining = 10.0  # stale â€” was set at t=0 when service started
    sim.servers[0].tsk_on_service = cur
    sim.servers[0].is_free = False
    sim.servers[0].time_to_end_service = 0.5  # true remaining = 0.5 (started at t=0)
    sim.free_channels = 0
    sim.ttek = 9.5  # now time is 9.5, so server ends at 0.5 => rem = 0.5 - 9.5 < 0... let's fix

    # Restart scenario: service started at t=0, size=10, ends at t=10
    sim.ttek = 9.0
    sim.servers[0].time_to_end_service = 10.0  # ends at t=10, so true remaining = 1.0

    # New arrival has size=5; stale comparison: 5 < 10 => preempt (WRONG)
    # Correct comparison: 5 < 1.0 => no preempt
    new_ts = T(sim.ttek)
    new_ts.size = 5.0
    new_ts.service_remaining = 5.0
    new_ts.predicted_size = 5.0
    new_ts.wait_time = 0

    sim._update_in_service_rank_fields(0)
    assert abs(cur.service_remaining - 1.0) < 1e-9, "Must update stale service_remaining before comparing"

    result = sim._try_preempt_on_arrival(new_ts)
    assert not result, "Should NOT preempt: new task (rem=5) has more remaining than in-service (rem=1)"
