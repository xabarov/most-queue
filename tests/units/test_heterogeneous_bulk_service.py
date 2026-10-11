"""Independent physical oracle, classical corners, source tables and API guards."""

import math

import numpy as np
import pytest

from examples.heterogeneous_bulk_service_oracle import physical_oracle
from most_queue.sim.heterogeneous_bulk_service import BulkServiceHeterogeneousSim
from most_queue.theory.batch import BulkServiceHeterogeneousCalc


def model(counts=(1, 1), rates=(1.2, 0.6), thresholds=(2, 3), capacities=(3, 4), l=1.3):
    m = BulkServiceHeterogeneousCalc(counts)
    m.set_sources(l)
    m.set_servers(rates, thresholds, capacities)
    return m


@pytest.mark.parametrize(
    "counts,rates,thresholds,capacities,l",
    [
        ((1, 1), (1.2, 0.6), (2, 3), (3, 4), 1.3),
        ((2, 1), (0.7, 1.1), (3, 2), (4, 3), 2.0),
        ((1, 1, 1), (1.3, 0.6, 0.9), (2, 2, 4), (2, 3, 5), 2.2),
        # Exercises removal of ahead jobs while the lower-priority type stays idle.
        ((1, 1), (1.1, 0.4), (1, 5), (2, 6), 2.1),
        ((1, 1), (1.1, 0.4), (1, 1), (1, 1), 0.8),
    ],
)
def test_independent_physical_and_tagged_chain(counts, rates, thresholds, capacities, l):
    m = model(counts, rates, thresholds, capacities, l)
    r = m.run()
    o = physical_oracle(counts, rates, thresholds, capacities, l, cap=70)
    assert o["boundary_mass"] < 1e-10
    assert r.mean_queue == pytest.approx(o["mean_queue"], rel=1e-7, abs=1e-9)
    assert r.w == pytest.approx(o["w"], rel=1e-7, abs=1e-9)
    assert r.v == pytest.approx(o["v"], rel=1e-7, abs=1e-9)
    assert r.served_probabilities == pytest.approx(o["served_probabilities"], rel=1e-8)
    assert r.zero_wait_probability == pytest.approx(o["zero_wait_probability"], abs=1e-9)
    assert [m.tail(t) for t in (0, 1, 3)] == pytest.approx(o["waiting_tails"], abs=1e-9)
    assert [m.tail(t, response=True) for t in (0, 1, 3)] == pytest.approx(o["response_tails"], abs=1e-9)
    assert [m.queue_probability(q) for q in range(15)] == pytest.approx(o["queue_pmf"][:15], abs=1e-9)


@pytest.mark.parametrize("c,l,mu", [(1, 0.6, 1), (2, 1, 1), (4, 3, 1), (3, 0.01, 1), (2, 1.98, 1)])
def test_erlang_c(c, l, mu):
    r = model((c,), (mu,), (1,), (1,), l).run()
    a = l / mu
    term = a**c / math.factorial(c) / (1 - a / c)
    delay = term / (sum(a**k / math.factorial(k) for k in range(c)) + term)
    assert r.zero_wait_probability == pytest.approx(1 - delay)
    assert r.w == pytest.approx([math.factorial(k) * delay / (c * mu - l) ** k for k in range(1, 4)])
    assert r.v[0] == pytest.approx(r.w[0] + 1 / mu)


@pytest.mark.parametrize(
    "thresholds,queue,zero",
    [
        # Table 2 diagonal priority labels are reversed; Table 3 is consistent.
        ((1, 1), 4.072, 0.077),
        ((2, 1), 3.604, 0.124),
        ((3, 3), 2.692, 0.194),
        ((4, 4), 2.236, 0.205),
        ((4, 6), 2.094, 0.196),
        ((5, 9), 2.350, 0.173),
    ],
)
def test_published_tables(thresholds, queue, zero):
    r = model((4, 2), (0.4, 0.2), thresholds, (5, 9), 6).run()
    assert r.mean_queue == pytest.approx(queue, abs=0.00051)
    assert r.zero_wait_probability == pytest.approx(zero, abs=0.00051)


def test_tie_priority_and_original_type_mapping():
    a = model((4, 2), (0.4, 0.2), (4, 4), (5, 9), 6).run()
    b = model((2, 4), (0.2, 0.4), (4, 4), (9, 5), 6).run()
    assert b.mean_queue == pytest.approx(2.389, abs=0.00051)
    assert b.zero_wait_probability == pytest.approx(0.195, abs=0.00051)
    assert a.mean_queue != pytest.approx(b.mean_queue)
    a = model((4, 2), (0.4, 0.2), (4, 6), (5, 9), 6).run()
    b = model((2, 4), (0.2, 0.4), (6, 4), (9, 5), 6).run()
    assert a.w == pytest.approx(b.w)
    assert a.v == pytest.approx(b.v)
    assert a.served_probabilities == pytest.approx(b.served_probabilities[::-1])


def test_ph_and_conservation():
    m = model()
    r = m.run(num=4)
    A, R = m.waiting_generator, m.waiting_absorption
    assert np.max(np.abs(A @ np.ones(A.shape[0]) + R.sum(axis=1))) < 1e-12
    assert np.min(R) >= 0
    assert max(r.root_residual, r.balance_residual, r.little_residual, r.flow_residual, r.class_residual) < 1e-10
    assert r.p is r.pi is None
    assert sum(m.queue_probability(q) for q in range(300)) == pytest.approx(1)
    assert sum(r.served_probabilities) == pytest.approx(1)
    assert r.batch_throughputs == pytest.approx(np.array(r.mean_busy_servers) * m.rates)
    assert r.v[0] == pytest.approx(r.w[0] + sum(p / mu for p, mu in zip(r.served_probabilities, m.rates)))
    assert m.tail(0) == pytest.approx(1 - r.zero_wait_probability)
    assert m.tail(0, response=True) == pytest.approx(1)
    for response in (False, True):
        assert m.lst(0, response=response) == pytest.approx(1)
        moments = r.v if response else r.w
        assert (1 - m.lst(1e-6, response=response)) / 1e-6 == pytest.approx(moments[0], rel=1e-4)
        tails = [m.tail(t, response=response) for t in (0, 0.5, 1, 2, 10, 100)]
        assert all(a >= b for a, b in zip(tails, tails[1:]))


def test_cache_reset_and_raw_moment_orders():
    m = model()
    before = m.run()
    assert m.get_w() == before.w
    m.set_sources(2)
    assert m._result is None and m.w is None and m.free_generator is None
    assert m.get_w() != before.w
    m.set_servers([1, 1], [1, 1], [1, 1])
    with pytest.raises(ValueError, match="load"):
        m.run()
    m.set_sources(1)
    assert len(m.run(num=8).v) == 8


@pytest.mark.parametrize("counts", [(), (0,), (-1,), (True,), (1.5,)])
def test_bad_counts(counts):
    with pytest.raises(ValueError):
        BulkServiceHeterogeneousCalc(counts)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_bad_arrival_rates(value):
    with pytest.raises(ValueError):
        model().set_sources(value)


@pytest.mark.parametrize(
    "rates,thresholds,capacities",
    [
        ((1,), (1, 2), (2, 3)),
        ((1, 0), (1, 1), (1, 1)),
        ((1, 1), (2, 3), (1, 4)),
        ((1, 1), (True, 1), (2, 2)),
        ((1, 1), (1, 1), (2, 1025)),
        ((1, 1), (50, 50), (50, 50)),
    ],
)
def test_bad_profiles(rates, thresholds, capacities):
    with pytest.raises(ValueError):
        model().set_servers(rates, thresholds, capacities)


def test_resource_guard_before_state_materialization():
    m = BulkServiceHeterogeneousCalc([1000, 1000])
    with pytest.raises(ValueError, match="20000"):
        m.set_servers([1, 1], [1, 1], [1, 1])


@pytest.mark.parametrize("c,a,b,l", [(1, 2, 4, 1), (2, 2, 3, 2), (3, 1, 2, 2)])
def test_identical_types_match_existing_bulk_solver(c, a, b, l):
    from most_queue.theory.batch.bulk_service_multiserver import BulkServiceMultiserverCalc

    old = BulkServiceMultiserverCalc(a=a, b=b, c=c, queue_truncation=90)
    old.set_sources(l)
    old.set_servers(1)
    new = model((c,), (1,), (a,), (b,), l)
    assert new.run().w == pytest.approx(old.get_w(num=3), rel=1e-7)


def test_identical_type_regrouping():
    a = model((3,), (1,), (2,), (4,), 2).run()
    b = model((1, 2), (1, 1), (2, 2), (4, 4), 2).run()
    assert a.w == pytest.approx(b.w)
    assert a.v == pytest.approx(b.v)
    assert a.mean_queue == pytest.approx(b.mean_queue)


def test_table_diagonal_discrepancy_independent_arbiters():
    # Table 2, (TA,TB)=(4,4): printed A-priority=2.389, B-priority=2.236.
    # Physical CTMC finds the reverse; Table 3's zero-wait priority labels match.
    counts, rates, thresholds, capacities, l = (4, 2), (0.4, 0.2), (4, 4), (5, 9), 6
    o = physical_oracle(counts, rates, thresholds, capacities, l, cap=110)
    assert o["boundary_mass"] < 3e-11
    assert o["mean_queue"] == pytest.approx(2.23641674084, abs=5e-9)
    assert o["zero_wait_probability"] == pytest.approx(0.20465484050, abs=1e-10)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1"])
def test_bad_tail_lst(value):
    m = model()
    with pytest.raises(ValueError):
        m.tail(value)
    with pytest.raises(ValueError):
        m.lst(value)


@pytest.mark.parametrize("value", [0, 9, True, 1.5])
def test_bad_moment_orders(value):
    with pytest.raises(ValueError):
        model().run(num=value)


def simulator(seed=42):
    s = BulkServiceHeterogeneousSim([1, 1], seed=seed)
    s.set_sources(1)
    s.set_servers([1, 0.5], [2, 3], [3, 4])
    return s


def test_hand_replay_fcfs_simultaneous_batch_completion_and_exposure():
    s = simulator()
    r = s.run(
        10, arrivals=[0, 1, 2, 3, 4], service_times=[[5, 5], [7]], tail_times=[0, 1, 2], cohort_end=5, record_trace=True
    )
    assert r.iteration_trace == [
        {"time": 1.0, "type": 0, "server": 0, "batch": 2, "queue": 0},
        {"time": 4.0, "type": 1, "server": 0, "batch": 3, "queue": 0},
    ]
    assert (r.arrived, r.started, r.completed, r.unfinished) == (5, 5, 2, 3)
    assert r.total_queue_area == 4
    assert r.total_population_area == 32
    assert r.started_wait_exposure == 4
    assert r.completed_residence == 11
    assert r.unfinished_residence == 21
    assert r.waiting_tails == pytest.approx([0.6, 0.2, 0])
    assert r.response_tails == [1, 1, 1]


def test_unfinished_cohort_tails_not_completed_only():
    s = simulator()
    s.set_servers([1, 0.5], [3, 3], [3, 4])
    r = s.run(10, arrivals=[0, 1], service_times=[[], []], cohort_end=2, tail_times=[0, 1, 5, 8])
    assert r.cohort_unstarted == r.cohort_unfinished == r.cohort_size == 2
    assert r.waiting_tails == r.response_tails == [1, 1, 1, 1]
    assert r.total_queue_area == r.unfinished_wait_exposure == 19
    assert r.w is r.v is None


@pytest.mark.parametrize("seed", [1, 10, 100])
def test_sim_accounting_and_reproducibility(seed):
    s = simulator(seed)
    r = s.run(1000, warmup=100, tail_times=[0, 2, 5])
    again = s.run(1000, warmup=100, tail_times=[0, 2, 5])
    assert r.arrived == r.completed + r.unfinished
    assert r.total_queue_area == pytest.approx(r.started_wait_exposure + r.unfinished_wait_exposure)
    assert r.total_population_area == pytest.approx(r.completed_residence + r.unfinished_residence)
    assert r.waiting_proxy == again.waiting_proxy
    assert r.waiting_tails == again.waiting_tails
    assert r.customer_throughputs == again.customer_throughputs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"horizon": 0},
        {"horizon": 10, "warmup": 10},
        {"horizon": 10, "tail_times": [6]},
        {"horizon": 10, "arrivals": [1, 0]},
        {"horizon": 10, "arrivals": [10]},
        {"horizon": 10, "arrivals": [float("nan")]},
        {"horizon": 10, "max_events": True},
        {"horizon": 10, "service_times": [[0], []]},
    ],
)
def test_sim_bad_run(kwargs):
    with pytest.raises(ValueError):
        simulator().run(**kwargs)


def test_sim_exhaustion_and_event_budget():
    with pytest.raises(ValueError, match="exhausted"):
        simulator().run(10, arrivals=[0, 1], service_times=[[], []])
    with pytest.raises(RuntimeError, match="budget"):
        simulator().run(100, max_events=1)
