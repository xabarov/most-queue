"""Independent stage enumeration, finite packing LP and exact replay arbiters."""

import itertools

import numpy as np
import pytest
from scipy.optimize import linprog

from most_queue.sim.kv_cache_stability import KvCacheStabilitySim
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.continuous_batching import KvCacheStabilityCalc


def configured(memory=40, chunk=2, pmf=None, rate=1, slot=0.1, **kwargs):
    calc = KvCacheStabilityCalc(memory, chunk, **kwargs)
    calc.set_sources(rate)
    calc.set_servers(pmf or {(4, 2): 1}, slot)
    return calc


@pytest.mark.parametrize("chunk,s,o", [(1, 1, 1), (2, 4, 2), (3, 12, 7), (512, 1024, 256)])
def test_area_is_enumerated_future_stage_footprints(chunk, s, o):
    stages = list(range(chunk, s + 1, chunk)) + list(range(s + 1, s + o + 1))
    calc = configured(memory=10 * (s + o), chunk=chunk, pmf={(s, o): 1})
    assert isinstance(calc, BaseQueue)
    assert calc.memory_area(s, o) == sum(stages)
    result = calc.run()
    assert result.mean_memory_area == sum(stages)
    assert result.capacity_upper == pytest.approx(calc.memory_tokens / (0.1 * sum(stages)))
    assert result.w is result.v is result.p is result.pi is None
    assert result.duration >= 0


@pytest.mark.parametrize(
    "rate,status",
    [
        (0, "stable_under_assumptions"),
        (19, "stable_under_assumptions"),
        (20, "inconclusive"),
        (25, "inconclusive"),
        (30, "inconclusive"),
        (31, "overloaded"),
    ],
)
def test_strict_boundaries(rate, status):
    # g(1,1)=3, M=6, slot=.1 => upper=20, lower=13.333...;
    # use slot=1/15 for exactly representable resource_time=.2.
    calc = configured(memory=6, chunk=1, pmf={(1, 1): 1}, rate=rate, slot=1 / 15)
    r = calc.run()
    assert r.capacity_upper == 30
    assert r.capacity_lower == 20
    assert r.status == status
    assert r.utilization == pytest.approx(rate / 30)
    assert r.expected_drift_upper == pytest.approx(rate * 0.2 - 4)


@pytest.mark.parametrize(
    "flags",
    [{"maximal_packing": False}, {"zero_cost_swap": False}, {"maximal_packing": False, "zero_cost_swap": False}],
)
def test_resident_or_nonmaximal_policy_has_no_sufficient_certificate(flags):
    r = configured(**flags).run()
    assert r.status == "inconclusive"
    assert r.capacity_lower is r.expected_drift_upper is r.sufficient_replicas is None
    assert configured(rate=100, **flags).run().status == "overloaded"


def test_correlated_lengths_and_zero_mass_essential_supremum():
    positive = configured(chunk=1, pmf={(2, 2): 0.5, (6, 6): 0.5, (100, 100): 0})
    negative = configured(chunk=1, pmf={(2, 6): 0.5, (6, 2): 0.5})
    a, b = positive.run(), negative.run()
    assert a.length_covariance == 4
    assert b.length_covariance == -4
    assert a.mean_memory_area - b.mean_memory_area == 8
    assert a.peak_tokens == 12
    assert b.peak_tokens == 8
    assert a.capacity_upper < b.capacity_upper


def test_replica_bounds_strictness_and_zero_rate():
    calc = configured(memory=6, chunk=1, pmf={(1, 1): 1}, rate=40, slot=1 / 15)
    r = calc.run()
    assert (r.necessary_replicas, r.sufficient_replicas) == (2, 3)
    calc.set_sources(0)
    assert calc.run().necessary_replicas == calc.run().sufficient_replicas == 0
    calc.set_sources(30)
    assert calc.run().necessary_replicas == 1  # Equality is necessary, NOT sufficient.
    assert calc.run().status == "inconclusive"


def test_delta_one_has_no_positive_stable_interval():
    r = configured(memory=6, rate=0).run()
    assert r.delta == 1
    assert r.capacity_lower == 0
    assert r.status == "inconclusive"
    assert r.sufficient_replicas is None


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan"), "4"])
@pytest.mark.parametrize("cls", [KvCacheStabilityCalc, KvCacheStabilitySim])
def test_constructor_invalid_memory_and_chunk(cls, value):
    with pytest.raises(ValueError):
        cls(value)
    with pytest.raises(ValueError):
        cls(40, value)


@pytest.mark.parametrize("rate", [-1, float("inf"), float("nan"), True, "1"])
@pytest.mark.parametrize("cls", [KvCacheStabilityCalc, KvCacheStabilitySim])
def test_invalid_arrival_rates(cls, rate):
    with pytest.raises(ValueError):
        cls(40).set_sources(rate)


@pytest.mark.parametrize(
    "pmf",
    [
        {},
        {(4, 2): 0.5},
        {(4, 2): 0},
        {(4, 2): -1},
        {(4, 2): float("nan")},
        {(4, 2): True},
        {(4, 0): 1},
        {(4.0, 2): 1},
        {(True, 2): 1},
        {(3, 2): 1},
        {(40, 2): 1},
        {4: 1},
        {(4, 2, 1): 1},
    ],
)
@pytest.mark.parametrize("cls", [KvCacheStabilityCalc, KvCacheStabilitySim])
def test_invalid_joint_support(cls, pmf):
    with pytest.raises(ValueError):
        cls(40, 2).set_servers(pmf, 0.1)


@pytest.mark.parametrize("slot", [0, -1, float("inf"), float("nan"), True, "1"])
@pytest.mark.parametrize("cls", [KvCacheStabilityCalc, KvCacheStabilitySim])
def test_invalid_slot_duration(cls, slot):
    with pytest.raises(ValueError):
        cls(40, 2).set_servers({(4, 2): 1}, slot)


def test_configuration_required_atomic_setters_and_no_stale_results():
    calc = KvCacheStabilityCalc(40, 2)
    with pytest.raises(ValueError):
        calc.run()
    calc.set_sources(1)
    with pytest.raises(ValueError):
        calc.run()
    calc.set_servers({(4, 2): 1}, 0.1)
    first = calc.run()
    first.diagnostics.clear()
    assert calc.run().diagnostics
    with pytest.raises(ValueError):
        calc.set_servers({(3, 2): 1}, 0.2)
    assert calc.run().capacity_upper == first.capacity_upper
    calc.set_servers({(4, 2): 1}, 0.2)
    assert calc.run().capacity_upper == first.capacity_upper / 2


@pytest.mark.parametrize("slot", [1e-324, 1e308])
def test_underflow_overflow_rejected(slot):
    with pytest.raises(ValueError):
        configured(slot=slot).run()


def test_literal_appendix_remaining_work_has_off_by_one_errors():
    # Independent direct sums, NOT production remaining-work formulas.
    s, o, chunk, c, d = 4, 2, 2, 1, 1
    true_prefill_remaining = sum([4, 5, 6])
    literal_prefill = (c * chunk + s) * (s / chunk - c + 1) / 2 + o * s + o * (o + 1) / 2
    true_decode_remaining = sum([6])
    literal_decode = s * (o - d + 1) + (o - d + 1) * (o + d) / 2
    assert (true_prefill_remaining, literal_prefill) == (15, 17)
    assert (true_decode_remaining, literal_decode) == (6, 11)


def make_sim(memory=5, pmf=None, chunk=1):
    sim = KvCacheStabilitySim(memory, chunk, seed=42)
    sim.set_sources(0)
    sim.set_servers(pmf or {(1, 1): 1}, 1)
    return sim


def test_hand_replay_growth_swap_and_release():
    sim = make_sim(memory=4, pmf={(2, 2): 1}, chunk=2)
    r = sim.run(6, arrivals_by_slot=[[(2, 2), (2, 2)], [], [], [], [], []], record_trace=True)
    assert [x["fulfilled"] for x in r.slot_trace] == [4, 3, 4, 3, 4, 0]
    assert [x["backlog_after"] for x in r.slot_trace] == [2, 2, 1, 1, 0, 0]
    assert r.swapped_request_slots == 2
    assert r.input_memory_area == r.fulfilled_memory_area == 18
    assert r.outstanding_memory_area == r.unfinished == 0
    assert r.completed == 2
    assert r.w is r.v is r.p is r.pi is None


def test_scan_past_nonfitting_ongoing_request_is_maximal_not_hol():
    sim = make_sim(memory=5, pmf={(4, 1): 1})
    r = sim.run(3, arrivals_by_slot=[[(4, 1)] * 3, [], []], record_trace=True)
    assert r.slot_trace[2]["selected"] == [(0, 2, 3), (2, 1, 2)]
    assert r.slot_trace[2]["fulfilled"] == 5
    assert r.unfinished == 3
    assert r.completed == 0  # No draining; finite unfinished cohort is preserved.


def test_full_small_configuration_maximality_and_accounting():
    # Exhaustively enumerate all ordered batches of 6 requests from 3 types.
    types = [(1, 1), (2, 1), (3, 1)]
    sim = make_sim(5, dict.fromkeys(types, 1 / 3))
    for arrivals in itertools.product(types, repeat=6):
        r = sim.run(3, arrivals_by_slot=[list(arrivals), [], []], record_trace=True)
        for slot in r.slot_trace:
            assert slot["work_after"] == slot["work_before"] - slot["fulfilled"]
            assert 0 <= slot["fulfilled"] <= 5
            if slot["backlog_before"] > 5:
                assert slot["fulfilled"] >= 5 - 4  # max footprint = 4
        assert r.arrived == r.completed + r.unfinished
        assert r.input_memory_area == r.fulfilled_memory_area + r.outstanding_memory_area


@pytest.mark.parametrize("memory", [2, 3, 4, 5, 6, 7, 8])
def test_independent_configuration_lp_and_saturated_cycle(memory):
    # Stage balance LP, built by enumerating all integer packing configurations.
    # footprint 1 then 2; every completed request requires one of each stage.
    configs = [(a, b) for a in range(memory + 1) for b in range(memory + 1) if a + 2 * b <= memory]
    variables = len(configs)
    objective = np.r_[np.zeros(variables), -1]
    equality = np.array([[1] * variables + [0], [a for a, _ in configs] + [-1], [b for _, b in configs] + [-1]])
    lp = linprog(objective, A_eq=equality, b_eq=[1, 0, 0], bounds=(0, None), method="highs")
    assert lp.success
    exact_capacity = -lp.fun
    calc = configured(memory=memory, chunk=1, pmf={(1, 1): 1}, slot=1).run()
    assert calc.capacity_lower <= exact_capacity <= calc.capacity_upper + 1e-12
    # Explicit ongoing-first cycles attain the LP optimum for these capacities.
    sim = make_sim(memory)
    r = sim.run(120, warmup_slots=30, arrivals_by_slot=[[(1, 1)] * memory] * 120, record_trace=True)
    assert r.throughput == pytest.approx(exact_capacity, abs=1 / 90)
    assert exact_capacity == pytest.approx(memory / 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_of_slots": 0},
        {"num_of_slots": True},
        {"num_of_slots": 1.2},
        {"num_of_slots": 2, "warmup_slots": 2},
        {"num_of_slots": 2, "warmup_slots": -1},
        {"num_of_slots": 2, "arrivals_by_slot": [[]]},
        {"num_of_slots": 1, "arrivals_by_slot": [[(2, 1)]]},
        {"num_of_slots": 1, "arrivals_by_slot": [[(True, 1)]]},
        {"num_of_slots": 1, "record_trace": 1},
    ],
)
def test_invalid_simulation_run(kwargs):
    with pytest.raises(ValueError):
        make_sim().run(**kwargs)


def test_seed_reproducibility_warmup_and_unconfigured_sim():
    with pytest.raises(ValueError):
        KvCacheStabilitySim(5).run(10)
    sim = make_sim()
    sim.set_sources(1.8)
    a = sim.run(30, record_trace=True)
    b = sim.run(30, warmup_slots=20, record_trace=True)
    assert a.slot_trace == b.slot_trace
    assert b.observation_time == 10
    final = b.slot_trace[20:]
    completions = sum(t["backlog_before"] - t["backlog_after"] for t in final)
    assert b.throughput == completions / 10
    assert b.memory_fraction == sum(t["fulfilled"] for t in final) / 50


def test_boolean_assumptions_and_area_domain():
    with pytest.raises(ValueError):
        KvCacheStabilityCalc(40, maximal_packing=1)
    with pytest.raises(ValueError):
        configured().memory_area(3, 2)
    with pytest.raises(ValueError):
        configured().memory_area(4, 0)


def test_sim_stage_limit_and_workload_size_limit():
    with pytest.raises(ValueError, match="stage"):
        KvCacheStabilitySim(2_000_000).set_servers({(1_000_000, 1): 1}, 1)

    # len() check fires before any request allocation.
    class OversizedBatch(list):
        def __len__(self):
            return 2_000_001

    with pytest.raises(ValueError, match="request"):
        make_sim().run(1, arrivals_by_slot=[OversizedBatch()])
