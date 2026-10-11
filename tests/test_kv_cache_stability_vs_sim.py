"""Fixed-horizon stochastic flow and overload checks, not recurrence proofs."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.sim.kv_cache_stability import KvCacheStabilitySim
from most_queue.theory.continuous_batching import KvCacheStabilityCalc

PARAMS = yaml.safe_load((Path(__file__).parent / "default_params.yaml").read_text())["kv_cache_stability"]


@pytest.mark.parametrize("memory,pmf", [(40, {(4, 2): 1}), (80, {(2, 2): 0.5, (6, 6): 0.5})])
def test_stable_certificate_independent_poisson_throughput(memory, pmf):
    calc = KvCacheStabilityCalc(memory, 2)
    calc.set_sources(1)
    calc.set_servers(pmf, 1)
    rate = 0.7 * calc.run().capacity_lower
    calc.set_sources(rate)
    assert calc.run().status == "stable_under_assumptions"
    values = []
    for seed in range(PARAMS["replications"]):
        sim = KvCacheStabilitySim(memory, 2, seed=1500 + seed)
        sim.set_sources(rate)
        sim.set_servers(pmf, 1)
        r = sim.run(PARAMS["slots_per_replication"], warmup_slots=PARAMS["warmup_slots"])
        assert r.arrived == r.completed + r.unfinished
        assert r.input_memory_area == r.fulfilled_memory_area + r.outstanding_memory_area
        values.append(r.throughput)
    mean, se = np.mean(values), np.std(values, ddof=1) / np.sqrt(len(values))
    assert abs(mean - rate) <= PARAMS["standard_errors"] * se + PARAMS["absolute_floor"]


def test_overload_work_growth_and_slot_lower_bound():
    sim = KvCacheStabilitySim(40, 2, seed=1717)
    # g(4,2) = 2+4+5+6 = 17 => capacity upper 40/17.
    sim.set_sources(4)
    sim.set_servers({(4, 2): 1}, 1)
    r = sim.run(1000, record_trace=True)
    assert r.outstanding_memory_area >= 17 * r.arrived - 40000
    assert r.outstanding_memory_area > 20000
    for slot in r.slot_trace:
        if slot["backlog_before"] > 40:
            assert slot["fulfilled"] >= 34


@pytest.mark.slow
def test_long_stable_joint_length_flow_and_no_backlog_trend():
    sim = KvCacheStabilitySim(200, 2, seed=1818)
    sim.set_sources(4)
    sim.set_servers({(2, 6): 0.5, (6, 2): 0.5}, 1)
    r = sim.run(100000, warmup_slots=10000, record_trace=True)
    assert r.throughput == pytest.approx(4, abs=PARAMS["absolute_floor"])
    assert r.outstanding_memory_area < 2000
    assert r.input_memory_area == r.fulfilled_memory_area + r.outstanding_memory_area
