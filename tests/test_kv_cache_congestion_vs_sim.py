"""Finite stochastic flow checks; these runs do not prove queue recurrence."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.sim.kv_cache_congestion import KvCacheCongestionSim
from most_queue.theory.continuous_batching import KvCacheCongestionCalc

PARAMS = yaml.safe_load(Path(__file__).with_name("default_params.yaml").read_text())["kv_cache_congestion"]


@pytest.mark.parametrize("rate", [2.5, 4.5])
def test_integer_cap_matches_arrivals_at_subcritical_flow(rate):
    # Exact integer equilibrium cap=5; no fractional-cap rounding assumption.
    calculator = KvCacheCongestionCalc(60)
    calculator.set_sources([1])
    calculator.set_servers([2], [3])
    assert calculator.run().nominal_throughput == 5
    rates = []
    for seed in range(PARAMS["replications"]):
        simulator = KvCacheCongestionSim(60, seed=7000 + seed)
        simulator.set_sources([rate])
        simulator.set_servers([2], [3])
        result = simulator.run(PARAMS["iterations"], admission_cap=5, warmup=PARAMS["warmup"])
        rates.append(result.throughput)
        assert result.evictions == 0
        assert result.arrived == result.completed + result.unfinished
        assert result.executed_work == result.completed_work + result.remaining_work + result.discarded_work
        assert result.w is None and result.v is None
    se = np.std(rates, ddof=1) / np.sqrt(len(rates))
    assert abs(np.mean(rates) - rate) <= PARAMS["standard_errors"] * se + PARAMS["absolute_floor"]


def test_overload_is_not_hidden_by_draining():
    simulator = KvCacheCongestionSim(60, seed=7010)
    simulator.set_sources([7])
    simulator.set_servers([2], [3])
    result = simulator.run(4000, admission_cap=5, warmup=500)
    assert result.final_queue > 7000
    assert result.throughput == 5
    assert result.evictions == 0
    assert result.arrived == result.completed + result.unfinished


@pytest.mark.slow
def test_long_poisson_accounting_without_drain():
    simulator = KvCacheCongestionSim(626, seed=7020)
    simulator.set_sources([2.5, 2.5])
    simulator.set_servers([50, 50], [2, 4])
    result = simulator.run(30000, admission_cap=4, warmup=1000)
    assert result.arrived == result.completed + result.unfinished
    assert result.executed_work == result.completed_work + result.discarded_work + result.remaining_work
    assert result.unfinished > 20000
    assert 0 <= result.memory_fraction <= 1
