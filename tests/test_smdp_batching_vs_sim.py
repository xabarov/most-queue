"""Finite real-queue observations are not infinite-optimality proofs."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.sim.smdp_batching import BatchSmdpSim
from most_queue.theory.batch import BatchSmdpCalc

PARAMS = yaml.safe_load(Path(__file__).with_name("default_params.yaml").read_text())["smdp_batching"]


def assert_accounting(result):
    assert result.initial_jobs + result.arrived == result.completed + result.unfinished
    assert result.total_population_area == pytest.approx(
        result.completed_residence + result.unfinished_residence, rel=1e-10
    )


def test_md1_nonexponential_corner():
    rate, period = 1, 0.4
    exact = period + rate * period**2 / (2 * (1 - rate * period))
    calc = BatchSmdpCalc(30, energy_weight=0)
    calc.set_sources(rate)
    calc.set_servers([period], [1])
    analytic = calc.run()
    assert analytic.response_proxy == pytest.approx(exact, abs=1e-8)
    values = []
    for seed in range(PARAMS["replications"]):
        sim = BatchSmdpSim(seed=9100 + seed, energy_weight=0)
        sim.set_sources(rate)
        sim.set_servers([period], [1])
        result = sim.run(PARAMS["horizon"], analytic.policy, warmup=PARAMS["warmup"])
        values.append(result.response_proxy)
        assert_accounting(result)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    assert abs(np.mean(values) - exact) <= PARAMS["standard_errors"] * se + PARAMS["absolute_floor"]


@pytest.mark.parametrize("energy_weight", [0, 4])
def test_small_tail_policy_vs_unbounded_des(energy_weight):
    calc = BatchSmdpCalc(24, energy_weight=energy_weight)
    calc.set_sources(1)
    calc.set_servers([0.4, 0.6], [1, 1.3])
    analytic = calc.run()
    assert analytic.tail_extension_stable
    values = []
    for seed in range(PARAMS["replications"]):
        sim = BatchSmdpSim(seed=9200 + seed, energy_weight=energy_weight)
        sim.set_sources(1)
        sim.set_servers([0.4, 0.6], [1, 1.3])
        result = sim.run(PARAMS["horizon"], analytic.policy, warmup=PARAMS["warmup"])
        assert_accounting(result)
        values.append(result.average_cost)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    assert abs(np.mean(values) - analytic.average_cost) <= PARAMS["standard_errors"] * se + PARAMS["absolute_floor"]


@pytest.mark.slow
def test_long_idle_extension_without_drain():
    sim = BatchSmdpSim(seed=9300)
    sim.set_sources(1)
    sim.set_servers([0.4], [1])
    result = sim.run(100000, [0, 0, 0], warmup=1000)
    assert result.completed == 0 and result.unfinished > 99000
    assert result.response_proxy > 49000
    assert_accounting(result)
