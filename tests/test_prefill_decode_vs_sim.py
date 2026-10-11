"""DES vs independent M/D/1 arbiter; approximation error is NOT numerical error."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import PrefillDecodeCalc

PARAMS = yaml.safe_load((Path(__file__).parent / "default_params.yaml").read_text())["prefill_decode"]


@pytest.mark.parametrize("rate", [0.2, 0.5, 0.7])
def test_single_slot_deterministic_des_vs_pk(rate):
    costs = (0.1, 0.1, 0)
    n, m = 1, 4
    # Five deterministic iterations: S=1. No occupancy-dependent cost at B=1.
    waiting = rate / (2 * (1 - rate))
    samples = []
    for seed in range(PARAMS["replications"]):
        sim = PrefillDecodeSim(batch_cap=1, token_budget=None, seed=1200 + seed)
        sim.set_sources(rate)
        sim.set_servers(n, m, *costs)
        r = sim.run(PARAMS["jobs_per_replication"])
        samples.append([r.w[0], r.v[0], r.ttft_mean, r.itl_mean])
    means = np.mean(samples, axis=0)
    se = np.std(samples, axis=0, ddof=1) / np.sqrt(len(samples))
    target = np.array([waiting, waiting + 1, waiting + 0.4, 0.2])
    assert np.all(np.abs(means - target) <= PARAMS["standard_errors"] * se + PARAMS["absolute_floor"])
    calc = PrefillDecodeCalc(batch_cap=1, token_budget=None)
    calc.set_sources(rate)
    calc.set_servers(n, m, *costs)
    assert calc.run().w[0] == pytest.approx(2 * waiting)


@pytest.mark.slow
def test_longer_single_slot_deterministic_sojourn_and_time_little():
    sim = PrefillDecodeSim(batch_cap=1, token_budget=None, seed=1357)
    sim.set_sources(0.7)
    sim.set_servers(1, 4, 0.1, 0.1, 0)
    result = sim.run(150_000)
    assert result.w[0] == pytest.approx(0.7 / (2 * 0.3), rel=0.06)
    assert result.batch_mean == pytest.approx(0.7, abs=0.025)
    assert result.jobs_mean == pytest.approx(0.7 * result.v[0], rel=0.04)
