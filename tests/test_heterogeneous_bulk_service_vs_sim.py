"""Replication-based independent DES check of means, class flows and W/V tails."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.sim.heterogeneous_bulk_service import BulkServiceHeterogeneousSim
from most_queue.theory.batch import BulkServiceHeterogeneousCalc


@pytest.mark.slow
@pytest.mark.parametrize("thresholds", [(1, 1), (2, 3), (4, 2)])
def test_heterogeneous_bulk_service_vs_sim(thresholds):
    cfg = yaml.safe_load((Path(__file__).parent / "default_params.yaml").read_text())["heterogeneous_bulk_service"]
    counts, rates, capacities, l = [2, 1], [1, 0.6], [4, 5], 2.5
    calc = BulkServiceHeterogeneousCalc(counts)
    calc.set_sources(l)
    calc.set_servers(rates, thresholds, capacities)
    r = calc.run()
    times = [0, 1, 3]
    expected = np.array(
        [
            r.w[0],
            r.v[0],
            *r.customer_throughputs,
            *(calc.tail(t) for t in times),
            *(calc.tail(t, response=True) for t in times),
        ]
    )
    observations = []
    for seed in range(cfg["replications"]):
        sim = BulkServiceHeterogeneousSim(counts, seed=8800 + seed)
        sim.set_sources(l)
        sim.set_servers(rates, thresholds, capacities)
        s = sim.run(cfg["horizon"], warmup=cfg["warmup"], tail_times=times)
        assert s.arrived == s.completed + s.unfinished
        assert s.total_queue_area == pytest.approx(s.started_wait_exposure + s.unfinished_wait_exposure)
        assert s.total_population_area == pytest.approx(s.completed_residence + s.unfinished_residence)
        observations.append(
            [s.waiting_proxy, s.response_proxy, *s.customer_throughputs, *s.waiting_tails, *s.response_tails]
        )
    samples = np.array(observations)
    allowance = cfg["standard_errors"] * samples.std(axis=0, ddof=1) / np.sqrt(len(samples)) + cfg["absolute_floor"]
    assert np.all(np.abs(samples.mean(axis=0) - expected) <= allowance)
