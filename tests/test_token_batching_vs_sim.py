"""Replications are the statistical unit, not correlated within-run requests."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.sim.token_batching import TokenBatchingSim
from most_queue.theory.batch.token_batching import FixedTokenBatchingCalc, TokenBatchingCalc
from most_queue.theory.batch.token_service import TokenLengthDistribution

PARAMS = yaml.safe_load((Path(__file__).parent / "default_params.yaml").read_text())["token_batching"]


@pytest.mark.parametrize("strategy", ["dynamic", "fixed", "elastic"])
@pytest.mark.parametrize("rate", [1.0, 2.0])
def test_three_strategies_against_individual_request_des(strategy, rate):
    _compare(strategy, rate, PARAMS["replications"], PARAMS["jobs_per_replication"])


@pytest.mark.slow
def test_high_load_elastic_against_independent_replications():
    _compare("elastic", 3.0, 10, 100_000)


def _compare(strategy, rate, replications, jobs):
    tokens = TokenLengthDistribution([1, 3, 9], [0.2, 0.6, 0.2])
    coeff = (0.1, 0.2, 0.025, 0.04)
    calc = FixedTokenBatchingCalc(3) if strategy == "fixed" else TokenBatchingCalc(tolerance=1e-8)
    expected = calc.set_sources(rate).set_servers(tokens, *coeff, elastic=strategy == "elastic").run()
    rows = []
    for seed in range(replications):
        result = (
            TokenBatchingSim(strategy, 3 if strategy == "fixed" else None, seed=100 + seed)
            .set_sources(rate)
            .set_servers(tokens, *coeff)
            .run(jobs)
        )
        rows.append([result.w[0], result.v[0], result.utilization])
        assert sum(result.p) == pytest.approx(1)
        # Finite observation windows need not obey Little's law exactly.
        assert abs(np.arange(len(result.p)) @ result.p - rate * result.v[0]) < 0.15 * max(1, rate * result.v[0])
    rows = np.asarray(rows)
    truth = np.array([expected.w[0], expected.v[0], expected.utilization])
    se = rows.std(axis=0, ddof=1) / np.sqrt(replications)
    tolerance = PARAMS["standard_errors"] * se + PARAMS["relative_floor"] * truth
    assert np.all(abs(rows.mean(axis=0) - truth) < tolerance), (rows.mean(axis=0), truth, se)
