"""Check pairing and interval mechanics of the reproducible experiment."""

import numpy as np
import pytest

from examples.msj_backfilling_experiment import interval, make_workload


@pytest.mark.parametrize("shape", ["erlang2", "exponential", "lognormal_cv1.5"])
def test_coupling_changes_pairing_not_marginals(shape):
    coupled = make_workload(shape, "coupled", 200, 47000)
    shuffled = make_workload(shape, "shuffled", 200, 47000)
    assert [job.service for job in coupled] == [job.service for job in shuffled]
    assert [job.arrival for job in coupled] == [job.arrival for job in shuffled]
    assert sorted(job.cls for job in coupled) == sorted(job.cls for job in shuffled)
    assert [job.cls for job in coupled] != [job.cls for job in shuffled]


def test_independent_replication_interval():
    result = interval([1, 2, 3, 4, 5, 6])
    assert result["mean"] == pytest.approx(3.5)
    assert np.isfinite(result["low"]) and result["low"] < result["mean"] < result["high"]
    assert interval([0, 0, 0]) == {"mean": 0, "low": 0, "high": 0}
