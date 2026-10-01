"""Protocol checks for law-based load matching and paired comparisons."""

from dataclasses import replace

import numpy as np
import pytest

from examples.msj_conservative_experiment import controlled_workload, evaluate, expected_work, paired_intervals


@pytest.mark.parametrize("coupling", ["coupled", "shuffled"])
@pytest.mark.parametrize("geometry", ["two_class", "three_class"])
def test_load_set_from_law_not_realized_services(coupling, geometry):
    traces = []
    for seed in (48, 49):
        trace, rate = controlled_workload("erlang2", coupling, 100, seed, 0.55, geometry)
        assert rate * expected_work(coupling, 100, geometry) / 4 == pytest.approx(0.55)
        traces.append(trace)
    assert traces[0] != traces[1]


def test_finite_permutation_expectation_and_class_marginals():
    assert expected_work("coupled", 100) == pytest.approx(2.57)
    assert expected_work("shuffled", 100) == pytest.approx(1.751 + 0.819 / 100)
    assert expected_work("shuffled", 1) == expected_work("coupled", 1)
    first, rate1 = controlled_workload("lognormal_cv1.5", "coupled", 200, 48, 0.55)
    second, rate2 = controlled_workload("lognormal_cv1.5", "shuffled", 200, 48, 0.55)
    assert [job.service for job in first] == [job.service for job in second]
    assert sorted(job.cls for job in first) == sorted(job.cls for job in second)
    assert np.allclose([job.arrival * rate1 for job in first], [job.arrival * rate2 for job in second])
    assert rate1 != rate2


def test_three_class_work_and_marginals():
    assert expected_work("coupled", 200, "three_class") == pytest.approx(3.94)
    assert expected_work("shuffled", 200, "three_class") == pytest.approx(2.86 + 1.08 / 200)
    first, _ = controlled_workload("erlang2", "coupled", 200, 48, 0.55, "three_class")
    second, _ = controlled_workload("erlang2", "shuffled", 200, 48, 0.55, "three_class")
    assert [job.service for job in first] == [job.service for job in second]
    assert sorted(job.cls for job in first) == sorted(job.cls for job in second)
    assert set(job.cls for job in first) == {0, 1, 2}


def test_controlled_workload_rescaling_does_not_change_jobs():
    first, rate1 = controlled_workload("erlang2", "coupled", 100, 48, 0.35)
    second, rate2 = controlled_workload("erlang2", "coupled", 100, 48, 0.55)
    assert [replace(job, arrival=0) for job in first] == [replace(job, arrival=0) for job in second]
    assert np.allclose([job.arrival * rate1 for job in first], [job.arrival * rate2 for job in second])


def test_pairing_uses_per_seed_differences():
    samples = [dict.fromkeys(("mean_t", "p99_t", "p99_wide"), x) for x in (100, 200, 300)]
    baseline = [{key: value - 2 for key, value in sample.items()} for sample in samples]
    assert paired_intervals(samples, baseline)["mean_t"] == {"mean": 2, "low": 2, "high": 2}


@pytest.mark.parametrize("policy", ["easy_class_mean", "conservative_class_mean"])
def test_class_prediction_does_not_rewrite_input(policy):
    trace, _ = controlled_workload("erlang2", "coupled", 100, 48, 0.35)
    before = tuple(trace)
    result = evaluate(trace, policy, "coupled", 10)
    assert trace == before
    assert result["mean_t"] >= 0
    assert all(job.estimate == job.service for job in trace)


def test_protocol_validation():
    for coupling, count in (("unknown", 100), ("coupled", 0)):
        with pytest.raises(ValueError):
            expected_work(coupling, count)
    for shape, load in (("unknown", 0.5), ("erlang2", 0), ("erlang2", np.nan)):
        with pytest.raises(ValueError):
            controlled_workload(shape, "coupled", 100, 48, load)
