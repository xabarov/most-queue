"""Offline protocol tests with synthetic SWF, never a network dependency."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from examples import real_trace_calibration_experiment as module
from most_queue.sim.utils.workload_trace import SwfJob, SwfTrace, chronological_split


def source_fixture():
    jobs = tuple(SwfJob(idx + 1, float(idx * 5), 0, float(1 + idx % 9), 1 + idx % 4) for idx in range(120))
    return SwfTrace(jobs, 4, {"fixture": "synthetic"}, (0, jobs[-1].submit))


def test_end_to_end_determinism_pairing_and_strict_json():
    source = source_fixture()
    config = module.ExperimentConfig(jobs=10, warmup=2, blocks=2, replications=2)
    first = module.experiment(source, config)
    assert first == module.experiment(source, config)
    assert first["scheduler_runs"] == 2 * (1 + 4 * 2) * len(module.POLICIES)
    json.dumps(first, allow_nan=False)
    for block in range(2):
        for family in ("observed", *module.FAMILIES):
            seeds = [None] if family == "observed" else [55000, 55001]
            for seed in seeds:
                rows = [r for r in first["rows"] if (r["block"], r["family"], r["seed"]) == (block, family, seed)]
                assert len(rows) == len(module.POLICIES)
                assert len({r["trace_sha256"] for r in rows}) == 1
    assert all(row["reference_regret"] >= 0 for row in first["decisions"])


def test_hidden_test_service_never_changes_fit_or_forecast():
    source = source_fixture()
    training, heldout, _ = chronological_split(source)
    models, fits = module.fit_history(training)
    altered = replace(source, jobs=tuple(replace(j, runtime=10000) if j.submit >= 357 else j for j in source.jobs))
    changed_models, changed_fits = module.fit_history(chronological_split(altered)[0])
    assert fits == changed_fits
    _, trace = module.make_trace(heldout[:10], [j.runtime for j in heldout[:10]], models)
    _, changed = module.make_trace(heldout[:10], [10000] * 10, changed_models)
    assert [j.estimate for j in trace] == [j.estimate for j in changed]
    assert [j.arrival for j in trace] == [j.arrival for j in changed]
    assert module.fit_history(training)[1][3]["pooled_fallback"]


def test_single_resource_matches_independent_lindley_recursion():
    jobs = tuple(SwfJob(i + 1, float(i), 100, service, 1) for i, service in enumerate([3, 1, 2, 4, 1]))
    models, _ = module.fit_history(jobs)
    needs, trace = module.make_trace(jobs, [j.runtime for j in jobs], models)
    finish, times = 0, []
    for job in jobs:
        finish = max(finish, job.submit) + job.runtime
        times.append(finish - job.submit)
    for policy in module.POLICIES:
        result = module.replay(1, needs, trace, policy, 1)
        assert result["mean_t"] == pytest.approx(np.mean(times[1:]))
        assert result["p99_t"] == pytest.approx(np.quantile(times[1:], 0.99))
        assert result["groups"][1]["mean_t"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"jobs": 0},
        {"warmup": -1},
        {"blocks": True},
        {"replications": 1},
        {"first_seed": -1},
        {"jobs": 2.5},
        {"train_fraction": 1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        module.ExperimentConfig(**kwargs)


def test_insufficient_test_data_is_not_silently_shortened():
    with pytest.raises(ValueError, match="not enough"):
        module.experiment(source_fixture())


def test_common_interval_recomputed_independently():
    result = module.experiment(source_fixture(), module.ExperimentConfig(jobs=10, warmup=2, blocks=1, replications=3))
    for summary in result["summaries"]:
        if summary["policy"] == "fcfs":
            assert summary["metrics"]["mean_t"]["difference_from_fcfs"] == {"mean": 0, "low": 0, "high": 0}
        rows = [
            row for row in result["rows"] if row["family"] == summary["family"] and row["policy"] == summary["policy"]
        ]
        assert summary["metrics"]["mean_t"]["mean"] == pytest.approx(np.mean([row["mean_t"] for row in rows]))


def test_cache_is_verified_and_never_silently_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "source.swf"
    with pytest.raises(ValueError, match="not cached"):
        module.verified_source(path)
    path.write_bytes(b"not a trace")
    with pytest.raises(ValueError, match="SHA-256"):
        module.verified_source(path, download=True)
    assert path.read_bytes() == b"not a trace"
    monkeypatch.setattr(module, "SOURCE_SHA256", hashlib.sha256(b"not a trace").hexdigest())
    assert module.verified_source(path) == b"not a trace"
