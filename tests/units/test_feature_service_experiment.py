"""Synthetic temporal selection and late replay checks without downloading data."""

import json
from dataclasses import replace

import numpy as np
import pytest

from examples import feature_service_experiment as module
from tests.units.test_modern_gpu_trace_experiment import setup as gpu_setup
from tests.units.test_real_trace_lifecycle_experiment import source_fixture


def setup(name):
    source = source_fixture() if name == "sdsc" else gpu_setup()[0]
    config = module.FeatureConfig(0.3, 0.7, validation_jobs=20, jobs=20, warmup=5, replications=2)
    return source, config, module.prepare(source, name, config)


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_splits_and_selection_do_not_read_test_or_future_service(name):
    _, config, prepared = setup(name)
    assert all(j.completed_at < prepared["validation_split"]["cutoff"] for j in prepared["early"])
    assert all(j.completed_at < prepared["test_split"]["cutoff"] for j in prepared["late"])
    assert not set(j.job_id for j in prepared["test"]) & set(j.job_id for j in prepared["late"])
    restricted = {k: prepared[k] for k in ("early", "validation", "validation_split")}
    assert module.select(prepared, name, config) == module.select(restricted, name, config)
    altered = dict(prepared, test=(object(),), late=(object(),), expanding=(object(),))
    assert module.select(prepared, name, config) == module.select(altered, name, config)


def test_predict_does_not_access_service():
    _, config, prepared = setup("sdsc")
    models = module.fit_models(prepared["early"], "sdsc", config.minimum)
    target = prepared["test"][0]
    changed = replace(target, runtime=1e100)
    for model in models.values():
        assert module.predict(model, target, "sdsc") == module.predict(model, changed, "sdsc")


def test_unfinished_validation_is_rejected_not_filtered_out():
    source, config, prepared = setup("sdsc")
    job = prepared["validation"][0]
    changed = replace(
        source, jobs=tuple(replace(j, runtime=10000) if j.job_id == job.job_id else j for j in source.jobs)
    )
    with pytest.raises(ValueError, match="all validation outcomes"):
        module.prepare(changed, "sdsc", config)


def test_insufficient_cohorts_fail_instead_of_changing_sample_size():
    source, config, _ = setup("sdsc")
    with pytest.raises(ValueError, match="full cohorts"):
        module.prepare(source, "sdsc", replace(config, jobs=1000))


def test_exact_validation_score_ties_use_prespecified_order(monkeypatch):
    _, config, prepared = setup("sdsc")
    monkeypatch.setattr(module, "score", lambda *args: {"crps": 1})
    assert module.select(prepared, "sdsc", config)["selected"] == "coarse"


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_replay_determinism_accounting_selection_and_independent_summaries(name):
    source, config, prepared = setup(name)
    selection = module.select(prepared, name, config)
    result = module.run_test(source, prepared, name, config, selection)
    assert result == module.run_test(source, prepared, name, config, selection)
    assert result["selected"] == selection["selected"]
    json.dumps(result, allow_nan=False)
    assert result["scheduler_runs"] == len(module.SCENARIOS[name]) * 7 * 6
    for row in result["rows"]:
        assert row["target_count"] == 20
        assert row["mean_release_t"] - row["mean_w"] == pytest.approx(row["target_consumed_service_mean"])
        assert row["total_resource_time"] == sum(row["resource_time_by_outcome"].values())
        assert row["success_rate"] + row["timeout_rate"] == 1
    for summary in result["summaries"]:
        rows = [r for r in result["rows"] if all(r[k] == summary[k] for k in ("scenario", "variant", "policy"))]
        for metric in module.METRICS:
            assert summary["metrics"][metric]["mean"] == np.mean([r[metric] for r in rows])
    for contrast in result["contrasts"]:
        rows = [r for r in result["rows"] if all(r[k] == contrast[k] for k in ("scenario", "variant", "policy"))]
        base = [
            r
            for r in result["rows"]
            if r["variant"] == "coarse" and all(r[k] == contrast[k] for k in ("scenario", "policy"))
        ]
        reference = next(
            r
            for r in result["rows"]
            if r["variant"] == "observed" and all(r[k] == contrast[k] for k in ("scenario", "policy"))
        )
        for metric, interval in contrast["absolute_error_change"].items():
            expected = [
                abs(r[metric] - reference[metric]) - abs(b[metric] - reference[metric]) for r, b in zip(rows, base)
            ]
            assert interval["mean"] == np.mean(expected)
    for decision in result["decisions"]:
        assert len(decision["reference_best"]) == decision["reference_tie_count"]
        assert decision["relative_regret"] >= -1e-12


def test_uniform_tape_is_paired_and_source_cutoff_seed_specific():
    values = module.uniform_tape("sdsc", 100, 59000, 10)
    np.testing.assert_array_equal(values, module.uniform_tape("sdsc", 100, 59000, 10))
    assert np.all((values > 0) & (values < 1))
    for name, cutoff, seed in [("kalos", 100, 59000), ("sdsc", 101, 59000), ("sdsc", 100, 59001)]:
        assert not np.array_equal(values, module.uniform_tape(name, cutoff, seed, 10))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_fraction": 0.9},
        {"test_fraction": 1},
        {"warmup": True},
        {"jobs": 0},
        {"minimum": 1},
        {"replications": 1},
        {"history_limit": 1},
    ],
)
def test_config_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        module.FeatureConfig(**kwargs)
