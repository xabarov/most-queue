"""Offline EPIC-056 protocol checks on synthetic source records only."""

import json
from dataclasses import replace

import numpy as np
import pytest

from examples import real_trace_temporal_experiment as module
from most_queue.sim.utils.workload_trace import SwfJob, SwfTrace


def fixture():
    jobs = tuple(SwfJob(i + 1, float(i * 5), 0, float(1 + i % 9), 1 + i % 4) for i in range(400))
    return SwfTrace(jobs, 4, {"synthetic": True}, (0, jobs[-1].submit))


def config(**kwargs):
    defaults = {"fractions": (0.4, 0.7), "jobs": 20, "warmup": 5, "replications": 2, "history_limit": 80, "minimum": 5}
    return module.TemporalConfig(**(defaults | kwargs))


def test_origins_strict_completion_boundary_and_recent_order():
    source, settings = fixture(), config()
    origins = module.prepare_origins(source, settings)
    for history, recent, test, audit in origins:
        assert all(job.completed_at < audit["cutoff"] for job in history)
        assert all(job.submit >= audit["cutoff"] for job in test)
        assert recent == history[-80:]
        assert len(test) == 25
    assert origins[0][2][-1].submit < origins[1][2][0].submit
    assert set(origins[0][2]) <= set(origins[1][0])


def test_no_test_service_leak_and_future_input_changes_do_not_move_cutoff():
    source, settings = fixture(), config(fractions=(0.4,))
    original = module.prepare_origins(source, settings)[0]
    cutoff = original[3]["cutoff"]
    changed = replace(
        source, jobs=tuple(replace(j, runtime=j.runtime * 1000) if j.submit >= cutoff else j for j in source.jobs)
    )
    other = module.prepare_origins(changed, settings)[0]
    assert original[:2] == other[:2]
    assert original[3] == other[3]
    first_models, first_ranks = module.fit_context(*original[:2], settings)
    second_models, second_ranks = module.fit_context(*other[:2], settings)
    assert first_models == second_models
    np.testing.assert_array_equal(first_ranks, second_ranks)


def test_mean_only_rescaling_is_training_only():
    history = tuple(SwfJob(i + 1, i, 0, 1 if i < 80 else 10, 1) for i in range(160))
    models, ranks = module.fit_context(history, history[-80:], config())
    uniforms = np.array([0.1, 0.4, 0.7, 0.9])
    samples = module.service_variants([1] * 4, models, ranks, uniforms)
    np.testing.assert_allclose(samples["recent_mean_only"], samples["expanding_coarse"] * 10 / 5.5, rtol=1e-14)
    np.testing.assert_array_equal(samples["recent_coarse"], [10] * 4)
    for variant in ("recent_rank_iid", "recent_block20", "recent_block60"):
        np.testing.assert_array_equal(samples[variant], [10] * 4)


def test_full_origin_determinism_shared_inputs_and_summary_audit():
    source, settings = fixture(), config(fractions=(0.4,))
    prepared = module.prepare_origins(source, settings)[0]
    result = module.run_origin(source, prepared, settings)
    assert result == module.run_origin(source, prepared, settings)
    json.dumps(result, allow_nan=False)
    assert result["scheduler_runs"] == (1 + 7 * 2) * 6
    assert sum(result["cohort"]["exact_fallback_counts"].values()) == settings.jobs
    for variant in ("observed", *module.VARIANTS):
        seeds = [None] if variant == "observed" else [56000, 56001]
        for seed in seeds:
            rows = [r for r in result["rows"] if (r["variant"], r["seed"]) == (variant, seed)]
            assert len(rows) == 6
            assert len({r["trace_sha256"] for r in rows}) == 1
            assert all(sum(g["count"] for g in r["groups"]) == settings.jobs for r in rows)
    for summary in result["summaries"]:
        rows = [r for r in result["rows"] if (r["variant"], r["policy"]) == (summary["variant"], summary["policy"])]
        assert summary["metrics"]["mean_t"]["mean"] == np.mean([r["mean_t"] for r in rows])
    for contrast in result["contrasts"]:
        policy = contrast["policy"]
        ref = next(r["mean_t"] for r in result["rows"] if r["variant"] == "observed" and r["policy"] == policy)
        values = [r["mean_t"] for r in result["rows"] if r["variant"] == contrast["variant"] and r["policy"] == policy]
        baseline = [
            r["mean_t"] for r in result["rows"] if r["variant"] == contrast["baseline"] and r["policy"] == policy
        ]
        expected = np.mean(np.abs(np.array(values) / ref - 1) - np.abs(np.array(baseline) / ref - 1))
        assert contrast["absolute_relative_error_change"]["mean_t"]["mean"] == expected
    assert len(module.aggregate([result])) == 7


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fractions": ()},
        {"fractions": (0.5, 0.4)},
        {"fractions": (0.5, 0.5)},
        {"fractions": (0,)},
        {"fractions": (float("nan"),)},
        {"jobs": True},
        {"warmup": -1},
        {"replications": 1},
        {"history_limit": 59},
        {"minimum": 1},
        {"first_seed": -1},
    ],
)
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


@pytest.mark.parametrize(
    "settings,match",
    [
        (config(fractions=(0.4, 0.401)), "overlap"),
        (config(fractions=(0.99,)), "not enough"),
        (config(fractions=(0.05,)), "too short"),
    ],
)
def test_unavailable_windows_fail_before_any_replay(settings, match):
    with pytest.raises(ValueError, match=match):
        module.prepare_origins(fixture(), settings)


def test_code_fingerprints_are_present_and_content_addressed():
    hashes = module.implementation_hashes()
    assert len(hashes) == 10
    assert "most_queue/random/trace_resampling.py" in hashes
    assert all(len(value) == 64 for value in hashes.values())
