"""Audit the EPIC-051 information split and paired experimental protocol."""

import json
from dataclasses import fields, replace

import numpy as np
import pytest

from examples import msj_age_runtime_experiment as pilot
from examples.msj_backfilling_experiment import interval
from examples.msj_runtime_prediction_experiment import fingerprint
from most_queue.sim.utils.residual_runtime import KaplanMeierRuntimeEstimator


def test_bundles_repeat_and_observed_history_does_not_retain_hidden_lifetimes():
    """History has only observable data and changing test length preserves it."""
    a, test_a, arrivals_a = pilot.make_bundle("erlang2", 110, 51000)
    b, test_b, arrivals_b = pilot.make_bundle("erlang2", 110, 51000)
    c, _, _ = pilot.make_bundle("erlang2", 120, 51000)
    assert {field.name for field in fields(a)} == {"classes", "observed_times", "completed"}
    for field in fields(a):
        assert np.array_equal(getattr(a, field.name), getattr(b, field.name))
        assert np.array_equal(getattr(a, field.name), getattr(c, field.name))
    assert fingerprint(test_a) == fingerprint(test_b)
    assert np.array_equal(arrivals_a, arrivals_b)
    assert 0 < np.sum(a.completed) < len(a.completed)
    models = pilot.fit_models(a)
    altered_test = replace(test_a, services=test_a.services * 1000)
    forecasts = [pilot.initial_forecasts(models, cohort.classes) for cohort in (test_a, altered_test)]
    for name in forecasts[0]:
        assert np.array_equal(forecasts[0][name], forecasts[1][name])


def test_unavailable_initial_forecast_aborts_instead_of_changing_the_cohort():
    """The declared comparison must not remove unsupported classes or jobs."""
    censored = KaplanMeierRuntimeEstimator().fit([1], [False])
    with pytest.raises(ValueError, match="initial class quantile unavailable"):
        pilot.initial_forecasts({"km": [censored] * 3}, np.array([0, 1, 2]))


def test_landmark_unavailability_and_empty_survivor_sets_remain_null():
    """Unavailable forecasts are not counted as successes in coverage."""
    _, cohort, _ = pilot.make_bundle("erlang2", 100, 51001)
    censored = KaplanMeierRuntimeEstimator().fit([1], [False])
    cohort = replace(cohort, services=np.full(100, 0.1))
    rows = pilot.landmark_metrics({"km": [censored] * 3}, cohort, 0)
    assert all(row["quantile"] is None and row["coverage"] is None for row in rows)
    assert all(row["survivors"] == 0 for row in rows if row["age_multiple"] > 0)
    summaries = pilot.landmark_summaries([{"seed": 1, **row} for row in rows])
    assert all(row["metrics"]["coverage"]["available_seeds"] == 0 for row in summaries)
    assert all(row["metrics"]["coverage"]["ci95"] is None for row in summaries)
    json.dumps(summaries, allow_nan=False)


def test_small_grid_preserves_work_initial_forecasts_and_seed_matched_deltas(monkeypatch):
    """Run real schedules and independently rebuild the paired summary."""
    original = pilot.replay
    snapshots = {}

    def audited_replay(cohort, arrivals, estimates, options):
        key = (
            fingerprint(cohort),
            options.load,
            options.discipline,
            None if estimates is None else pilot.array_digest(estimates),
        )
        if options.predictor is None:
            snapshots[key] = pilot.array_digest(arrivals)
        else:
            assert key in snapshots  # corresponding fixed mode was run first
            assert snapshots[key] == pilot.array_digest(arrivals)
        return original(cohort, arrivals, estimates, options)

    monkeypatch.setattr(pilot, "replay", audited_replay)
    result = pilot.experiment("erlang2", jobs=100, replications=2)
    assert len(result["replications"]) == 44
    assert len(result["history_runs"]) == 2
    assert len(result["landmark_runs"]) == 36
    for row in result["replications"]:
        assert row["measured_jobs"] == 100 and row["drained_jobs"] == 110
        assert row["arrival_rate"] == pytest.approx(row["load"] * 4 / 3.94)
        if not row["policy"].endswith("_age"):
            assert row["runtime_updates"] == row["unavailable_runtime_updates"] == 0
        if row["policy"].endswith("oracle"):
            assert row["reservation_violations"] == 0
    for load in pilot.LOADS:
        for seed in (51000, 51001):
            rows = [r for r in result["replications"] if r["load"] == load and r["seed"] == seed]
            assert len({r["sample_offered_load"] for r in rows}) == 1
    for row in result["summaries"]:
        samples = [r for r in result["replications"] if r["load"] == row["load"] and r["policy"] == row["policy"]]
        reference = {
            r["seed"]: r for r in result["replications"] if r["load"] == row["load"] and r["policy"] == row["baseline"]
        }
        expected = interval([r["mean_t"] - reference[r["seed"]]["mean_t"] for r in samples])
        assert row["paired_delta"]["mean_t"] == pytest.approx(expected)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("shape,jobs,reps", [("bad", 100, 2), ("erlang2", 99, 2), ("erlang2", 100, 1)])
def test_invalid_protocol_inputs(shape, jobs, reps):
    """Refuse unsupported grids before generating or scheduling work."""
    with pytest.raises(ValueError):
        pilot.experiment(shape, jobs, reps)
