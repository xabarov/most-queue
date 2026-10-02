"""Audit eligibility, information boundaries and paired general-service grid."""

import json
from dataclasses import fields, replace

import numpy as np
import pytest

from examples import msj_packing_experiment as pilot
from examples.msj_age_runtime_experiment import array_digest, initial_forecasts
from examples.msj_backfilling_experiment import interval
from examples.msj_runtime_prediction_experiment import fingerprint


@pytest.mark.parametrize(
    "regime,expected_work,count", [("one_or_all", 2.08, 11), ("powers_of_two", 3.28, 9), ("general", 3.94, 8)]
)
def test_declared_domains_and_load_law(regime, expected_work, count):
    """The prespecified eligibility and theoretical resource load are explicit."""
    assert pilot.WORKLOADS[regime].mean_work == pytest.approx(expected_work)
    policies = pilot.policies_for(regime)
    assert len(policies) == count
    assert ("server_filling" in policies) == (regime != "general")
    assert ("msfq:1" in policies) == (regime == "one_or_all")


@pytest.mark.parametrize("regime", pilot.WORKLOADS)
def test_independent_history_and_forecasts_have_no_test_labels(regime):
    """Changing future runtimes cannot affect historical estimates."""
    history, cohort, arrivals = pilot.make_bundle(regime, "erlang2", 110, 52000)
    repeated, repeated_cohort, repeated_arrivals = pilot.make_bundle(regime, "erlang2", 110, 52000)
    changed, _, _ = pilot.make_bundle(regime, "erlang2", 120, 52000)
    assert {field.name for field in fields(history)} == {"classes", "observed_times", "completed"}
    for field in fields(history):
        assert np.array_equal(getattr(history, field.name), getattr(repeated, field.name))
        assert np.array_equal(getattr(history, field.name), getattr(changed, field.name))
    assert fingerprint(cohort) == fingerprint(repeated_cohort)
    assert np.array_equal(arrivals, repeated_arrivals)
    models = {"km": pilot.fit_history(history, len(pilot.WORKLOADS[regime].needs))}
    actual = initial_forecasts(models, cohort.classes)["km"]
    altered = replace(cohort, services=cohort.services * 100)
    assert np.array_equal(actual, initial_forecasts(models, altered.classes)["km"])


@pytest.mark.parametrize("regime", pilot.WORKLOADS)
def test_small_grid_preserves_trace_hashes_and_paired_intervals(regime, monkeypatch):
    """Reconstruct paired intervals and check identical input work for all modes."""
    original = pilot.replay
    snapshots = {}

    def audit(workload, trace, policy, warmup, models):
        key = (trace[0].arrival, trace[-1].arrival)
        digest = array_digest(np.array([(job.arrival, job.cls, job.service, job.estimate) for job in trace]))
        if key in snapshots:
            assert snapshots[key] == digest
        snapshots[key] = digest
        return original(workload, trace, policy, warmup, models)

    monkeypatch.setattr(pilot, "replay", audit)
    result = pilot.experiment(regime, "erlang2", 100, 2)
    assert len(result["replications"]) == 4 * len(pilot.policies_for(regime))
    assert len(result["history_runs"]) == 2
    assert len(snapshots) == 4
    for row in result["replications"]:
        assert row["measured_jobs"] == 100 and row["drained_jobs"] == 110
        assert row["arrival_rate"] == pytest.approx(row["load"] * 4 / pilot.WORKLOADS[regime].mean_work)
        assert row["preemption_capability"] == (row["policy"] == "server_filling")
        if not row["policy"].startswith(("easy:", "conservative:")):
            assert row["promise_violation_rate"] is None
            assert row["reservations"] == row["reservation_violations"] == 0
        if row["policy"] != "server_filling":
            assert row["preemptions"] == row["preempted_jobs"] == 0
        if not row["policy"].endswith("_age"):
            assert row["runtime_updates"] == 0
        weights = (
            np.array(pilot.WORKLOADS[regime].probabilities)
            * pilot.WORKLOADS[regime].needs
            * pilot.WORKLOADS[regime].means
        )
        expected = (
            sum(weight * row[f"mean_t_k{need}"] for weight, need in zip(weights, pilot.WORKLOADS[regime].needs))
            / weights.sum()
        )
        assert row["weighted_mean_t"] == pytest.approx(expected)
    for summary in result["summaries"] + result["msf_contrasts"]:
        samples = [
            row
            for row in result["replications"]
            if row["load"] == summary["load"] and row["policy"] == summary["policy"]
        ]
        baseline = {
            row["seed"]: row
            for row in result["replications"]
            if row["load"] == summary["load"] and row["policy"] == summary["baseline"]
        }
        expected = interval([row["mean_t"] - baseline[row["seed"]]["mean_t"] for row in samples])
        assert summary["paired_delta"]["mean_t"] == pytest.approx(expected)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "regime,shape,jobs,reps",
    [
        ("bad", "erlang2", 100, 2),
        ("general", "bad", 100, 2),
        ("general", "erlang2", 99, 2),
        ("general", "erlang2", 100, 1),
    ],
)
def test_invalid_protocol_inputs(regime, shape, jobs, reps):
    """Refuse invalid grids before generating data."""
    with pytest.raises(ValueError):
        pilot.experiment(regime, shape, jobs, reps)
