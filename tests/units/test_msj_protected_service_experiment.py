"""Independent tuning/test boundaries, complete candidate grids and paired CIs."""

import json
from dataclasses import replace

import numpy as np
import pytest

from examples import msj_protected_service_experiment as pilot
from examples.msj_age_runtime_experiment import array_digest
from examples.msj_backfilling_experiment import interval


@pytest.mark.parametrize("regime", pilot.REGIMES)
@pytest.mark.parametrize("shape", pilot.SHAPES)
@pytest.mark.parametrize("cost", pilot.COST_LEVELS)
def test_small_registered_grid_has_independent_tuning_and_common_test_work(regime, shape, cost, monkeypatch):
    """Audit actual replay inputs, fixed theoretical load and exact tuned aliases."""
    original, snapshots, calls = pilot.replay, {}, []

    def audited(workload, trace, policy, settings):
        key = trace[0].arrival, trace[-1].arrival
        digest = array_digest(np.array([(j.arrival, j.cls, j.service, j.estimate) for j in trace]))
        if key in snapshots:
            assert snapshots[key] == digest
        snapshots[key] = digest
        calls.append(policy)
        return original(workload, trace, policy, settings)

    monkeypatch.setattr(pilot, "replay", audited)
    result = pilot.experiment(regime, shape, cost, size=pilot.StudySize(100, 2, 100, 2))
    assert len(calls) == result["protocol"]["scheduler_runs"] == 48
    assert len(snapshots) == 8
    assert [r["seed"] for r in result["tuning_history"]] == [54000, 54001]
    assert [r["seed"] for r in result["history_runs"]] == [54100, 54101]
    assert len(result["tuning_runs"]) == 16 and len(result["replications"]) == 36
    assert {r["test_hash"] for r in result["tuning_history"]}.isdisjoint(
        {r["test_hash"] for r in result["history_runs"]}
    )
    assert result["selection"] == pilot.select_policy(result["tuning_runs"])
    for row in result["replications"]:
        assert row["measured_jobs"] == 100 and row["drained_jobs"] == 110
        assert row["arrival_rate"] == pytest.approx(row["load"] * 4 / pilot.WORKLOADS[regime].mean_work)
        assert row["mean_w"] == pytest.approx(row["mean_first_wait"] + row["mean_interruption"])
        assert row["mean_w"] == pytest.approx(row["mean_queue_wait"] + row["mean_checkpoint"] + row["mean_resume"])
        assert row["utilization"] == pytest.approx(
            row["productive_utilization"] + row["checkpoint_utilization"] + row["resume_utilization"]
        )
        if row["factor"] is not None:
            assert row["min_service_time"] == pytest.approx(row["factor"] * cost * result["protocol"]["mean_service"])
            assert row["checkpoint_time"] == row["resume_time"]
        if row["derived"]:
            chosen = next(
                r
                for r in result["replications"]
                if r["seed"] == row["seed"] and r["load"] == row["load"] and r["policy"] == row["selected_policy"]
            )
            assert row == {**chosen, "derived": True, "policy": "tuned", "selected_policy": chosen["policy"]}
        else:
            assert row["policy"] in pilot.POLICIES
    for key in ("unprotected_contrasts", "first_fit_contrasts", "msf_contrasts"):
        for summary in result[key]:
            rows = [r for r in result["replications"] if r["load"] == summary["load"]]
            refs = {r["seed"]: r for r in rows if r["policy"] == summary["baseline"]}
            samples = [r for r in rows if r["policy"] == summary["policy"]]
            for metric in ("weighted_mean_t", "preemptions", "protected_preemptions"):
                expected = interval([r[metric] - refs[r["seed"]][metric] for r in samples])
                assert summary["paired_delta"][metric] == expected
    json.dumps(result, allow_nan=False)


def test_test_outcomes_cannot_change_selection_and_tuning_precedes_test(monkeypatch):
    """Change all test service labels; the historical choice must stay frozen."""
    sizes = pilot.StudySize(100, 2, 100, 2)
    expected = pilot.experiment(size=sizes)
    original, original_select, seen = pilot.make_bundle, pilot.select_policy, []

    def changed(regime, shape, count, seed):
        seen.append(seed)
        history, cohort, arrivals = original(regime, shape, count, seed)
        if seed >= 54100:
            cohort = replace(cohort, services=cohort.services * 3)
        return history, cohort, arrivals

    def select_before_test(rows):
        assert max(seen) < 54100
        assert {r["seed"] for r in rows} == {54000, 54001}
        return original_select(rows)

    monkeypatch.setattr(pilot, "make_bundle", changed)
    monkeypatch.setattr(pilot, "select_policy", select_before_test)
    actual = pilot.experiment(size=sizes)
    assert actual["selection"] == expected["selection"]
    assert actual["tuning_runs"] == expected["tuning_runs"]
    assert actual["history_runs"][0]["test_hash"] != expected["history_runs"][0]["test_hash"]


def test_selection_ties_choose_the_smallest_factor_and_require_paired_finite_data():
    """No lucky seed, missing candidate or nonfinite objective is accepted."""
    rows = [
        {"seed": seed, "load": load, "policy": policy, "weighted_mean_t": 2.0}
        for load in pilot.LOADS
        for policy in pilot.CANDIDATES
        for seed in (54000, 54001)
    ]
    assert all(choice["factor"] == 0 for choice in pilot.select_policy(rows))
    for bad in (rows[1:], rows + [rows[0]], [{**rows[0], "weighted_mean_t": float("nan")}, *rows[1:]]):
        with pytest.raises(ValueError):
            pilot.select_policy(bad)


@pytest.mark.parametrize(
    "values",
    [
        {"jobs": 99},
        {"replications": 1},
        {"tuning_jobs": 99},
        {"tuning_replications": 1},
        {"tuning_replications": 101},
        {"jobs": True},
        {"tuning_jobs": 100.0},
    ],
)
def test_invalid_sizes_or_overlapping_seed_ranges_are_rejected(values):
    """The tuning seed range ends before the first held-out seed."""
    with pytest.raises(ValueError):
        pilot.StudySize(**values)


@pytest.mark.parametrize(
    "values", [{"regime": "general"}, {"shape": "bad"}, {"cost_factor": -1}, {"cost_factor": True}]
)
def test_invalid_scenario_is_rejected(values):
    """Resource and overhead grids are declared before evaluation."""
    with pytest.raises(ValueError):
        pilot.experiment(**values)
