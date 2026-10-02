"""Checkpoint experiment: fixed useful work/load and independently paired CIs."""

import json

import numpy as np
import pytest

from examples import msj_checkpoint_experiment as pilot
from examples.msj_age_runtime_experiment import array_digest
from examples.msj_backfilling_experiment import interval


def test_prespecified_cost_grid_uses_theoretical_mean():
    """No cost uses realized S; asymmetric splits preserve total declared cost."""
    assert len(pilot.POLICIES) == 13
    for regime, expected in (("one_or_all", 0.76), ("powers_of_two", 1.3)):
        workload = pilot.WORKLOADS[regime]
        assert np.dot(workload.probabilities, workload.means) == pytest.approx(expected)
    assert pilot.COSTS["sf:checkpoint:0.2"] == (0.2, 0)
    assert pilot.COSTS["sf:resume:0.2"] == (0, 0.2)
    assert pilot.COSTS["sf:balanced:0.2"] == (0.1, 0.1)


@pytest.mark.parametrize("regime", pilot.REGIMES)
@pytest.mark.parametrize("shape", pilot.SHAPES)
def test_small_cost_grid_common_work_metrics_and_pairing(regime, shape, monkeypatch):
    """Run actual schedules, audit identical inputs and reconstruct paired CIs."""
    original = pilot.replay
    snapshots = {}

    def audited(workload, trace, policy, warmup, models):
        key = (trace[0].arrival, trace[-1].arrival)
        digest = array_digest(np.array([(job.arrival, job.cls, job.service, job.estimate) for job in trace]))
        if key in snapshots:
            assert snapshots[key] == digest
        snapshots[key] = digest
        return original(workload, trace, policy, warmup, models)

    monkeypatch.setattr(pilot, "replay", audited)
    result = pilot.experiment(regime, shape, 100, 2)
    assert len(result["replications"]) == 52
    assert len(snapshots) == 4
    assert [r["seed"] for r in result["history_runs"]] == [53000, 53001]
    for row in result["replications"]:
        assert row["measured_jobs"] == 100 and row["drained_jobs"] == 110
        assert row["arrival_rate"] == pytest.approx(row["load"] * 4 / pilot.WORKLOADS[regime].mean_work)
        assert row["mean_w"] == pytest.approx(row["mean_first_wait"] + row["mean_interruption"])
        assert row["mean_w"] == pytest.approx(row["mean_queue_wait"] + row["mean_checkpoint"] + row["mean_resume"])
        assert row["utilization"] == pytest.approx(
            row["productive_utilization"] + row["checkpoint_utilization"] + row["resume_utilization"]
        )
        assert row["backlog_at_last_arrival"] >= 1
        if row["policy"] not in pilot.COSTS or row["policy"] == pilot.ZERO:
            assert row["checkpoint_resource_time"] == row["resume_resource_time"] == 0
        if row["policy"] in pilot.COSTS:
            assert row["promise_violation_rate"] is None
    for load in pilot.LOADS:
        for seed in (53000, 53001):
            rows = [r for r in result["replications"] if r["load"] == load and r["seed"] == seed]
            assert len({r["arrival_rate"] for r in rows}) == len({r["sample_offered_load"] for r in rows}) == 1
    for key in ("zero_contrasts", "first_fit_contrasts", "msf_contrasts"):
        for summary in result[key]:
            samples = [
                r for r in result["replications"] if r["load"] == summary["load"] and r["policy"] == summary["policy"]
            ]
            refs = {
                r["seed"]: r
                for r in result["replications"]
                if r["load"] == summary["load"] and r["policy"] == summary["baseline"]
            }
            expected = interval([r["weighted_mean_t"] - refs[r["seed"]]["weighted_mean_t"] for r in samples])
            assert summary["paired_delta"]["weighted_mean_t"] == pytest.approx(expected)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "regime,shape,jobs,reps",
    [
        ("general", "erlang2", 100, 2),
        ("powers_of_two", "bad", 100, 2),
        ("one_or_all", "erlang2", 99, 2),
        ("one_or_all", "erlang2", 100, 1),
    ],
)
def test_invalid_grid_is_rejected(regime, shape, jobs, reps):
    """No unsupported domains or underpowered grids are silently substituted."""
    with pytest.raises(ValueError):
        pilot.experiment(regime, shape, jobs, reps)
