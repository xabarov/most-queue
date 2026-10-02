"""Information separation, unavailable forecasts and paired experimental units."""

import json

import numpy as np
import pytest

from examples.msj_group_calibration_experiment import (
    NEEDS,
    QUEUE_METRICS,
    ObservedJobs,
    experiment,
    fingerprint,
    fit_predictor,
    forecast_metrics,
    forecasts,
    make_splits,
    replay,
    scarcity_sweep,
    summarize,
)
from most_queue.sim.msj_general import MsjGeneralSim
from most_queue.sim.utils.runtime_prediction import LogLinearRuntimePredictor


def test_forecasts_ignore_actual_test_durations_and_preserve_point_model():
    train, cal, test, _ = make_splits("lognormal_hetero", 100, 50000)
    model = fit_predictor(train, cal)
    first = forecasts(model, test.features, test.classes)
    point = model.predict(test.features)
    changed = ObservedJobs(test.features, test.classes, test.services * 100)
    second = forecasts(model, changed.features, changed.classes)
    for mode in first:
        np.testing.assert_array_equal(first[mode], second[mode])
    model.calibrate_by_group(cal.features, cal.services * 2, NEEDS[cal.classes])
    np.testing.assert_array_equal(model.predict(test.features), point)
    np.testing.assert_array_equal(model.predict(test.features, upper=True), first["pooled95"])


def test_scarcity_uses_nested_prefixes_and_does_not_refit_regression():
    train, cal, test, _ = make_splits("erlang2", 100, 50000)
    model = fit_predictor(train, cal)
    point, before = model.predict(test.features), model.group_calibrations
    rows = scarcity_sweep(model, cal, test, 10)
    for row in rows:
        n = row["calibration_size"]
        prefix = ObservedJobs(cal.features[:n], cal.classes[:n], cal.services[:n])
        assert row["calibration_fingerprint"] == fingerprint(prefix)
        assert sum(g["calibration"]["samples"] for g in row["groups"] if g["calibration"] is not None) == n
        assert sum(g["test_jobs"] for g in row["groups"]) == 90
    np.testing.assert_array_equal(model.predict(test.features), point)
    assert model.group_calibrations == before  # last prefix is the full original cohort


@pytest.mark.parametrize("counts", [(19, 18, 1), (19, 19, 0)])
def test_scarcity_marks_insufficient_and_unseen_groups_without_dropping_jobs(counts, monkeypatch):
    def must_not_schedule(*args, **kwargs):
        raise AssertionError("scarcity diagnostics must not schedule a filtered cohort")

    monkeypatch.setattr(MsjGeneralSim, "run_trace", must_not_schedule)
    model = LogLinearRuntimePredictor().fit(np.empty((2, 0)), [1, 1])
    classes = np.repeat(np.arange(3), counts)
    cal = ObservedJobs(np.empty((38, 0)), classes, np.ones(38))
    test = ObservedJobs(np.empty((6, 0)), np.array([0, 1, 2, 0, 1, 2]), np.ones(6))
    row = scarcity_sweep(model, cal, test, 0, sizes=[38])[0]
    assert not row["all_groups_finite"]
    assert row["finite_job_fraction"] == pytest.approx((1 if counts[1] == 18 else 2) / 3)
    assert row["groups"][0]["grouped_coverage"] == 1
    assert row["groups"][2]["grouped_coverage"] is None
    assert row["groups"][2]["grouped_estimate_ratio"] is None
    assert row["groups"][2]["status"] == ("unseen" if counts[2] == 0 else "insufficient")
    assert sum(group["test_jobs"] for group in row["groups"]) == 6
    json.dumps(row, allow_nan=False)


def test_replay_keeps_actual_work_and_class_metrics_for_every_policy(monkeypatch):
    train, cal, test, arrivals = make_splits("erlang2", 200, 50000)
    model = fit_predictor(train, cal)
    bounds = forecasts(model, test.features, test.classes)
    recorded = []
    original = MsjGeneralSim.run_trace

    def capture(self, trace, **kwargs):
        measured = original(self, trace, **kwargs)
        recorded.append((trace, measured))
        return measured

    monkeypatch.setattr(MsjGeneralSim, "run_trace", capture)
    for policy in ("easy", "conservative"):
        for estimate in bounds.values():
            metrics = replay(test, arrivals, estimate, policy, 0.55, 1, 20)
            measured = recorded[-1][1]
            assert sum(metrics[f"reservations_k{k}"] for k in NEEDS) == measured.reservations
            assert sum(metrics[f"reservation_violations_k{k}"] for k in NEEDS) == measured.reservation_violations
            for cls, need in enumerate(NEEDS):
                assert metrics[f"mean_t_k{need}"] == measured.v_per_class[cls]
                assert metrics[f"p99_t_k{need}"] == measured.v_quantiles_per_class[cls][0.99]
    work = [[(j.arrival, j.cls, j.service) for j in trace] for trace, _ in recorded]
    assert all(trace == work[0] for trace in work)


def test_shift_preserves_offered_load_and_oracle_scales_time_only():
    _, _, test, arrivals = make_splits("lognormal_hetero", 200, 50000)
    first = replay(test, arrivals, test.services, "conservative", 0.55, 1, 20)
    slow = ObservedJobs(test.features, test.classes, test.services * 2)
    second = replay(slow, arrivals, slow.services, "conservative", 0.55, 2, 20)
    assert second["arrival_rate"] == first["arrival_rate"] / 2
    assert second["sample_offered_load"] == first["sample_offered_load"]
    assert second["utilization"] == pytest.approx(first["utilization"])
    assert second["reservation_violations"] == first["reservation_violations"] == 0
    for key in ("mean_t", "p99_t", "mean_t_k1", "mean_t_k3", "mean_t_k4"):
        assert second[key] == pytest.approx(first[key] * 2)


def test_paired_summary_matches_seed_not_input_row_order():
    rows = [
        {"seed": 2, "mode": "grouped", "metric": 23},
        {"seed": 1, "mode": "pooled", "metric": 10},
        {"seed": 1, "mode": "grouped", "metric": 13},
        {"seed": 2, "mode": "pooled", "metric": 20},
    ]
    result = summarize(rows, (), ("metric",), "mode", lambda _: "pooled")
    grouped = next(row for row in result if row["mode"] == "grouped")
    assert grouped["paired_delta"]["metric"] == {"mean": 3, "low": 3, "high": 3}


def test_group_metrics_exclude_warmup_and_expose_bound_size():
    test = ObservedJobs(np.empty((6, 0)), np.array([0, 1, 2, 0, 1, 2]), np.ones(6))
    result = forecast_metrics(np.array([0.1, 0.1, 0.1, 2, 3, 4]), test, 3)
    assert result["coverage"] == 1
    for need, ratio in zip(NEEDS, [2, 3, 4]):
        assert result[f"coverage_k{need}"] == 1
        assert result[f"estimate_ratio_k{need}"] == ratio


def test_small_full_grid_is_strict_json_with_expected_experimental_units():
    result = experiment("erlang2", jobs=100, replications=2)
    json.dumps(result, allow_nan=False)
    assert len(result["replications"]) == 56
    assert len(result["prediction_runs"]) == 12
    assert len(result["calibration_runs"]) == 2
    assert len(result["scarcity_runs"]) == 8
    assert len(result["summaries"]) == 28
    assert len(result["prediction_summaries"]) == 6
    assert set(result["summaries"][0]["metrics"]) == set(QUEUE_METRICS)
    assert {r["seed"] for r in result["replications"]} == {50000, 50001}
