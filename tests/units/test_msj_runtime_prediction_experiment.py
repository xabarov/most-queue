"""Split provenance and equal-work/equal-information checks for EPIC-049."""

import numpy as np
import pytest

from examples.msj_runtime_prediction_experiment import (
    ObservedJobs,
    fingerprint,
    make_splits,
    predict_modes,
    prediction_metrics,
    replay,
    train_predictors,
)
from most_queue.sim.msj_general import MsjGeneralSim


def test_test_cohort_length_cannot_change_training_or_calibration():
    first = make_splits("erlang2", 100, 49)
    second = make_splits("erlang2", 200, 49)
    for idx in (0, 1):
        assert fingerprint(first[idx]) == fingerprint(second[idx])
    assert len({fingerprint(cohort) for cohort in first[:3]}) == 3
    assert first[0].features.shape == (1500, 3)
    assert first[1].features.shape == (1000, 3)


def test_submission_features_do_not_depend_on_service_noise_law():
    erlang = make_splits("erlang2", 100, 49)
    lognormal = make_splits("lognormal_hetero", 100, 49)
    for a, b in zip(erlang[:3], lognormal[:3]):
        assert np.array_equal(a.classes, b.classes)
        assert np.array_equal(a.features, b.features)
        assert not np.array_equal(a.services, b.services)
    assert np.array_equal(erlang[3], lognormal[3])


def test_predictions_invariant_to_unknown_test_durations():
    train, cal, test, _ = make_splits("lognormal_hetero", 100, 49)
    class_means, model = train_predictors(train, cal)
    before = predict_modes(class_means, model, test.features, test.classes)
    changed = ObservedJobs(test.features, test.classes, test.services * 100)
    after = predict_modes(class_means, model, changed.features, changed.classes)
    assert set(before) == {"class_mean", "feature_point", "feature_upper95"}
    for mode in before:
        assert np.array_equal(before[mode], after[mode])
    assert model.calibration.rank == 951
    assert model.calibration.samples == 1000


def test_replay_policies_receive_identical_arrivals_and_actual_work(monkeypatch):
    train, cal, test, arrivals = make_splits("erlang2", 100, 49)
    means, model = train_predictors(train, cal)
    estimates = predict_modes(means, model, test.features, test.classes)["feature_upper95"]
    traces = []
    original = MsjGeneralSim.run_trace

    def capture(self, trace, **kwargs):
        traces.append(trace)
        return original(self, trace, **kwargs)

    monkeypatch.setattr(MsjGeneralSim, "run_trace", capture)
    for policy in ("fcfs", "easy", "conservative"):
        replay(test, arrivals, estimates, policy, 0.55, 1.0, 10)
    assert traces[0] == traces[1] == traces[2]
    assert [job.service for job in traces[0]] == list(test.services)
    assert [job.estimate for job in traces[0]] == list(estimates)


def test_slowdown_rescales_rate_without_changing_offered_load():
    _, _, test, arrivals = make_splits("erlang2", 100, 49)
    first = replay(test, arrivals, None, "fcfs", 0.55, 1, 10)
    slow = ObservedJobs(test.features, test.classes, test.services * 2)
    second = replay(slow, arrivals, None, "fcfs", 0.55, 2, 10)
    assert second["arrival_rate"] == pytest.approx(first["arrival_rate"] / 2)
    assert second["sample_offered_load"] == first["sample_offered_load"]
    assert second["mean_t"] == pytest.approx(first["mean_t"] * 2)
    assert second["utilization"] == pytest.approx(first["utilization"])


def test_metrics_only_measure_post_warmup_arrivals():
    test = ObservedJobs(np.zeros((6, 3)), np.array([0, 1, 2, 0, 1, 2]), np.ones(6))
    metrics = prediction_metrics(np.array([0.1, 0.1, 0.1, 2, 2, 2]), test, 3)
    assert metrics["coverage"] == 1
    assert metrics["coverage_by_class"] == [1, 1, 1]
    assert metrics["class_counts"] == [1, 1, 1]
    assert metrics["mean_estimate_over_mean_service"] == 2
