"""Frozen development provenance and complete per-cohort SLO reporting."""

import copy
import json
from collections import Counter

import pytest

from examples import serving_reservation_slo as example


@pytest.fixture(scope="module")
def development_report():
    """Run the actual saved development cohort once for this module."""
    return example.build_report()


def test_development_report_covers_all_evaluation_cohorts(development_report):
    report = development_report
    assert report["data_role"] == "development; model chosen after inspecting these observations"
    assert report["audit"]["status"] == "complete_success"
    assert report["fit"]["advance_reservation"]["training_requests"] == 27
    assert report["fit"]["baseline"]["training_requests"] == 27
    assert Counter(row["mode"] for row in report["cohorts"]) == {"isolated": 9, "burst": 36, "open_loop": 6}
    pairs = {(t, p) for t in example.SLO_TTFT for p in example.SLO_TPOT}
    assert len(pairs) == 12
    for row in report["cohorts"]:
        assert "baseline" in row and "advance_reservation" in row
        assert len(row["joint_client_slo"]) == 12
        assert {(s["ttft_limit_seconds"], s["tpot_limit_seconds"]) for s in row["joint_client_slo"]} == pairs
        assert row["baseline"]["requests"] == row["advance_reservation"]["requests"]
    json.dumps(report, allow_nan=False)


def test_freeze_fit_uses_only_original_training(development_report, monkeypatch):
    captured = {}
    original_reader = example.read_runpod_trace

    def reader(*args, **kwargs):
        cohorts = original_reader(*args, **kwargs)
        captured["train"] = [b.metadata["series_id"] for b in cohorts if b.metadata["split"] == "train"]
        return cohorts

    monkeypatch.setattr(example, "read_runpod_trace", reader)
    frozen = example.freeze_fit()
    assert frozen["training_series"] == captured["train"]
    assert frozen == development_report["fit"]
    assert frozen["advance_reservation"]["training_requests"] == 27


@pytest.fixture
def cheap_source_checks(monkeypatch, development_report):
    """Keep contract rejection paths independent of expensive trace parsing."""
    frozen = copy.deepcopy(development_report["fit"])
    monkeypatch.setattr(example, "digest", lambda path: frozen["training_trace_sha256"])
    monkeypatch.setattr(
        example, "read_runpod_trace", lambda *a, **k: pytest.fail("trace read before contract validated")
    )
    return frozen


@pytest.mark.parametrize("field", example.PROFILE_FIELDS)
def test_profile_mismatch_is_strict(field, cheap_source_checks):
    frozen = cheap_source_checks
    frozen["profile"][field] = "incompatible"
    with pytest.raises(ValueError, match="model, image, revision and server flags"):
        example.build_report(frozen=frozen)


@pytest.mark.parametrize("field", ["gpu", "vllm", "torch", "cuda"])
def test_runtime_mismatch_is_strict(field, cheap_source_checks, monkeypatch, tmp_path):
    directory = tmp_path / "evaluation"
    directory.mkdir()
    (directory / "profile.json").write_text((example.DEVELOPMENT / "profile.json").read_text())
    runtime = json.loads((example.DEVELOPMENT / "runtime.json").read_text())
    runtime[field] = "incompatible"
    (directory / "runtime.json").write_text(json.dumps(runtime))
    with pytest.raises(ValueError, match="pinned A40/vLLM runtime"):
        example.build_report(directory, frozen=cheap_source_checks)


def test_other_trace_requires_explicit_frozen_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(example, "digest", lambda path: "original" if path.parent == example.DEVELOPMENT else "new")
    monkeypatch.setattr(example, "freeze_fit", lambda: pytest.fail("must not fit fresh evaluation"))
    with pytest.raises(ValueError, match="explicitly frozen fit"):
        example.build_report(tmp_path)


@pytest.mark.parametrize("status", ["incomplete", "failed"])
def test_incomplete_audit_returns_failure_report(status, cheap_source_checks, monkeypatch):
    audit = {"status": status, "reason": "missing or unsuccessful offered request"}
    monkeypatch.setattr(example, "audit_fixture", lambda *a, **k: audit)
    report = example.build_report(frozen=cheap_source_checks)
    assert report["audit"] == audit
    assert report["data_role"] == "incomplete or failed; accuracy evaluation refused"
    assert report["summary"] == {} and report["cohorts"] == []


def target_checks(report):
    """Index named diagnostics so a changed gate cannot hide another one."""
    evaluated = example.evaluate_targets(report["summary"], report["cohorts"], report["fit"])
    return {check["name"]: check for check in evaluated["checks"]}


def test_open_loop_maximum_error_changes_only_its_named_gate(development_report):
    report = copy.deepcopy(development_report)
    before = target_checks(report)
    name = "open_loop.server_ttft.maximum"
    assert before[name]["passed"]
    report["summary"]["open_loop"]["server_ttft"]["advance_reservation"]["maximum"] = before[name]["limit"] + 1
    after = target_checks(report)
    assert not after[name]["passed"]
    assert {k: v for k, v in after.items() if k != name} == {k: v for k, v in before.items() if k != name}


def test_primary_joint_gate_is_independent_of_marginal_latency_errors(development_report):
    report = copy.deepcopy(development_report)
    before = target_checks(report)
    primary = report["fit"]["primary_slo"]
    row = next(r for r in report["cohorts"] if r["mode"] == "open_loop")
    slo = next(
        s
        for s in row["joint_client_slo"]
        if s["ttft_limit_seconds"] == primary["ttft"] and s["tpot_limit_seconds"] == primary["tpot"]
    )
    slo["absolute_fraction_error"] = primary["maximum_fraction_error"] + 0.01
    after = target_checks(report)
    name = "primary_joint_client_slo.maximum_fraction_error"
    assert not after[name]["passed"]
    assert {k: v for k, v in after.items() if k != name} == {k: v for k, v in before.items() if k != name}
    assert report["summary"] == development_report["summary"]


def test_missing_primary_slo_from_one_open_loop_cohort_rejected(development_report):
    report = copy.deepcopy(development_report)
    primary = report["fit"]["primary_slo"]
    row = next(r for r in report["cohorts"] if r["mode"] == "open_loop")
    row["joint_client_slo"] = [
        s
        for s in row["joint_client_slo"]
        if not (s["ttft_limit_seconds"] == primary["ttft"] and s["tpot_limit_seconds"] == primary["tpot"])
    ]
    with pytest.raises(ValueError, match="primary SLO.*all six"):
        example.evaluate_targets(report["summary"], report["cohorts"], report["fit"])


def test_relative_baseline_improvement_can_fail_while_absolute_targets_pass(development_report):
    report = copy.deepcopy(development_report)
    for mode, metric, statistic, target in report["fit"]["accuracy_targets"]:
        report["summary"][mode][metric]["advance_reservation"][statistic] = target / 2
    # A better baseline makes the required relative improvement harder even
    # though advance-reservation errors remain below every absolute limit.
    data = report["summary"]["burst"]["server_ttft"]
    data["baseline"]["median"] = data["advance_reservation"]["median"]
    checks = target_checks(report)
    absolute_names = [
        f"{mode}.{metric}.{statistic}" for mode, metric, statistic, _ in report["fit"]["accuracy_targets"]
    ]
    assert all(checks[name]["passed"] for name in absolute_names)
    assert not checks["burst_baseline_comparison.server_ttft.median"]["passed"]


def test_explicit_frozen_fit_never_refits(development_report, monkeypatch):
    frozen = copy.deepcopy(development_report["fit"])
    monkeypatch.setattr(example, "freeze_fit", lambda: pytest.fail("explicit frozen model must not be refitted"))
    report = example.build_report(frozen=frozen)
    assert report["fit"] == frozen
    assert report["cohorts"] == development_report["cohorts"]
