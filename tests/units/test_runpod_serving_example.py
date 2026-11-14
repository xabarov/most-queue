"""The calibration example audits complete designs and never retunes on holdout."""

import copy
import json
import math
from dataclasses import replace

import pytest

from examples import runpod_serving_calibration as example
from most_queue.theory.continuous_batching.serving_calibration import IsolatedServingFit
from tools.runpod_calibration_workload import generate_plan

PROFILE = "runpod-qwen3-8b-a40-timestamps-calibration-v1"


def test_default_cli_preserves_pilot_and_reports_exploratory_fit(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert example.main(["--output", str(output)]) == 0
    serialized = output.read_text(encoding="utf-8")
    assert serialized == capsys.readouterr().out
    report = json.loads(serialized, parse_constant=lambda value: pytest.fail(f"nonfinite JSON: {value}"))
    assert report["requests"] == 81
    assert len(report["cohorts"]) == 21
    assert report["statuses"] == {"completed": 81}
    assert report["calibration_status"] == "exploratory_only_no_holdout"
    assert report["fit"]["training_requests"] == 6
    assert math.isfinite(report["fit"]["scaled_condition"])
    assert all(math.isfinite(report["fit"][name]) for name in ("alpha", "beta", "gamma", "client_offset"))
    assert any("no prespecified held-out split" in message for message in report["diagnostics"])
    assert any("exact server arrivals are absent" in message for message in report["diagnostics"])
    assert report["isolated_validation"] is None
    assert report["burst_validation"] == report["open_loop_validation"] == []
    assert report["validation_passed"] is None and report["experiment_valid"] is False
    assert report["provenance"]["trace"]["path"].startswith("tests/fixtures/")
    assert "/home/" not in serialized


@pytest.fixture
def campaign(tmp_path):
    config = json.loads((example.ROOT / "configs/runpod/calibration-workload.json").read_text(encoding="utf-8"))
    series = generate_plan(config, 2026111402)
    records, series_results = [], []
    for series_index, group in enumerate(series):
        specs = group["requests"]
        series_start = 1000 + series_index * 300
        for offset, spec in enumerate(specs):
            spec["offer_status"] = "OFFERED"
            scheduled_arrival = series_start + spec["arrival_offset_s"]
            start = scheduled_arrival + 0.002 + offset * 0.000001
            queued, scheduled, first, last = start + 0.01, start + 0.02, start + 0.03, start + 0.5
            engine_id = f"cmpl-{spec['request_id']}"
            records.append(
                {
                    **spec,
                    "client_started_monotonic_s": start,
                    "dispatch_started_monotonic_s": scheduled_arrival + 0.001,
                    "scheduled_arrival_monotonic_s": scheduled_arrival,
                    "usage": {"prompt_tokens": spec["input_tokens"], "completion_tokens": spec["output_tokens"]},
                    "completion_tokens": spec["output_tokens"],
                    "client_ttft_s": 0.1,
                    "latency_to_done_s": 1.0,
                    "client_elapsed_s": 1.01,
                    "metrics": {
                        "queue_time_ms": (scheduled - queued) * 1000,
                        "time_to_first_token_ms": (first - scheduled) * 1000,
                        "generation_time_ms": (last - first) * 1000,
                    },
                    "server_request_id": engine_id,
                    "server_timestamps": {
                        "request_id": engine_id + "-0",
                        "queued_ts": queued,
                        "scheduled_ts": scheduled,
                        "first_token_ts": first,
                        "last_token_ts": last,
                        "num_generation_tokens": spec["output_tokens"],
                        "num_preemptions": 0,
                    },
                    "success": True,
                    "done": True,
                    "error": None,
                    "status": "succeeded",
                }
            )
        series_results.append(
            {
                "series_id": group["series_id"],
                "planned": len(specs),
                "offered": len(specs),
                "successful": len(specs),
                "failed": 0,
                "not_offered_request_indices": [],
                "deadline_exhausted": False,
                "scheduled_start_monotonic_s": series_start,
                "scheduled_end_monotonic_s": series_start + group["duration_s"],
                "completed_monotonic_s": max(series_start + group["duration_s"], start + 1.01),
            }
        )
    plan = {"seed": 2026111402, "expected": 721, "series": series}
    summary = {
        "expected": 721,
        "completed": 721,
        "offered": 721,
        "successful": 721,
        "not_offered": 0,
        "measured_completed": 700,
        "measured_successful": 700,
        "experiment_valid": True,
        "abort_reason": None,
        "deadline_exhausted": False,
        "series": series_results,
    }
    source, plan_path, summary_path = (tmp_path / name for name in ("requests.jsonl", "plan.json", "summary.json"))

    def save():
        source.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    save()
    return {
        "records": records,
        "plan": plan,
        "summary": summary,
        "source": source,
        "plan_path": plan_path,
        "summary_path": summary_path,
        "save": save,
    }


@pytest.fixture
def mock_model(monkeypatch):
    fit = IsolatedServingFit(0.1, 0.01, 0.001, 0.0001, 10.0, 27, (128, 4096), (32, 512), PROFILE, "client_to_done")
    calls = {"training": [], "isolated": [], "replay": [], "fit": fit}

    def train(cohorts):
        calls["training"].append(cohorts)
        assert len(cohorts) == 27
        assert all(c.metadata["split"] == "train" and c.metadata["mode"] == "isolated" for c in cohorts)
        return calls["fit"]

    def isolated(actual_fit, cohorts):
        assert actual_fit is calls["fit"]
        calls["isolated"].append(cohorts)
        assert len(cohorts) == 9
        assert all(c.metadata["split"] == "validation" and c.metadata["mode"] == "isolated" for c in cohorts)
        return {
            "requests": 9,
            "ttft": {"median_absolute_relative_error": 0.15},
            "tpot": {"median_absolute_relative_error": 0.15},
        }

    def replay(actual_fit, cohort, *, batch_cap, token_budget):
        assert actual_fit is calls["fit"]
        assert batch_cap == 4 and token_budget == 18432
        calls["replay"].append(cohort)
        assert cohort.metadata["split"] == "validation"
        assert cohort.metadata["mode"] in {"burst", "open_loop"}
        metric = {
            "observed_mean_seconds": 1.0,
            "predicted_mean_seconds": 1.2,
            "mae_seconds": 0.2,
            "absolute_mean_error_seconds": 0.2,
            "relative_mean_error": 0.20,
            "normalized_mean_error": 0.25,
            "observed": [1.0] * len(cohort.requests),
            "predicted": [1.2] * len(cohort.requests),
        }
        return {
            "requests": len(cohort.requests),
            "metrics": {
                name: copy.deepcopy(metric)
                for name in ("client_ttft", "server_ttft", "server_tpot", "server_queue_wait")
            },
        }

    monkeypatch.setattr(example, "fit_isolated_serving", train)
    monkeypatch.setattr(example, "evaluate_isolated_serving", isolated)
    monkeypatch.setattr(example, "replay_serving_cohort", replay)
    return calls


def _report(campaign, **kwargs):
    return example.build_report(
        campaign["source"],
        measurement_profile=PROFILE,
        plan_path=campaign["plan_path"],
        summary_path=campaign["summary_path"],
        **kwargs,
    )


def test_full_campaign_uses_only_training_and_all_fixed_holdouts(campaign, mock_model):
    report = _report(campaign)
    assert report["experiment_valid"] is True
    assert report["validation_passed"] is True
    assert report["calibration_status"] == "validated"
    assert report["diagnostics"] == []
    assert len(mock_model["training"]) == len(mock_model["isolated"]) == 1
    assert len(mock_model["replay"]) == 42
    assert len(report["burst_validation"]) == 36
    assert sum(result["requests"] for result in report["burst_validation"]) == 108
    assert [result["requests"] for result in report["open_loop_validation"]] == [35, 41, 78, 89, 147, 166]
    assert len(report["outcomes"]) == 721
    assert report["provenance"]["trace"]["path"] == "requests.jsonl"
    assert report["gates"]["queue_denominator_floor_seconds"] == 0.05
    assert "not exact vLLM" in report["scope"] and "p99" in report["scope"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "metric,error", [("client_ttft", 0.200001), ("server_tpot", 0.200001), ("server_queue_wait", 0.250001)]
)
def test_each_loaded_frozen_gate_failure_is_retained_without_refit(campaign, mock_model, monkeypatch, metric, error):
    original = example.replay_serving_cohort

    def replay(*args, **kwargs):
        result = original(*args, **kwargs)
        key = "normalized_mean_error" if metric == "server_queue_wait" else "relative_mean_error"
        result["metrics"][metric][key] = error
        return result

    monkeypatch.setattr(example, "replay_serving_cohort", replay)
    report = _report(campaign)
    assert report["experiment_valid"] is True
    assert report["validation_passed"] is False
    assert report["calibration_status"] == "failed_frozen_gates"
    assert len(report["open_loop_validation"]) == 6
    assert len(mock_model["training"]) == 1


def test_isolated_gate_failure_does_not_hide_loaded_results(campaign, mock_model, monkeypatch):
    def evaluate(_fit, _cohorts):
        return {"ttft": {"median_absolute_relative_error": 0.151}, "tpot": {"median_absolute_relative_error": 0.15}}

    monkeypatch.setattr(example, "evaluate_isolated_serving", evaluate)
    report = _report(campaign)
    assert report["validation_passed"] is False
    assert report["gates"]["isolated"] == {"ttft": False, "tpot": True}
    assert len(report["open_loop_validation"]) == 6
    assert len(mock_model["training"]) == 1


@pytest.mark.parametrize("damage", ["missing", "failed", "duplicate", "status", "summary", "plan", "timestamps"])
def test_partial_failed_and_inconsistent_evidence_is_retained_and_invalidated(campaign, mock_model, damage):
    records = campaign["records"]
    if damage == "missing":
        records.pop()
    elif damage == "failed":
        records[-1].update(success=False, status="failed", done=False, error="timeout")
        campaign["summary"].update(successful=720, measured_successful=699, experiment_valid=False)
        campaign["summary"]["series"][-1].update(successful=165, failed=1)
    elif damage == "duplicate":
        records[-1]["request_index"] = records[-2]["request_index"]
    elif damage == "status":
        records[-1]["status"] = "failed"
    elif damage == "summary":
        campaign["summary"]["offered"] = 720
    elif damage == "plan":
        campaign["plan"]["series"][-1]["requests"][-1]["output_tokens"] = 32
    elif damage == "timestamps":
        records[-1].pop("server_timestamps")
    campaign["save"]()
    report = _report(campaign)
    assert report["experiment_valid"] is False
    assert report["validation_passed"] is None
    assert report["requests"] == len(records)
    assert len(report["outcomes"]) == len(records)
    assert report["diagnostics"]
    assert not mock_model["training"] and not mock_model["replay"]
    if damage == "failed":
        assert report["statuses"]["failed"] == 1
        assert report["outcomes"][-1]["error"] == "timeout"


def test_new_campaign_requires_plan_summary_and_non_smoke_profile(campaign, mock_model):
    missing = example.build_report(campaign["source"], measurement_profile=PROFILE)
    assert missing["experiment_valid"] is False
    assert "both --plan and --summary" in missing["diagnostics"][0]
    wrong_profile = example.build_report(
        campaign["source"], plan_path=campaign["plan_path"], summary_path=campaign["summary_path"]
    )
    assert wrong_profile["experiment_valid"] is False
    assert any("explicit measurement profile" in error for error in wrong_profile["diagnostics"])
    assert not mock_model["training"]


def test_nonpositive_alpha_is_diagnostic_and_skips_des(campaign, mock_model):
    mock_model["fit"] = replace(mock_model["fit"], alpha=0)
    report = _report(campaign)
    assert report["experiment_valid"] is True
    assert report["calibration_status"] == "nonpositive_alpha"
    assert report["validation_passed"] is None
    assert report["fit"]["alpha"] == 0
    assert not mock_model["replay"] and not mock_model["isolated"]


def test_nonfinite_model_output_never_enters_report_json(campaign, mock_model):
    mock_model["fit"] = replace(mock_model["fit"], alpha=float("nan"))
    report = _report(campaign)
    assert report["calibration_status"] == "evaluation_unavailable"
    assert report["fit"] is None
    assert not mock_model["replay"]
    json.dumps(report, allow_nan=False)


def test_legacy_identical_lengths_report_real_rank_deficiency(campaign):
    row = copy.deepcopy(campaign["records"][0])
    row.pop("series_id")
    row.pop("split")
    row["phase"] = "measured"
    second = dict(row, request_index=2)
    campaign["source"].write_text(json.dumps(row) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    report = example.build_report(campaign["source"], measurement_profile=PROFILE)
    assert report["calibration_status"] == "exploratory_fit_unavailable"
    assert report["fit"] is None and report["validation_passed"] is None
    assert any("rank-deficient" in error for error in report["diagnostics"])


@pytest.mark.parametrize("damage", ["failure", "invalid_import"])
def test_legacy_failed_or_invalid_trace_retains_evidence_without_fit(tmp_path, damage):
    records = example._read_records(example.DEFAULT_TRACE)  # pylint: disable=protected-access
    if damage == "failure":
        records[-1].update(success=False, done=False, error="timeout")
    else:
        records[-1]["latency_to_done_s"] = None
    source = tmp_path / "legacy.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    report = example.build_report(source)
    assert report["calibration_status"] == "exploratory_fit_unavailable"
    assert report["fit"] is None and report["validation_passed"] is None
    assert len(report["outcomes"]) == report["requests"] == 81
    assert report["diagnostics"]


@pytest.mark.parametrize(
    "damage",
    [
        "campaign_seed",
        "request_seed",
        "request_id",
        "open_loop_offset",
        "fixed_offset",
        "arrival_seed",
        "duration",
        "scheduled_clock",
        "early_dispatch",
        "summary_clock",
    ],
)
def test_frozen_arrival_plan_rejects_mutations_without_changing_counts(campaign, mock_model, damage):
    series = campaign["plan"]["series"][-1]
    spec = series["requests"][-1]
    record = campaign["records"][-1]
    if damage == "campaign_seed":
        campaign["plan"]["seed"] += 1
    elif damage == "request_seed":
        spec["seed"] += 1
        record["seed"] += 1
    elif damage == "request_id":
        spec["request_id"] = record["request_id"] = "another-id"
    elif damage == "open_loop_offset":
        spec["arrival_offset_s"] += 0.1
        record["arrival_offset_s"] += 0.1
        record["scheduled_arrival_monotonic_s"] += 0.1
        record["dispatch_started_monotonic_s"] += 0.1
    elif damage == "fixed_offset":
        campaign["plan"]["series"][0]["requests"][0]["arrival_offset_s"] = 0.1
        campaign["records"][0]["arrival_offset_s"] = 0.1
    elif damage == "arrival_seed":
        series["arrival_seed"] += 1
    elif damage == "duration":
        series["duration_s"] = 181
    elif damage == "scheduled_clock":
        record["scheduled_arrival_monotonic_s"] += 0.00001
    elif damage == "early_dispatch":
        record["dispatch_started_monotonic_s"] = record["scheduled_arrival_monotonic_s"] - 0.00001
    elif damage == "summary_clock":
        campaign["summary"]["series"][-1]["scheduled_start_monotonic_s"] = float("nan")
    campaign["save"]()
    report = _report(campaign)
    assert report["requests"] == len(report["outcomes"]) == 721
    assert report["experiment_valid"] is False
    assert report["diagnostics"]
    assert not mock_model["training"] and not mock_model["replay"]
    json.dumps(report, allow_nan=False)


def test_schedule_clock_tolerance_and_significant_lateness_reporting(campaign, mock_model):
    record = campaign["records"][-1]
    record["scheduled_arrival_monotonic_s"] += 0.00000005
    record["dispatch_started_monotonic_s"] += 10
    record["client_started_monotonic_s"] += 10
    for key in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts"):
        record["server_timestamps"][key] += 10
    campaign["summary"]["series"][-1]["completed_monotonic_s"] += 10
    campaign["save"]()
    report = _report(campaign)
    assert report["experiment_valid"] is True
    assert report["validation_passed"] is True
    assert report["cohorts"][-1]["dispatch_lateness_s"][-1] > 10
