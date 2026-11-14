#!/usr/bin/env python3
"""Compare two serving policies and paired client SLOs on saved measurements.

Default data are DEVELOPMENT data. A fresh evaluation requires --frozen-fit;
no observations from the evaluated campaign are then used to fit coefficients.
The example performs no network calls and allocates no GPU.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from most_queue.io.runpod import read_runpod_trace
from most_queue.theory.continuous_batching.advance_reservation_calibration import (
    AdvanceReservationFit,
    fit_advance_reservation,
)
from most_queue.theory.continuous_batching.advance_reservation_validation import (
    compare_joint_slo,
    replay_advance_reservation,
)
from most_queue.theory.continuous_batching.serving_calibration import IsolatedServingFit, fit_isolated_serving
from most_queue.theory.continuous_batching.serving_validation import replay_serving_cohort
from tools.audit_reservation_validation import audit_fixture

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "tests/fixtures/serving_measurements/runpod_qwen3_8b_a40_calibration_v1_20261114"
PROFILE = "qwen3-8b-a40-vllm032-b4-unchunked"
SLO_TTFT = (0.25, 0.5, 1.0, 2.0)
SLO_TPOT = (0.030, 0.035, 0.040)
PROFILE_FIELDS = ("image", "model", "revision", "server_args")
ACCURACY_TARGETS = (
    ("isolated", "server_ttft", "maximum", 0.15),
    ("isolated", "server_tpot", "maximum", 0.05),
    ("burst", "server_ttft", "median", 0.15),
    ("burst", "server_ttft", "maximum", 0.35),
    ("burst", "client_ttft", "median", 0.15),
    ("burst", "client_ttft", "maximum", 0.40),
    ("burst", "server_tpot", "maximum", 0.10),
    ("open_loop", "client_ttft", "maximum", 0.20),
    ("open_loop", "server_ttft", "maximum", 0.20),
    ("open_loop", "server_tpot", "maximum", 0.05),
    ("open_loop", "server_queue_wait", "maximum", 0.25),
)


def digest(path):
    """Hash exact saved bytes for provenance."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_fit():
    """Fit only the 27 original isolated training observations, retaining baseline."""
    cohorts = read_runpod_trace(DEVELOPMENT / "requests.jsonl.gz", measurement_profile=PROFILE, include_warmup=False)
    training = [b for b in cohorts if b.metadata["split"] == "train"]
    profile = json.loads((DEVELOPMENT / "profile.json").read_text())
    return {
        "schema_version": 1,
        "training_trace_sha256": digest(DEVELOPMENT / "requests.jsonl.gz"),
        "training_series": [b.metadata["series_id"] for b in training],
        "profile": {k: profile[k] for k in PROFILE_FIELDS},
        "baseline": asdict(fit_isolated_serving(training)),
        "advance_reservation": asdict(fit_advance_reservation(training)),
        "slo_ttft_seconds": SLO_TTFT,
        "slo_tpot_seconds": SLO_TPOT,
        "accuracy_targets": ACCURACY_TARGETS,
        "primary_slo": {"ttft": 1.0, "tpot": 0.035, "maximum_fraction_error": 0.10},
        "baseline_burst_error_ratio_target": 0.5,
    }


def evaluate_targets(summary, cohorts, frozen):
    """Report every frozen target; a failed target never triggers refitting."""
    checks = []
    for mode, metric, statistic, target in frozen["accuracy_targets"]:
        value = summary[mode][metric]["advance_reservation"][statistic]
        checks.append(
            {"name": f"{mode}.{metric}.{statistic}", "value": value, "limit": target, "passed": value <= target}
        )
    primary = frozen["primary_slo"]
    values = [
        slo["absolute_fraction_error"]
        for row in cohorts
        if row["mode"] == "open_loop"
        for slo in row["joint_client_slo"]
        if slo["ttft_limit_seconds"] == primary["ttft"] and slo["tpot_limit_seconds"] == primary["tpot"]
    ]
    if len(values) != 6:
        raise ValueError("primary SLO must be present in all six open-loop series")
    checks.append(
        {
            "name": "primary_joint_client_slo.maximum_fraction_error",
            "value": max(values),
            "limit": primary["maximum_fraction_error"],
            "passed": max(values) <= primary["maximum_fraction_error"],
        }
    )
    for metric, statistic in (("server_ttft", "median"), ("server_tpot", "maximum")):
        data = summary["burst"][metric]
        value = data["advance_reservation"][statistic]
        limit = data["baseline"][statistic] * frozen["baseline_burst_error_ratio_target"]
        checks.append(
            {
                "name": f"burst_baseline_comparison.{metric}.{statistic}",
                "value": value,
                "limit": limit,
                "passed": value <= limit,
            }
        )
    return {"all_targets_met_on_this_data": all(c["passed"] for c in checks), "checks": checks}


def build_report(directory=DEVELOPMENT, *, frozen=None):
    """Replay all nontraining cohorts, with per-request errors and paired SLOs."""
    directory = Path(directory)
    is_development = digest(directory / "requests.jsonl.gz") == digest(DEVELOPMENT / "requests.jsonl.gz")
    if frozen is None and not is_development:
        raise ValueError("a different evaluation trace requires an explicitly frozen fit")
    frozen = freeze_fit() if frozen is None else frozen
    if frozen.get("schema_version") != 1:
        raise ValueError("unsupported fit schema")
    if frozen["training_trace_sha256"] != digest(DEVELOPMENT / "requests.jsonl.gz"):
        raise ValueError("frozen training provenance differs from the original fixture")
    profile = json.loads((directory / "profile.json").read_text())
    if {k: profile[k] for k in PROFILE_FIELDS} != frozen["profile"]:
        raise ValueError("evaluation model, image, revision and server flags must match training")
    runtime = json.loads((directory / "runtime.json").read_text())
    reference_runtime = json.loads((DEVELOPMENT / "runtime.json").read_text())
    if any(runtime.get(k) != reference_runtime[k] for k in ("gpu", "vllm", "torch", "cuda")):
        raise ValueError("this fit requires the pinned A40/vLLM runtime")
    config_name = "calibration-workload.json" if is_development else "reservation-v2-workload.json"
    config = json.loads((ROOT / "configs/runpod" / config_name).read_text())
    audit = audit_fixture(directory, config, 2026111402 if is_development else 2026111403)
    if audit["status"] != "complete_success":
        return {
            "data_role": "incomplete or failed; accuracy evaluation refused",
            "audit": audit,
            "summary": {},
            "cohorts": [],
        }
    fit = AdvanceReservationFit(**frozen["advance_reservation"])
    baseline = IsolatedServingFit(**frozen["baseline"])
    cohorts = read_runpod_trace(directory / "requests.jsonl.gz", measurement_profile=PROFILE, include_warmup=False)
    reports = []
    for benchmark in cohorts:
        if benchmark.metadata["split"] == "train":
            if not is_development:
                raise ValueError("fresh validation must not introduce new training observations")
            continue
        replay = replay_advance_reservation(fit, benchmark)
        row = {k: benchmark.metadata[k] for k in ("series_id", "mode", "repetition", "request_rate", "max_concurrency")}
        row.update(
            input_tokens=benchmark.requests[0].input_tokens,
            output_tokens=benchmark.requests[0].output_tokens,
            baseline=replay_serving_cohort(baseline, benchmark),
            advance_reservation=replay,
            joint_client_slo=[
                compare_joint_slo(benchmark, replay, ttft_limit=t, tpot_limit=p)
                for t in frozen["slo_ttft_seconds"]
                for p in frozen["slo_tpot_seconds"]
            ],
        )
        reports.append(row)
    summary = {}
    for mode in ("isolated", "burst", "open_loop"):
        selected = [r for r in reports if r["mode"] == mode]
        if not selected:
            raise ValueError(f"missing evaluation mode: {mode}")
        summary[mode] = {}
        for metric in ("client_ttft", "server_ttft", "server_tpot", "server_queue_wait"):
            key = "normalized_mean_error" if metric == "server_queue_wait" else "relative_mean_error"
            summary[mode][metric] = {}
            for policy in ("baseline", "advance_reservation"):
                values = [r[policy]["metrics"][metric][key] for r in selected]
                summary[mode][metric][policy] = {"median": float(np.median(values)), "maximum": max(values)}
    return {
        "schema_version": 1,
        "data_role": (
            "development; model chosen after inspecting these observations"
            if is_development
            else "fresh evaluation; fixed fit"
        ),
        "trace_sha256": digest(directory / "requests.jsonl.gz"),
        "audit": audit,
        "fit": frozen,
        "summary": summary,
        "targets": evaluate_targets(summary, reports, frozen),
        "cohorts": reports,
        "limitations": [
            "Specified advance reservation approximation, not exact vLLM scheduling.",
            "Client residual is isolated-trained and does not model frontend contention.",
            "Client TPOT approximates server TPOT; both SLO conditions use client observations.",
            "Finite cohorts, no p99/steady-state/capacity guarantee or iid confidence interval.",
        ],
    }


def main():
    """Write a reproducible report and optionally freeze training coefficients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-fixture", type=Path, default=DEVELOPMENT)
    parser.add_argument("--frozen-fit", type=Path)
    parser.add_argument("--freeze-fit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_fit.read_text()) if args.frozen_fit else freeze_fit()
    if args.freeze_fit:
        args.freeze_fit.write_text(json.dumps(frozen, indent=2, allow_nan=False) + "\n")
    # Without --frozen-fit, do not silently apply a newly fitted model to fresh data.
    report = build_report(args.evaluation_fixture, frozen=frozen if args.frozen_fit else None)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(json.dumps({"data_role": report["data_role"], "summary": report["summary"]}, indent=2))
    if report["audit"]["status"] != "complete_success":
        print(json.dumps(report["audit"], indent=2))
        return 1
    print(json.dumps(report["targets"], indent=2))
    print("\nJoint client SLO: TTFT <= 1 s AND TPOT <= 35 ms")
    for row in report["cohorts"]:
        if row["mode"] == "open_loop":
            slo = next(
                s for s in row["joint_client_slo"] if s["ttft_limit_seconds"] == 1 and s["tpot_limit_seconds"] == 0.035
            )
            print(
                f"rate={row['request_rate']:.1f} seed={row['repetition']}: "
                f"observed={slo['observed_fraction']:.3f}, predicted={slo['predicted_fraction']:.3f}"
            )
    return 0 if report["targets"]["all_targets_met_on_this_data"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
