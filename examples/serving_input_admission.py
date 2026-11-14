#!/usr/bin/env python3
"""Replay every compatible saved campaign with a fixed input-admission prototype."""

import argparse
import gzip
import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from examples.serving_frontend_diagnostics import unpack_fixture
from most_queue.io.runpod import read_runpod_trace
from most_queue.theory.continuous_batching.advance_reservation_calibration import AdvanceReservationFit
from most_queue.theory.continuous_batching.input_admission_replay import replay_input_admission
from most_queue.theory.continuous_batching.input_timing import fit_input_timing
from tools.audit_frontend_measurements import audit
from tools.audit_reservation_validation import audit_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/serving_measurements"
PROFILE = "qwen3-8b-a40-vllm032-b4-unchunked"
PRIMARY = "input_two_batches"
VARIANTS = (
    "baseline_v2",
    "continuous_reservation",
    "input_one_batch",
    "two_batches_continuous_input",
    PRIMARY,
)
DATASETS = {
    "calibration_v1": (700, 78, "calibration-workload.json", 2026111402),
    "reservation_v2": (669, 51, "reservation-v2-workload.json", 2026111403),
    "frontend_v1": (609, 104, "frontend-v1-workload.json", 2026111404),
}
PLAN = ROOT / "docs/research/serving-input-admission-development-plan.md"
METRICS = ("client_ttft", "server_ttft", "server_tpot", "server_queue_wait", "outside_engine", "prequeue")


def digest(path):
    """Hash exact stored bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_study(name, frozen):
    """Verify the complete fixture before returning all nonwarmup records."""
    directory = FIXTURES / f"runpod_qwen3_8b_a40_{name}_20261114"
    manifest = json.loads((directory / "manifest.json").read_text())
    for path, expected in manifest["sha256"].items():
        if digest(directory / path) != expected:
            raise ValueError(f"fixture identity differs: {name}/{path}")
    raw = gzip.decompress((directory / "requests.jsonl.gz").read_bytes())
    if "uncompressed_requests_sha256" in manifest:
        if hashlib.sha256(raw).hexdigest() != manifest["uncompressed_requests_sha256"]:
            raise ValueError("uncompressed request identity differs")
    config = json.loads((ROOT / "configs/runpod" / DATASETS[name][2]).read_text())
    profile = json.loads((directory / "profile.json").read_text())
    if any(profile[key] != frozen["profile"][key] for key in ("image", "model", "revision", "server_args")):
        raise ValueError("engine configuration differs from the frozen fit")
    runtime = json.loads((directory / "runtime.json").read_text())
    reference = json.loads((FIXTURES / "runpod_qwen3_8b_a40_calibration_v1_20261114/runtime.json").read_text())
    if any(runtime[key] != reference[key] for key in ("gpu", "vllm", "torch", "cuda")):
        raise ValueError("runtime differs from the frozen comparison")
    if name == "frontend_v1":
        with tempfile.TemporaryDirectory(prefix="mq-input-audit-") as temporary:
            measurements = unpack_fixture(directory, Path(temporary))
            checked = audit(measurements, config, profile["seed"])
        checked = {key: value for key, value in checked.items() if key != "intervals"}
    else:
        checked = audit_fixture(directory, config, DATASETS[name][3])
    if checked["status"] != "complete_success":
        raise ValueError("complete successful audited study required: " + name)
    records = [json.loads(line) for line in raw.splitlines()]
    measured = [row for row in records if row["phase"] != "warmup"]
    if (len(measured), len({row["series_id"] for row in measured})) != DATASETS[name][:2]:
        raise ValueError("missing or extra nonwarmup requests/series")
    return measured, {
        "audit": checked,
        "requests_gzip_sha256": digest(directory / "requests.jsonl.gz"),
        "manifest_sha256": digest(directory / "manifest.json"),
        "source_split_counts": dict(Counter(row["split"] for row in measured)),
    }


def point(variant, ttft):
    """Select a prespecified 35-ms TPOT comparison without tolerance rounding."""
    return next(
        item
        for item in variant["joint_client_slo"]
        if item["ttft_limit_seconds"] == ttft and item["tpot_limit_seconds"] == 0.035
    )


def fraction_error(variant, ttft):
    """Return absolute fraction error, not the per-request misclassification rate."""
    item = point(variant, ttft)
    return abs(item["predicted_passed"] - item["observed_passed"]) / item["offered"]


def evaluate_gate(cohorts):
    """Apply the recorded numerical gate to the primary candidate, never a winner."""
    selected = [cohort for cohort in cohorts if cohort["mode"] == "open_loop"]
    if Counter(row["dataset"] for row in selected) != {"calibration_v1": 6, "reservation_v2": 6, "frontend_v1": 4}:
        raise ValueError("all 16 open-loop series are required by the decision gate")
    baseline = np.asarray([fraction_error(row["variants"]["baseline_v2"], 0.25) for row in selected])
    candidate = np.asarray([fraction_error(row["variants"][PRIMARY], 0.25) for row in selected])
    checks = []

    def check(name, actual, limit):
        checks.append({"name": name, "actual": float(actual), "maximum": float(limit), "passed": bool(actual <= limit)})

    check("tight_slo_max_reduced_by_25_percent", candidate.max(), 0.75 * baseline.max())
    check("tight_slo_median_reduced_by_20_percent", np.median(candidate), 0.8 * np.median(baseline))
    check(
        "one_second_slo_max_at_most_5_points",
        max(fraction_error(row["variants"][PRIMARY], 1.0) for row in selected),
        0.05,
    )
    for name in DATASETS:
        rows = [row for row in selected if row["dataset"] == name]
        base = np.median([fraction_error(row["variants"]["baseline_v2"], 0.25) for row in rows])
        new = np.median([fraction_error(row["variants"][PRIMARY], 0.25) for row in rows])
        check(name + "_tight_slo_median_not_increased", new, base)
        total = sum(row["requests"] for row in rows)
        maes = {
            key: sum(row["requests"] * row["variants"][key]["metrics"]["client_ttft"]["mae_seconds"] for row in rows)
            / total
            for key in ("baseline_v2", PRIMARY)
        }
        check(name + "_client_ttft_mae_within_5_percent", maes[PRIMARY], 1.05 * maes["baseline_v2"])
    return {
        "primary_variant": PRIMARY,
        "scope": "all 16 development open-loop series; engineering gate, not fresh validation",
        "numerical_gate_passed": all(item["passed"] for item in checks),
        "checks": checks,
    }


def aggregate(cohorts):
    """Report cohort mean errors and request-weighted MAE without dropping data."""
    result = {}
    for dataset in (*DATASETS, "all"):
        result[dataset] = {}
        for mode in ("isolated", "burst", "open_loop"):
            selected = [row for row in cohorts if row["mode"] == mode and dataset in ("all", row["dataset"])]
            if not selected:
                continue
            total = sum(row["requests"] for row in selected)
            policies = {}
            for key in VARIANTS:
                metrics = {}
                for metric in METRICS:
                    if metric not in selected[0]["variants"][key]["metrics"]:
                        continue
                    values = [row["variants"][key]["metrics"][metric] for row in selected]
                    errors = [
                        value["relative_mean_error"] for value in values if value["relative_mean_error"] is not None
                    ]
                    metrics[metric] = {
                        "request_mae_seconds": sum(
                            row["requests"] * value["mae_seconds"] for row, value in zip(selected, values)
                        )
                        / total,
                        "median_cohort_relative_mean_error": float(np.median(errors)) if errors else None,
                        "maximum_cohort_relative_mean_error": max(errors) if errors else None,
                    }
                policies[key] = {
                    "metrics": metrics,
                    "slo": {
                        str(ttft): {
                            "median_fraction_error": float(
                                np.median([fraction_error(row["variants"][key], ttft) for row in selected])
                            ),
                            "maximum_fraction_error": max(
                                fraction_error(row["variants"][key], ttft) for row in selected
                            ),
                        }
                        for ttft in (0.25, 0.5, 1.0)
                    },
                }
            result[dataset][mode] = {"series": len(selected), "requests": total, "variants": policies}
    return result


def brief_cohort(cohort):
    """Retain every scalar result; store request arrays in the companion archive."""
    brief = {key: value for key, value in cohort.items() if key != "variants"}
    brief["variants"] = {}
    for name, variant in cohort["variants"].items():
        brief["variants"][name] = {
            "metrics": {
                metric: {
                    key: value[key]
                    for key in ("observed_mean_seconds", "predicted_mean_seconds", "relative_mean_error", "mae_seconds")
                }
                for metric, value in variant["metrics"].items()
            },
            "joint_client_slo": variant["joint_client_slo"],
        }
    return brief


def build_report():
    """Audit three complete studies, fit only idle timing, then replay fixed policies."""
    frozen = json.loads((ROOT / "configs/runpod/reservation-v2-fit.json").read_text())
    studies = {name: load_study(name, frozen) for name in DATASETS}
    training = [
        row
        for row in studies["frontend_v1"][0]
        if row["condition"] == "idle" and row["role"] == "probe" and row["trace_enabled"] is True
    ]
    if len(training) != 40:
        raise ValueError("exactly 40 prespecified instrumented idle probes required")
    input_fit = fit_input_timing(training)
    server_fit = AdvanceReservationFit(**frozen["advance_reservation"])
    cohorts = []
    for name, (records, _) in studies.items():
        sources = {row["request_index"]: row for row in records}
        adapted = [{**row, "split": "validation" if row["split"] == "diagnostic" else row["split"]} for row in records]
        for benchmark in read_runpod_trace(adapted, measurement_profile=PROFILE, include_warmup=False):
            replay = replay_input_admission(server_fit, input_fit, benchmark)
            indices = replay["request_indices"]
            replay.update(
                dataset=name,
                series_id=benchmark.metadata["series_id"],
                mode=benchmark.metadata["mode"],
                source_split=sources[indices[0]]["split"],
                request_rate=benchmark.metadata["request_rate"],
                source_requests=[
                    {
                        key: sources[index].get(key)
                        for key in (
                            "request_index",
                            "input_tokens",
                            "output_tokens",
                            "block_id",
                            "condition",
                            "role",
                            "trace_enabled",
                        )
                    }
                    for index in indices
                ],
            )
            cohorts.append(replay)
    if len(cohorts) != 233 or sum(row["requests"] for row in cohorts) != 1978:
        raise ValueError("the complete 233-series / 1978-request development set is required")
    details = {"data_role": "development only; no fresh model validation", "cohorts": cohorts}
    report = {
        "schema_version": 1,
        "data_role": details["data_role"],
        "units": "seconds except token/count fields and fractions",
        "primary_variant": PRIMARY,
        "counts": {"requests": 1978, "series": 233, "open_loop_series": 16, "input_timing_training": 40},
        "input_fit": asdict(input_fit),
        "input_training_request_indices": [row["request_index"] for row in training],
        "server_fit": frozen,
        "provenance": {
            "studies": {name: provenance for name, (_, provenance) in studies.items()},
            "plan_sha256": digest(PLAN),
            "source_sha256": {
                path: digest(ROOT / path)
                for path in (
                    "most_queue/sim/iteration_admission.py",
                    "most_queue/theory/continuous_batching/input_timing.py",
                    "most_queue/theory/continuous_batching/input_admission_replay.py",
                    "examples/serving_input_admission.py",
                    "configs/runpod/reservation-v2-fit.json",
                )
            },
        },
        "decision": evaluate_gate(cohorts),
        "summary": aggregate(cohorts),
        "open_loop_cohorts": [brief_cohort(row) for row in cohorts if row["mode"] == "open_loop"],
        "series_inventory": [
            {key: row[key] for key in ("dataset", "series_id", "mode", "source_split", "requests", "request_indices")}
            for row in cohorts
        ],
        "limitations": [
            "All datasets are development for this policy; historical split labels are archival.",
            "Idle frontend probes estimate shared ready/delivery delays; server costs remain frozen v2.",
            "Only baseline v2 uses observed engine arrivals; all new policies use client arrivals and lengths.",
            "Input polling repartitions waiting but has no CPU service cost; two planned batches change the timeline.",
            "No CPU scheduling/IPC variability, preemption, cache limits, early stopping or kernel variation.",
            "Ready/delivery fits extrapolate beyond 128..1024; equal delivery at first/last output is assumed.",
            "Finite drained and partly closed-loop cohorts; no fresh validation, p99, iid CI or capacity guarantee.",
        ],
    }
    return report, details


def main():
    """Save deterministic summary and complete request-level compressed diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/research/serving-input-admission-development.json")
    args = parser.parse_args()
    report, details = build_report()
    raw = (json.dumps(details, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    compressed = gzip.compress(raw, mtime=0)
    archive = args.output.with_suffix(".requests.json.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(compressed)
    report["request_details"] = {
        "path": archive.name,
        "sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(compressed),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
