"""Audit real serving measurements, fit a solo-request model, and score joint SLOs.

Uses the pinned, licensed nine-run fixture in tests/fixtures/serving_measurements.
No model downloads, GPU or network access. Run:
    python -m examples.serving_measurements --output-dir /tmp/serving-measurements
"""

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

# Select a headless backend before loading pyplot.
# pylint: disable=wrong-import-position
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from most_queue.io.serving import from_prefill_decode, read_sglang_stream_benchmark, summarize_serving_slo
from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching.serving_calibration import evaluate_isolated_serving, fit_isolated_serving


def synthetic_slo_replications():
    """Separate seeded DES experiment, with a t interval across independent runs."""
    goodputs = []
    fractions = []
    for seed in range(99100, 99112):
        rng = np.random.default_rng(seed)
        arrivals = np.cumsum(rng.exponential(1 / 3, 500))
        inputs = rng.choice([16, 64], len(arrivals))
        outputs = rng.choice([4, 16], len(arrivals))
        sim = PrefillDecodeSim(batch_cap=4, token_budget=128)
        sim.set_servers(16, 4, alpha=0.006, beta=0.0002, gamma=0.00001)
        result = sim.replay(arrivals, inputs, outputs)
        report = summarize_serving_slo(
            from_prefill_decode(result, arrivals, inputs, outputs), ttft_limit=0.05, tpot_limit=0.012
        )
        goodputs.append(report["observed_goodput"])
        fractions.append(report["joint_fraction_bounds"][0])

    def interval(values):
        mean = float(np.mean(values))
        half = float(t.ppf(0.975, len(values) - 1) * np.std(values, ddof=1) / np.sqrt(len(values)))
        return {"mean": mean, "replicate_t95": [mean - half, mean + half], "replicates": values}

    return {
        "kind": "synthetic independent seeded DES; not GPU calibration",
        "seeds": list(range(99100, 99112)),
        "jobs_per_replication": 500,
        "initial_state": "empty; full finite cohort drained",
        "ttft_limit": 0.05,
        "tpot_limit": 0.012,
        "goodput": interval(goodputs),
        "joint_fraction": interval(fractions),
        "uncertainty": "Student t interval across independent runs, not iid requests or a steady-state guarantee",
    }


def experiment(fixture):
    """Apply the frozen manifest split; never tune on the held-out run."""
    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    training, heldout, run_reports = [], [], []
    excluded = set(manifest["calibration_policy"]["excluded_request_indices"])
    for entry in manifest["files"]:
        path = fixture / entry["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"fixture checksum mismatch: {path.name}")
        bench = read_sglang_stream_benchmark(path, measurement_profile=manifest["measurement_profile"])
        calibration = replace(
            bench, requests=tuple(row for index, row in enumerate(bench.requests) if index not in excluded)
        )
        (training if entry["split"] == "train" else heldout).append(calibration)
        run_reports.append(
            {
                "filename": path.name,
                "split": entry["split"],
                **summarize_serving_slo(bench, ttft_limit=0.5, tpot_limit=0.02),
            }
        )
    fit = fit_isolated_serving(training)
    validation = evaluate_isolated_serving(fit, heldout)
    passes = all(validation[metric]["median_absolute_relative_error"] <= 0.15 for metric in ["ttft", "tpot"])
    return {
        "source_repository": manifest["source_repository"],
        "source_commit": manifest["source_commit"],
        "protocol": "docs/epics/EPIC-099-serving-measurements.md",
        "fit": asdict(fit),
        "validation": validation,
        "diagnostic_target": {"max_median_relative_error_per_metric": 0.15, "passed": passes},
        "runs": run_reports,
        "calibration_excluded_per_run": sorted(excluded),
        "server_queue_calibration": "unavailable: no server queue observations in this fixture",
        "loaded_serving_validation": False,
        "synthetic_slo_replications": synthetic_slo_replications(),
    }


def plot(payload, output):
    """Show held-out measured-versus-predicted values and joint SLO counts."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, metric, color in zip(axes[:2], ["ttft", "tpot"], ["#235789", "#B64B32"]):
        report = payload["validation"][metric]
        observed, predicted = np.asarray(report["observed"]), np.asarray(report["predicted"])
        ax.scatter(observed * 1000, predicted * 1000, color=color, s=30)
        bound = max(observed.max(), predicted.max()) * 1050
        ax.plot([0, bound], [0, bound], linestyle=":", color="#777777")
        error = report["median_absolute_relative_error"] * 100
        ax.set(
            xlabel="Measured (ms)",
            ylabel="Predicted (ms)",
            title=f"Held-out {metric.upper()} · median error {error:.1f}%",
        )
        ax.grid(alpha=0.15)
    grouped = {}
    for run in payload["runs"]:
        length = run["filename"].split("_in")[1].split("_")[0]
        passed, count = grouped.get(length, (0, 0))
        grouped[length] = passed + run["joint_passed"], count + run["offered"]
    axes[2].bar(list(grouped), [passed / count for passed, count in grouped.values()], color="#235789")
    for index, (passed, count) in enumerate(grouped.values()):
        axes[2].text(index, passed / count + 0.03, f"{passed}/{count}", ha="center")
    axes[2].set(
        ylim=(0, 1.15), xlabel="Nominal input tokens", ylabel="Joint SLO fraction", title="TTFT ≤ 500 ms, TPOT ≤ 20 ms"
    )
    fig.suptitle("Public vLLM measurements via SGLang client · single RTX PRO 6000 · concurrency 1", fontsize=11)
    fig.savefig(output / "serving-measurements.png", dpi=170)
    plt.close(fig)


def main():
    """Write a fully offline calibration/SLO report and plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests/fixtures/serving_measurements/flash_next",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "most-queue-serving-measurements"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = experiment(args.fixture_dir)
    (args.output_dir / "serving-measurements.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot(payload, args.output_dir)
    print("Held-out requests:", payload["validation"]["requests"])
    for metric in ["ttft", "tpot"]:
        print(
            metric.upper(),
            "median absolute relative error:",
            payload["validation"][metric]["median_absolute_relative_error"],
        )
    print("Frozen 15% target passed:", payload["diagnostic_target"]["passed"])
    print("Server queue calibration:", payload["server_queue_calibration"])
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
