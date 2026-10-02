"""Paired pooled/grouped runtime calibration pilot (EPIC-050).

    python -m examples.msj_group_calibration_experiment --shape lognormal_hetero \
        --jobs 4000 --replications 12 --output works/msj_group_calibration/lognormal.json

Uses the EPIC-049 data law but new seed bundles, without modifying its artifacts.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from examples.msj_backfilling_experiment import interval
from examples.msj_runtime_prediction_experiment import (
    FEATURE_SLOPE,
    LOADS,
    MEANS,
    NEEDS,
    PROBABILITIES,
    REGIMES,
    SHAPES,
    ObservedJobs,
    fingerprint,
    make_splits,
    prediction_metrics,
)
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.utils.runtime_prediction import LogLinearRuntimePredictor

MODES = ("pooled95", "grouped95", "oracle")
POLICIES = ("fcfs",) + tuple(f"{policy}:{mode}" for policy in ("easy", "conservative") for mode in MODES)
CALIBRATION_SIZES = (60, 100, 300, 1000)
QUEUE_METRICS = ("mean_t", "p99_t", "idle_with_queue", "reservation_violations") + tuple(
    f"{metric}_k{need}" for metric in ("mean_t", "p99_t", "reservation_violations") for need in NEEDS
)
PREDICTION_METRICS = ("coverage", "mean_estimate_over_mean_service") + tuple(
    f"{metric}_k{need}" for metric in ("coverage", "estimate_ratio") for need in NEEDS
)


def fit_predictor(train, cal):
    """Fit once on historical training labels, then calibrate both modes."""
    model = LogLinearRuntimePredictor().fit(train.features, train.services)
    model.calibrate(cal.features, cal.services)
    model.calibrate_by_group(cal.features, cal.services, NEEDS[cal.classes])
    return model


def forecasts(model, features, classes):
    """No actual future runtime is accepted by this non-oracle path."""
    return {
        "pooled95": model.predict(features, upper=True),
        "grouped95": model.predict(features, upper=True, groups=NEEDS[classes]),
    }


def forecast_metrics(estimate, cohort, warmup):
    """Report post-warmup coverage and forecast size for every resource class."""
    result = prediction_metrics(estimate, cohort, warmup)
    for cls, need in enumerate(NEEDS):
        mask = cohort.classes[warmup:] == cls
        if not mask.any():
            raise ValueError("measured cohort must include all three resource classes")
        result[f"coverage_k{need}"] = result["coverage_by_class"][cls]
        result[f"estimate_ratio_k{need}"] = float(
            estimate[warmup:][mask].mean() / cohort.services[warmup:][mask].mean()
        )
    return result


def replay(cohort, arrivals, estimates, discipline, load, scale, warmup):
    """Replay identical work, preserving the theoretical offered resource load."""
    rate = load * 4 / (float(np.sum(PROBABILITIES * NEEDS * MEANS)) * scale)
    trace = tuple(
        MsjTraceJob(float(a / rate), int(c), float(s), None if estimates is None else float(estimates[idx]))
        for idx, (a, c, s) in enumerate(zip(arrivals, cohort.classes, cohort.services))
    )
    sim = MsjGeneralSim(4, discipline)
    sim.set_servers(NEEDS)
    result = sim.run_trace(trace, warmup_jobs=warmup)
    output = {
        "mean_t": result.v[0],
        "p99_t": result.v_quantiles[0.99],
        "idle_with_queue": result.idle_with_queue,
        "utilization": result.utilization,
        "backfilled": result.backfilled,
        "reservations": result.reservations,
        "reservation_violations": result.reservation_violations,
        "arrival_rate": rate,
        "sample_offered_load": float(rate * np.mean(NEEDS[cohort.classes] * cohort.services) / 4),
    }
    for cls, need in enumerate(NEEDS):
        output[f"mean_t_k{need}"] = result.v_per_class[cls]
        output[f"p99_t_k{need}"] = result.v_quantiles_per_class[cls][0.99]
        promises = [(idx, time) for idx, time in result.reserved_start_times.items() if trace[idx].cls == cls]
        output[f"reservations_k{need}"] = len(promises)
        output[f"reservation_violations_k{need}"] = int(
            sum(result.start_times[idx] > time + 1e-10 * max(1.0, abs(time)) for idx, time in promises)
        )
    assert sum(output[f"reservation_violations_k{need}"] for need in NEEDS) == result.reservation_violations
    return output


def scarcity_sweep(model, calibration, test, warmup, sizes=CALIBRATION_SIZES):
    """Predict only: never delete unsupported jobs from a scheduler comparison.

    Sizes are nested prefixes of the same independent calibration cohort.
    This replaces the model's calibration state, not its regression fit.
    Null grouped coverage is unavailable, not zero and not successful coverage.
    """
    records = []
    for size in sizes:
        if size < 19 or size > len(calibration.services):
            raise ValueError("scarcity sizes must be between 19 and the full calibration count")
        cal = ObservedJobs(
            *(values[:size] for values in (calibration.features, calibration.classes, calibration.services))
        )
        model.calibrate(cal.features, cal.services)
        diagnostics = model.calibrate_by_group(cal.features, cal.services, NEEDS[cal.classes])
        pooled = model.predict(test.features, upper=True)
        groups = []
        for cls, need in enumerate(NEEDS):
            mask = test.classes[warmup:] == cls
            if not mask.any():
                raise ValueError("measured cohort must include all three resource classes")
            record = diagnostics.get(int(need))
            finite = record is not None and record.is_finite
            actual = test.services[warmup:][mask]
            bound = None
            if finite:
                bound = model.predict(test.features[warmup:][mask], upper=True, groups=np.full(mask.sum(), need))
            groups.append(
                {
                    "need": int(need),
                    "calibration": None if record is None else asdict(record),
                    "status": "finite" if finite else "unseen" if record is None else "insufficient",
                    "test_jobs": int(mask.sum()),
                    "pooled_coverage": float(np.mean(actual <= pooled[warmup:][mask])),
                    "grouped_coverage": None if bound is None else float(np.mean(actual <= bound)),
                    "grouped_estimate_ratio": None if bound is None else float(bound.mean() / actual.mean()),
                }
            )
        records.append(
            {
                "calibration_size": size,
                "calibration_fingerprint": fingerprint(cal),
                "all_groups_finite": all(g["status"] == "finite" for g in groups),
                "finite_job_fraction": sum(g["test_jobs"] for g in groups if g["status"] == "finite")
                / (len(test.services) - warmup),
                "groups": groups,
            }
        )
    return records


def summarize(rows, keys, metrics, mode_key, baseline_mode):
    """Student intervals across bundles and seed-matched differences to baseline."""
    summaries = []
    for key in sorted({tuple(row[name] for name in keys) for row in rows}):
        selected = [r for r in rows if tuple(r[name] for name in keys) == key]
        for mode in sorted({r[mode_key] for r in selected}):
            samples = sorted((r for r in selected if r[mode_key] == mode), key=lambda r: r["seed"])
            baseline = baseline_mode(mode)
            reference = sorted((r for r in selected if r[mode_key] == baseline), key=lambda r: r["seed"])
            assert [r["seed"] for r in samples] == [r["seed"] for r in reference]
            summaries.append(
                {
                    **dict(zip(keys, key)),
                    mode_key: mode,
                    "baseline": baseline,
                    "metrics": {name: interval([r[name] for r in samples]) for name in metrics},
                    "paired_delta": {
                        name: interval([r[name] - b[name] for r, b in zip(samples, reference)]) for name in metrics
                    },
                }
            )
    return summaries


def experiment(shape="lognormal_hetero", jobs=4000, replications=12):
    """Run the prespecified paired grid; calibration/test labels never select modes."""
    if shape not in SHAPES or jobs < 100 or replications < 2:
        raise ValueError("supported shape, at least 100 jobs and two replications required")
    warmup = jobs // 10
    runs, prediction_runs, calibration_runs, scarcity_runs = [], [], [], []
    for rep in range(replications):
        seed = 50000 + rep
        train, cal, test, arrivals = make_splits(shape, jobs + warmup, seed)
        model = fit_predictor(train, cal)
        estimates = forecasts(model, test.features, test.classes)
        calibration_runs.append(
            {
                "seed": seed,
                "pooled": asdict(model.calibration),
                "grouped": {str(group): asdict(info) for group, info in model.group_calibrations.items()},
                "fingerprints": {
                    name: fingerprint(data) for name, data in zip(("train", "calibration", "test"), (train, cal, test))
                },
            }
        )
        for regime, scale in REGIMES.items():
            cohort = ObservedJobs(test.features, test.classes, test.services * scale)
            modes = {**estimates, "oracle": cohort.services}
            for mode, forecast in modes.items():
                prediction_runs.append(
                    {"seed": seed, "regime": regime, "mode": mode, **forecast_metrics(forecast, cohort, warmup)}
                )
            for load in LOADS:
                for policy in POLICIES:
                    discipline, _, mode = policy.partition(":")
                    metrics = replay(cohort, arrivals, modes.get(mode), discipline, load, scale, warmup)
                    runs.append({"seed": seed, "regime": regime, "load": load, "policy": policy, **metrics})
        scarcity_runs.extend({"seed": seed, **row} for row in scarcity_sweep(model, cal, test, warmup))
    return {
        "protocol": {
            "shape": shape,
            "jobs_measured": jobs,
            "warmup": warmup,
            "replications": replications,
            "first_seed": 50000,
            "training_jobs": 1500,
            "calibration_jobs": 1000,
            "coverage_target": 0.95,
            "k": 4,
            "needs": NEEDS.tolist(),
            "class_probabilities": PROBABILITIES.tolist(),
            "service_means": MEANS.tolist(),
            "features": ["class=1", "class=2", "log_input_size ~ Normal(0,1)"],
            "feature_slope": FEATURE_SLOPE,
            "residual_cv": [0.8, 1.5, 2.0] if shape == "lognormal_hetero" else float(1 / np.sqrt(2)),
            "regimes": REGIMES,
            "loads": LOADS,
            "policies": POLICIES,
            "scarcity_sizes": CALIBRATION_SIZES,
            "split": "Four independent child streams for train/calibration/test/arrivals. Historical cohorts "
            "complete before time zero; fixed resource grouping, no test-label fitting or drift adaptation.",
            "note": "Synthetic finite cohorts; load matched by law, not by realized work. Student 95% intervals "
            "across independent seed bundles; paired differences to pooled95 within the same discipline. "
            "No multiplicity correction, stationarity or SLO guarantee. Per-class means/p99 exclude warmup; "
            "reservation counts include it. Scarcity is iid prediction-only: null means unavailable, "
            "not covered; no hidden fallback or job removal. Group coverage needs within-group exchangeability.",
        },
        "calibration_runs": calibration_runs,
        "prediction_runs": prediction_runs,
        "scarcity_runs": scarcity_runs,
        "replications": runs,
        "prediction_summaries": summarize(
            prediction_runs, ("regime",), PREDICTION_METRICS, "mode", lambda _: "pooled95"
        ),
        "summaries": summarize(
            runs,
            ("regime", "load"),
            QUEUE_METRICS,
            "policy",
            lambda policy: "fcfs" if policy == "fcfs" else policy.split(":")[0] + ":pooled95",
        ),
    }


def main():
    """Save strict JSON and print grouped minus pooled queue comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPES, default="lognormal_hetero")
    parser.add_argument("--jobs", type=int, default=4000)
    parser.add_argument("--replications", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment(args.shape, args.jobs, args.replications)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["prediction_summaries"]:
        print(row["regime"], row["mode"], {key: row["metrics"][key] for key in ("coverage", "coverage_k4")})
    for row in result["summaries"]:
        if row["policy"].endswith(":grouped95"):
            print(row["regime"], row["load"], row["policy"], row["paired_delta"])


if __name__ == "__main__":
    main()
