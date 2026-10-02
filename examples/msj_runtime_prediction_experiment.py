"""Leakage-separated runtime prediction and MSJ backfilling pilot (EPIC-049).

    python -m examples.msj_runtime_prediction_experiment --shape lognormal_hetero \
        --jobs 4000 --replications 6 --output works/msj_runtime_prediction/lognormal.json

Historical train/calibration cohorts finish before replay begins. No online
learning or current-job duration is available to non-oracle prediction modes.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from examples.msj_backfilling_experiment import interval
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.utils.runtime_prediction import LogLinearRuntimePredictor

NEEDS = np.array([1, 3, 4])
PROBABILITIES = np.array([0.5, 0.3, 0.2])
MEANS = np.array([0.4, 2.2, 2.2])
FEATURE_SLOPE = 0.7
SHAPES = ("erlang2", "lognormal_hetero")
MODES = ("class_mean", "feature_point", "feature_upper95", "oracle")
POLICIES = ("fcfs",) + tuple(f"{policy}:{mode}" for policy in ("easy", "conservative") for mode in MODES)
REGIMES = {"iid": 1.0, "slowdown2": 2.0}
LOADS = (0.35, 0.55)
QUEUE_METRICS = ("mean_t", "p99_t", "p99_wide", "idle_with_queue", "reservation_violations")


@dataclass(frozen=True)
class ObservedJobs:
    """Offline observations, kept separate from the prediction-only interface."""

    features: np.ndarray
    classes: np.ndarray
    services: np.ndarray


def sample_jobs(shape, count, rng):
    """Generate submission features first, then independent service noise."""
    if shape not in SHAPES or count < 1:
        raise ValueError("supported service shape and positive job count required")
    classes = rng.choice(3, count, p=PROBABILITIES)
    log_size = rng.normal(size=count)
    features = np.column_stack((classes == 1, classes == 2, log_size))
    conditional_means = MEANS[classes] * np.exp(FEATURE_SLOPE * log_size - FEATURE_SLOPE**2 / 2)
    if shape == "erlang2":
        unit_services = rng.gamma(2, 0.5, size=count)
    else:
        sigma = np.sqrt(np.log1p(np.array([0.8, 1.5, 2.0])[classes] ** 2))
        unit_services = rng.lognormal(-(sigma**2) / 2, sigma)
    return ObservedJobs(features, classes, conditional_means * unit_services)


def make_splits(shape, jobs, seed, training=1500, calibration=1000):
    """Use separate RNG child streams for train, calibration, test and arrivals."""
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]
    cohorts = (
        sample_jobs(shape, training, streams[0]),
        sample_jobs(shape, calibration, streams[1]),
        sample_jobs(shape, jobs, streams[2]),
    )
    return (*cohorts, np.cumsum(streams[3].exponential(size=jobs)))


def fingerprint(cohort):
    """Record byte-level cohort provenance for exact environment reproduction."""
    digest = hashlib.sha256()
    for values in (cohort.features, cohort.classes, cohort.services):
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def train_predictors(training, calibration):
    """Fit only historical training labels; reserve calibration for the margin."""
    class_means = np.array([training.services[training.classes == cls].mean() for cls in range(3)])
    if not np.all(np.isfinite(class_means)):
        raise ValueError("training must contain all three resource classes")
    model = LogLinearRuntimePredictor().fit(training.features, training.services)
    model.calibrate(calibration.features, calibration.services, coverage=0.95)
    return class_means, model


def predict_modes(class_means, model, features, classes):
    """Non-oracle path has no test-duration argument or trace access."""
    return {
        "class_mean": class_means[classes],
        "feature_point": model.predict(features),
        "feature_upper95": model.predict(features, upper=True),
    }


def prediction_metrics(estimates, cohort, warmup):
    """Evaluate only after prediction, on the same post-warmup arrival cohort."""
    estimates, services, classes = estimates[warmup:], cohort.services[warmup:], cohort.classes[warmup:]
    covered = services <= estimates
    return {
        "coverage": float(covered.mean()),
        "coverage_by_class": [float(covered[classes == cls].mean()) for cls in range(3)],
        "mean_absolute_log_error": float(np.abs(np.log(estimates) - np.log(services)).mean()),
        "mean_estimate_over_mean_service": float(estimates.mean() / services.mean()),
        "class_counts": [int(np.sum(classes == cls)) for cls in range(3)],
    }


def replay(cohort, unit_arrivals, estimates, policy, load, service_scale, warmup):
    """Keep resource load fixed from the known law, including test-time slowdown."""
    rate = load * 4 / (float(np.sum(PROBABILITIES * NEEDS * MEANS)) * service_scale)
    trace = tuple(
        MsjTraceJob(float(a / rate), int(c), float(s), None if estimates is None else float(estimates[idx]))
        for idx, (a, c, s) in enumerate(zip(unit_arrivals, cohort.classes, cohort.services))
    )
    sim = MsjGeneralSim(4, policy)
    sim.set_servers(NEEDS)
    result = sim.run_trace(trace, warmup_jobs=warmup)
    return {
        "mean_t": result.v[0],
        "p99_t": result.v_quantiles[0.99],
        "p99_wide": result.v_quantiles_per_class[2][0.99],
        "idle_with_queue": result.idle_with_queue,
        "utilization": result.utilization,
        "backfilled": result.backfilled,
        "reservations": result.reservations,
        "reservation_violations": result.reservation_violations,
        "arrival_rate": rate,
        "sample_offered_load": float(rate * np.mean(NEEDS[cohort.classes] * cohort.services) / 4),
    }


def experiment(shape="lognormal_hetero", jobs=4000, replications=6):
    """Run a fixed grid; the independent unit is the train/cal/test seed bundle."""
    if shape not in SHAPES or jobs < 100 or replications < 2:
        raise ValueError("supported shape, at least 100 jobs and two replications required")
    warmup = jobs // 10
    records, prediction_runs, calibration_runs, summaries, prediction_summaries = [], [], [], [], []
    for rep in range(replications):
        seed = 49000 + rep
        train, cal, test, arrivals = make_splits(shape, jobs + warmup, seed)
        class_means, model = train_predictors(train, cal)
        estimates = predict_modes(class_means, model, test.features, test.classes)
        calibration_runs.append(
            {
                "seed": seed,
                "training_samples": len(train.services),
                "class_means": class_means.tolist(),
                "calibration": asdict(model.calibration),
                "fingerprints": {
                    "train": fingerprint(train),
                    "calibration": fingerprint(cal),
                    "test": fingerprint(test),
                },
            }
        )
        for regime, scale in REGIMES.items():
            cohort = ObservedJobs(test.features, test.classes, test.services * scale)
            # Oracle is created in the benchmark only, never by predict_modes.
            forecasts = {**estimates, "oracle": cohort.services}
            for mode, forecast in forecasts.items():
                prediction_runs.append(
                    {"seed": seed, "regime": regime, "mode": mode, **prediction_metrics(forecast, cohort, warmup)}
                )
            for load in LOADS:
                for policy in POLICIES:
                    discipline, _, mode = policy.partition(":")
                    result = replay(cohort, arrivals, forecasts.get(mode), discipline, load, scale, warmup)
                    records.append({"seed": seed, "regime": regime, "load": load, "policy": policy, **result})
    for regime in REGIMES:
        for mode in MODES:
            rows = [r for r in prediction_runs if r["regime"] == regime and r["mode"] == mode]
            prediction_summaries.append(
                {
                    "regime": regime,
                    "mode": mode,
                    "metrics": {
                        key: interval([r[key] for r in rows])
                        for key in ("coverage", "mean_absolute_log_error", "mean_estimate_over_mean_service")
                    },
                    "coverage_by_class": [interval([r["coverage_by_class"][cls] for r in rows]) for cls in range(3)],
                }
            )
        for load in LOADS:
            samples = {
                policy: [r for r in records if r["regime"] == regime and r["load"] == load and r["policy"] == policy]
                for policy in POLICIES
            }
            for policy, rows in samples.items():
                baseline = "fcfs" if policy == "fcfs" else policy.split(":")[0] + ":class_mean"
                summaries.append(
                    {
                        "regime": regime,
                        "load": load,
                        "policy": policy,
                        "metrics": {key: interval([r[key] for r in rows]) for key in QUEUE_METRICS},
                        "baseline": baseline,
                        "paired_delta": {
                            key: interval([r[key] - b[key] for r, b in zip(rows, samples[baseline])])
                            for key in ("mean_t", "p99_t", "p99_wide", "reservation_violations")
                        },
                    }
                )
    return {
        "protocol": {
            "shape": shape,
            "jobs_measured": jobs,
            "warmup": warmup,
            "replications": replications,
            "first_seed": 49000,
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
            "split": "Four independent SeedSequence child streams: train, calibration, test, arrival gaps. "
            "Historical cohorts complete before time zero; no online updates or test-label fitting.",
            "note": "Synthetic finite cohorts; law-based load matching; no stationarity/SLO guarantee. "
            "95% Student intervals over independent seed bundles, not population quantile CIs. "
            "No multiplicity adjustment. Prediction coverage excludes warm-up; reservation counts include it. "
            "Coverage is marginal under exchangeability, not per-class or simultaneous; slowdown breaks it.",
        },
        "calibration_runs": calibration_runs,
        "prediction_runs": prediction_runs,
        "prediction_summaries": prediction_summaries,
        "replications": records,
        "summaries": summaries,
    }


def main():
    """Print prediction/queue outcomes and optionally save strict JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPES, default="lognormal_hetero")
    parser.add_argument("--jobs", type=int, default=4000)
    parser.add_argument("--replications", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment(args.shape, args.jobs, args.replications)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["prediction_summaries"]:
        print("coverage", row["regime"], row["mode"], row["metrics"]["coverage"])
    for row in result["summaries"]:
        print(
            row["regime"], row["load"], row["policy"], *(f"{row['metrics'][key]['mean']:.4f}" for key in QUEUE_METRICS)
        )


if __name__ == "__main__":
    main()
