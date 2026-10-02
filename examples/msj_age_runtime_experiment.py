"""Censored history and age-aware MSJ predictions: EPIC-051 paired pilot.

    python -m examples.msj_age_runtime_experiment --shape lognormal_hetero \
        --output works/msj_age_runtime/lognormal.json

The prespecified data law is inherited from EPIC-049, but new independent
history/test bundles are used. Only class and observed age enter predictions.
"""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.msj_backfilling_experiment import interval
from examples.msj_group_calibration_experiment import summarize
from examples.msj_runtime_prediction_experiment import (
    FEATURE_SLOPE,
    LOADS,
    MEANS,
    NEEDS,
    PROBABILITIES,
    SHAPES,
    fingerprint,
    sample_jobs,
)
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob, RemainingPredictor
from most_queue.sim.utils.residual_runtime import KaplanMeierRuntimeEstimator

QUANTILE = 0.9
MODES = ("km_fixed", "km_age", "completed_fixed", "completed_age", "oracle")
POLICIES = ("fcfs",) + tuple(f"{policy}:{mode}" for policy in ("easy", "conservative") for mode in MODES)
QUEUE_METRICS = (
    "mean_t",
    "p99_t",
    "idle_with_queue",
    "utilization",
    "backfilled",
    "reservations",
    "reservation_violations",
    "promise_lateness_sum",
    "runtime_updates",
    "unavailable_runtime_updates",
    "forecast_calendar_resets",
) + tuple(f"{metric}_k{need}" for metric in ("mean_t", "p99_t") for need in NEEDS)
LANDMARK_METRICS = ("quantile", "coverage", "restricted_mean", "empirical_restricted_mean", "restricted_mean_error")


@dataclass(frozen=True)
class CensoredHistory:
    """Available historical observations; unobserved true runtimes are absent."""

    classes: np.ndarray
    observed_times: np.ndarray
    completed: np.ndarray


@dataclass(frozen=True)
class ReplayOptions:
    """Explicit scheduling and measurement options for one common-work replay."""

    discipline: str
    load: float
    warmup: int
    predictor: RemainingPredictor | None = None


def array_digest(*arrays):
    """Fingerprint exact generated arrays for same-environment reproduction."""
    digest = hashlib.sha256()
    for values in arrays:
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def make_bundle(shape, jobs, seed, history_jobs=4000):
    """Independent history/test/arrival/exposure streams, all history known at zero.

    Historical starts can be placed at -C: S<=C finishes before replay, S>C is
    still censored at zero. These historical jobs are not replay-pool occupants.
    C is independent of S conditional on class; it does not depend on size.
    """
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]
    latent = sample_jobs(shape, history_jobs, streams[0])
    test = sample_jobs(shape, jobs, streams[1])
    arrivals = np.cumsum(streams[2].exponential(size=jobs))
    exposure = streams[3].exponential(5 * MEANS[latent.classes])
    history = CensoredHistory(latent.classes, np.minimum(latent.services, exposure), latent.services <= exposure)
    return history, test, arrivals


def fit_models(history):
    """Fit censoring-aware and intentionally biased completed-only class curves."""
    models = {"km": [], "completed": []}
    for cls in range(len(NEEDS)):
        mask = history.classes == cls
        models["km"].append(KaplanMeierRuntimeEstimator().fit(history.observed_times[mask], history.completed[mask]))
        finished = history.observed_times[mask & history.completed]
        models["completed"].append(KaplanMeierRuntimeEstimator().fit(finished, np.ones(finished.size, dtype=bool)))
    return models


def initial_forecasts(models, classes):
    """Require finite class quantiles; no silent fallback or deletion of jobs."""
    estimates = {}
    for name, curves in models.items():
        quantiles = [curve.remaining_quantile(0, QUANTILE) for curve in curves]
        if any(q is None for q in quantiles):
            raise ValueError("initial class quantile unavailable; the declared replay comparison cannot run")
        estimates[name] = np.asarray(quantiles)[classes]
    return estimates


def landmark_metrics(models, cohort, warmup):
    """Evaluate residuals on held-out survivors at fixed, scheduler-free ages."""
    rows = []
    for name, curves in models.items():
        for cls, need in enumerate(NEEDS):
            services = cohort.services[warmup:][cohort.classes[warmup:] == cls]
            for multiple in (0, 1, 3):
                age, horizon = multiple * MEANS[cls], 10 * MEANS[cls]
                residual = services[services > age] - age
                quantile = curves[cls].remaining_quantile(age, QUANTILE)
                restricted = curves[cls].remaining_mean(age, horizon)
                empirical = float(np.minimum(residual, horizon - age).mean()) if residual.size else None
                rows.append(
                    {
                        "model": name,
                        "need": int(need),
                        "age_multiple": multiple,
                        "age": float(age),
                        "horizon": float(horizon),
                        "survivors": int(residual.size),
                        "quantile": quantile,
                        "quantile_available": quantile is not None,
                        "coverage": (
                            float(np.mean(residual <= quantile)) if residual.size and quantile is not None else None
                        ),
                        "full_mean": curves[cls].remaining_mean(age),
                        "restricted_mean": restricted,
                        "empirical_restricted_mean": empirical,
                        "restricted_mean_error": (
                            None if restricted is None or empirical is None else restricted - empirical
                        ),
                    }
                )
    return rows


def replay(cohort, arrivals, estimates, options):
    """Drain the identical arrival cohort; counters/promises include warmup."""
    rate = options.load * 4 / float(np.sum(PROBABILITIES * NEEDS * MEANS))
    trace = tuple(
        MsjTraceJob(float(a / rate), int(c), float(s), None if estimates is None else float(estimates[idx]))
        for idx, (a, c, s) in enumerate(zip(arrivals, cohort.classes, cohort.services))
    )
    sim = MsjGeneralSim(4, options.discipline)
    sim.set_servers(NEEDS)
    result = sim.run_trace(trace, warmup_jobs=options.warmup, remaining_predictor=options.predictor)
    promises = result.reserved_start_times
    lateness = [max(0.0, result.start_times[idx] - when) for idx, when in promises.items()]
    output = {
        "mean_t": result.v[0],
        "p99_t": result.v_quantiles[0.99],
        "idle_with_queue": result.idle_with_queue,
        "utilization": result.utilization,
        "backfilled": result.backfilled,
        "reservations": result.reservations,
        "reservation_violations": result.reservation_violations,
        "promise_lateness_sum": float(sum(lateness)),
        "promise_lateness_max": float(max(lateness, default=0)),
        "promised_wait_sum": float(sum(when - trace[idx].arrival for idx, when in promises.items())),
        "runtime_updates": result.runtime_updates,
        "unavailable_runtime_updates": result.unavailable_runtime_updates,
        "forecast_calendar_resets": result.forecast_calendar_resets,
        "measured_jobs": sum(result.counts_per_class),
        "drained_jobs": len(result.completion_times),
        "arrival_rate": rate,
        "sample_offered_load": float(rate * np.mean(NEEDS[cohort.classes] * cohort.services) / 4),
    }
    for cls, need in enumerate(NEEDS):
        output[f"mean_t_k{need}"] = result.v_per_class[cls]
        output[f"p99_t_k{need}"] = result.v_quantiles_per_class[cls][0.99]
    return output


def landmark_summaries(rows):
    """Report availability explicitly; intervals use available independent seeds.

    Conditional-on-availability averages do not treat missing forecasts as zero
    or successful coverage. With fewer than two available seeds no CI is given.
    """
    summaries = []
    keys = ("model", "need", "age_multiple")
    for key in sorted({tuple(row[k] for k in keys) for row in rows}):
        samples = [r for r in rows if tuple(r[k] for k in keys) == key]
        metrics = {}
        for name in LANDMARK_METRICS:
            values = [r[name] for r in samples if r[name] is not None]
            metrics[name] = {
                "available_seeds": len(values),
                "mean": float(np.mean(values)) if values else None,
                "ci95": interval(values) if len(values) >= 2 else None,
            }
        summaries.append({**dict(zip(keys, key)), "seeds": len(samples), "metrics": metrics})
    return summaries


def baseline(policy):
    """Age minus same-estimator fixed; other backfills minus KM fixed."""
    if policy == "fcfs":
        return policy
    if policy.endswith("_age"):
        return policy[:-4] + "_fixed"
    return policy.split(":")[0] + ":km_fixed"


def age_predictor(curves):
    """Bind historical class curves without giving the callback future labels."""

    def predict(cls, age):
        return curves[cls].remaining_quantile(age, QUANTILE)

    return predict


def experiment(shape="lognormal_hetero", jobs=4000, replications=8):
    """Run the declared 176-replay grid per shape without using test labels to fit."""
    if shape not in SHAPES or jobs < 100 or replications < 2:
        raise ValueError("supported shape, at least 100 jobs and two replications required")
    warmup = jobs // 10
    runs, landmarks, histories = [], [], []
    for rep in range(replications):
        seed = 51000 + rep
        history, test, arrivals = make_bundle(shape, jobs + warmup, seed)
        models = fit_models(history)
        initial = initial_forecasts(models, test.classes)
        landmarks.extend({"seed": seed, **row} for row in landmark_metrics(models, test, warmup))
        histories.append(
            {
                "seed": seed,
                "fingerprints": {
                    "observed_history": array_digest(history.classes, history.observed_times, history.completed),
                    "test": fingerprint(test),
                    "unit_arrivals": array_digest(arrivals),
                    **{name + "_initial": array_digest(values) for name, values in initial.items()},
                },
                "classes": [
                    {
                        "need": int(need),
                        "observations": int(np.sum(history.classes == cls)),
                        "completed": int(np.sum((history.classes == cls) & history.completed)),
                        "km_terminal_survival": models["km"][cls].curve[-1].survival,
                        "last_observed_time": models["km"][cls].curve[-1].time,
                        "km_initial_quantile": models["km"][cls].remaining_quantile(0, QUANTILE),
                        "completed_initial_quantile": models["completed"][cls].remaining_quantile(0, QUANTILE),
                    }
                    for cls, need in enumerate(NEEDS)
                ],
            }
        )
        for load in LOADS:
            for policy in POLICIES:
                discipline, _, mode = policy.partition(":")
                predictor = None
                if mode.endswith("_age"):
                    predictor = age_predictor(models[mode.split("_")[0]])
                estimates = test.services if mode == "oracle" else initial.get(mode.split("_")[0])
                metrics = replay(test, arrivals, estimates, ReplayOptions(discipline, load, warmup, predictor))
                runs.append({"seed": seed, "load": load, "policy": policy, **metrics})
    return {
        "protocol": {
            "shape": shape,
            "jobs_measured": jobs,
            "warmup": warmup,
            "replications": replications,
            "first_seed": 51000,
            "history_jobs": 4000,
            "quantile_target": QUANTILE,
            "k": 4,
            "needs": NEEDS.tolist(),
            "class_probabilities": PROBABILITIES.tolist(),
            "service_means": MEANS.tolist(),
            "feature_slope": FEATURE_SLOPE,
            "residual_cv": [0.8, 1.5, 2.0] if shape == "lognormal_hetero" else float(1 / np.sqrt(2)),
            "loads": LOADS,
            "policies": POLICIES,
            "landmark_age_multiples": [0, 1, 3],
            "restricted_horizon_multiple": 10,
            "censor_mean_multiple": 5,
            "unavailable_update_semantics": "None or a positive residual below floating-point timestamp resolution",
            "split": "Independent SeedSequence streams for history, test, unit arrival gaps, censoring. "
            "Historical starts at -C make min(S,C) and S<=C known at zero, before replay; "
            "historical jobs are a separate pool. C independent of S within class.",
            "note": "Same EPIC-049 service law including log-size mixture, but only class conditions KM. "
            "Fixed and age modes share initial forecasts. Refresh uses class/elapsed service only at "
            "existing events; None suspends new backfills. Historical best promises survive calendar resets. "
            "Counter and promise sums include warmup; latency and landmarks exclude it. "
            "Promise sums/max are zero for no reservations. All jobs are drained. "
            "Load matched by law, not by realized work. Plug-in quantiles, not coverage bounds. "
            "Student 95% CIs across seeds, paired within seed; no multiplicity adjustment or SLO/stability claim. "
            "Landmark CIs condition on availability, which is separately reported; null is never coverage.",
        },
        "history_runs": histories,
        "landmark_runs": landmarks,
        "landmark_summaries": landmark_summaries(landmarks),
        "replications": runs,
        "summaries": summarize(runs, ("load",), QUEUE_METRICS, "policy", baseline),
    }


def main():
    """Save strict JSON and print paired KM-age minus KM-fixed comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=SHAPES, default="lognormal_hetero")
    parser.add_argument("--jobs", type=int, default=4000)
    parser.add_argument("--replications", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment(args.shape, args.jobs, args.replications)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["summaries"]:
        if row["policy"].endswith(":km_age"):
            print(row["load"], row["policy"], row["paired_delta"])


if __name__ == "__main__":
    main()
