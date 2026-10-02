"""EPIC-052: information-matched packing baselines on common general-service work.

    python -m examples.msj_packing_experiment --regime one_or_all \
        --shape lognormal_hetero --output works/msj_packing/one_or_all-lognormal.json

ServerFilling is a separately labelled zero-cost preemption reference, not a
nonpreemptive competitor. Unsupported policies are excluded by domain, not by
performance. Finite-cohort intervals do not prove stability or population SLOs.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from examples.msj_age_runtime_experiment import CensoredHistory, age_predictor, array_digest, initial_forecasts
from examples.msj_group_calibration_experiment import summarize
from examples.msj_runtime_prediction_experiment import FEATURE_SLOPE, LOADS, SHAPES, ObservedJobs, fingerprint
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.utils.residual_runtime import KaplanMeierRuntimeEstimator


@dataclass(frozen=True)
class Workload:
    """Prespecified resource/service law, independent of any sampled test jobs."""

    needs: tuple[int, ...]
    probabilities: tuple[float, ...]
    means: tuple[float, ...]
    noise_cv: tuple[float, ...]

    @property
    def mean_work(self) -> float:
        """Theoretical E[K*S], not a normalization by realized work."""
        return float(np.sum(np.array(self.needs) * self.probabilities * self.means))


WORKLOADS = {
    "one_or_all": Workload((1, 4), (0.8, 0.2), (0.4, 2.2), (0.8, 2.0)),
    "powers_of_two": Workload((1, 2, 4), (0.5, 0.3, 0.2), (0.4, 2.2, 2.2), (0.8, 1.5, 2.0)),
    "general": Workload((1, 3, 4), (0.5, 0.3, 0.2), (0.4, 2.2, 2.2), (0.8, 1.5, 2.0)),
}
BASE_POLICIES = ("fcfs", "first_fit", "msf", "adaptive_quickswap") + tuple(
    f"{policy}:{mode}" for policy in ("easy", "conservative") for mode in ("km_fixed", "km_age")
)
METRICS = (
    "mean_w",
    "mean_t",
    "weighted_mean_t",
    "p99_t",
    "idle_with_queue",
    "utilization",
    "preemptions",
    "preempted_jobs",
    "reservations",
    "reservation_violations",
    "promise_lateness_sum",
)


def policies_for(regime: str) -> tuple[str, ...]:
    """Declare eligibility before replay; never round resource needs."""
    if regime not in WORKLOADS:
        raise ValueError("unknown workload regime")
    extra = ("server_filling",) if regime != "general" else ()
    if regime == "one_or_all":
        extra += ("msfq:1", "msfq:3")
    return BASE_POLICIES + extra


def sample_jobs(workload: Workload, shape: str, count: int, rng: np.random.Generator) -> ObservedJobs:
    """Generate class, log-size feature, and independent mean-one service noise."""
    classes = rng.choice(len(workload.needs), count, p=workload.probabilities)
    log_size = rng.normal(size=count)
    means = np.array(workload.means)[classes] * np.exp(FEATURE_SLOPE * log_size - FEATURE_SLOPE**2 / 2)
    if shape == "erlang2":
        noise = rng.gamma(2, 0.5, count)
    elif shape == "lognormal_hetero":
        sigma = np.sqrt(np.log1p(np.array(workload.noise_cv)[classes] ** 2))
        noise = rng.lognormal(-(sigma**2) / 2, sigma)
    else:
        raise ValueError("unknown service shape")
    return ObservedJobs(log_size[:, None], classes, means * noise)


def make_bundle(regime: str, shape: str, count: int, seed: int) -> tuple:
    """Independent history, test, arrival, censor streams; no hidden S in history."""
    workload = WORKLOADS[regime]
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]
    latent = sample_jobs(workload, shape, 4000, streams[0])
    cohort = sample_jobs(workload, shape, count, streams[1])
    arrivals = np.cumsum(streams[2].exponential(size=count))
    exposure = streams[3].exponential(5 * np.array(workload.means)[latent.classes])
    history = CensoredHistory(latent.classes, np.minimum(latent.services, exposure), latent.services <= exposure)
    return history, cohort, arrivals


def fit_history(history: CensoredHistory, num_classes: int) -> list[KaplanMeierRuntimeEstimator]:
    """Fit only observable class-conditioned right-censored histories."""
    models = []
    for cls in range(num_classes):
        mask = history.classes == cls
        models.append(KaplanMeierRuntimeEstimator().fit(history.observed_times[mask], history.completed[mask]))
    return models


def replay(workload: Workload, trace: tuple[MsjTraceJob, ...], policy: str, warmup: int, models: list) -> dict:
    """Collect cohort/class metrics; preemption counters include warmup and drain."""
    discipline, _, mode = policy.partition(":")
    kwargs = {"msfq_threshold": int(mode)} if discipline == "msfq" else {}
    sim = MsjGeneralSim(4, discipline, **kwargs)
    sim.set_servers(workload.needs)
    predictor = age_predictor(models) if mode == "km_age" else None
    result = sim.run_trace(trace, warmup_jobs=warmup, remaining_predictor=predictor)
    if not all(result.counts_per_class):
        raise ValueError("measured cohort must include every resource class")
    weights = np.array(workload.probabilities) * workload.needs * workload.means / workload.mean_work
    promises = result.reserved_start_times
    lateness = [max(0.0, result.start_times[idx] - when) for idx, when in promises.items()]
    output = {
        "mean_w": result.w[0],
        "mean_t": result.v[0],
        "p99_t": result.v_quantiles[0.99],
        "weighted_mean_t": float(np.dot(weights, result.v_per_class)),
        "idle_with_queue": result.idle_with_queue,
        "utilization": result.utilization,
        "reservations": result.reservations,
        "reservation_violations": result.reservation_violations,
        "promise_lateness_sum": float(sum(lateness)),
        "promise_violation_rate": result.reservation_violations / result.reservations if result.reservations else None,
        "preemptions": result.preemptions,
        "preempted_jobs": sum(count > 0 for count in result.preemptions_per_job),
        "service_episodes": len(trace) + result.preemptions,
        "runtime_updates": result.runtime_updates,
        "unavailable_runtime_updates": result.unavailable_runtime_updates,
        "measured_jobs": sum(result.counts_per_class),
        "drained_jobs": len(result.completion_times),
        "schedule_hash": array_digest(np.array(result.start_times), np.array(result.completion_times)),
        "preemption_capability": discipline == "server_filling",
    }
    for cls, need in enumerate(workload.needs):
        output[f"mean_t_k{need}"] = result.v_per_class[cls]
        output[f"p99_t_k{need}"] = result.v_quantiles_per_class[cls][0.99]
        output[f"count_k{need}"] = result.counts_per_class[cls]
        class_promises = [(idx, when) for idx, when in promises.items() if trace[idx].cls == cls]
        output[f"reservations_k{need}"] = len(class_promises)
        output[f"reservation_violations_k{need}"] = int(
            sum(result.start_times[idx] > when + 1e-10 * max(1, abs(when)) for idx, when in class_promises)
        )
    return output


def experiment(
    regime: str = "one_or_all", shape: str = "lognormal_hetero", jobs: int = 4000, replications: int = 8
) -> dict:
    """Run the registered grid, retaining raw runs and paired FCFS/MSF contrasts."""
    if regime not in WORKLOADS or shape not in SHAPES or jobs < 100 or replications < 2:
        raise ValueError("supported regime/shape, at least 100 jobs and 2 replications required")
    workload, warmup = WORKLOADS[regime], jobs // 10
    runs, histories = [], []
    for rep in range(replications):
        seed = 52000 + rep
        history, cohort, unit_arrivals = make_bundle(regime, shape, jobs + warmup, seed)
        models = fit_history(history, len(workload.needs))
        forecasts = initial_forecasts({"km": models}, cohort.classes)["km"]
        histories.append(
            {
                "seed": seed,
                "history_hash": array_digest(history.classes, history.observed_times, history.completed),
                "test_hash": fingerprint(cohort),
                "unit_arrivals_hash": array_digest(unit_arrivals),
                "forecasts_hash": array_digest(forecasts),
                "completed_fraction": float(history.completed.mean()),
                "initial_quantiles": [model.remaining_quantile(0, 0.9) for model in models],
                "test_coverage_by_class": [
                    float(
                        np.mean(
                            forecasts[warmup:][cohort.classes[warmup:] == cls]
                            >= cohort.services[warmup:][cohort.classes[warmup:] == cls]
                        )
                    )
                    for cls in range(len(workload.needs))
                ],
            }
        )
        for load in LOADS:
            rate = load * 4 / workload.mean_work
            trace = tuple(
                MsjTraceJob(float(a / rate), int(c), float(s), float(e))
                for a, c, s, e in zip(unit_arrivals, cohort.classes, cohort.services, forecasts)
            )
            for policy in policies_for(regime):
                metrics = replay(workload, trace, policy, warmup, models)
                runs.append(
                    {
                        "seed": seed,
                        "load": load,
                        "policy": policy,
                        "arrival_rate": rate,
                        "sample_offered_load": float(
                            rate * np.mean(np.array(workload.needs)[cohort.classes] * cohort.services) / 4
                        ),
                        **metrics,
                    }
                )
    metrics = METRICS + tuple(f"{metric}_k{need}" for metric in ("mean_t", "p99_t") for need in workload.needs)
    return {
        "protocol": {
            "regime": regime,
            "shape": shape,
            "k": 4,
            **asdict(workload),
            "mean_work": workload.mean_work,
            "feature_slope": FEATURE_SLOPE,
            "loads": LOADS,
            "policies": policies_for(regime),
            "jobs_measured": jobs,
            "warmup": warmup,
            "replications": replications,
            "first_seed": 52000,
            "history_jobs": 4000,
            "quantile_target": 0.9,
            "censor_mean_multiple": 5,
            "note": "Synthetic log-size mixture, not marginal Erlang. Independent history/test/arrival/censor streams. "
            "History is observable before replay in a separate pool; C independent of S within class. "
            "Same test work, arrivals and initial forecasts for every policy. Packing ignores forecasts. "
            "No oracle or test-tuned thresholds. ServerFilling is zero-cost preemptive-resume, powers of two only. "
            "MSFQ is the literal four-phase version, inclusive threshold and initial small batch. "
            "W includes all pauses; weighted T uses theoretical class load weights. "
            "Latency excludes warmup; counters include warmup/drain; time averages exclude drain. "
            "No reservations means null violation rate, not a promise guarantee. All jobs are drained. "
            "Student 95% intervals across seeds; paired within seed, no multiplicity adjustment. "
            "No claim of stability, starvation bounds, general-service throughput optimality or real-trace validation.",
        },
        "history_runs": histories,
        "replications": runs,
        "summaries": summarize(runs, ("load",), metrics, "policy", lambda _: "fcfs"),
        "msf_contrasts": summarize(runs, ("load",), metrics, "policy", lambda _: "msf"),
    }


def main() -> None:
    """Write strict JSON and print compact paired mean-time contrasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=WORKLOADS, default="one_or_all")
    parser.add_argument("--shape", choices=SHAPES, default="lognormal_hetero")
    parser.add_argument("--jobs", type=int, default=4000)
    parser.add_argument("--replications", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = experiment(args.regime, args.shape, args.jobs, args.replications)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in result["summaries"]:
        print(row["load"], row["policy"], row["paired_delta"]["mean_t"])


if __name__ == "__main__":
    main()
