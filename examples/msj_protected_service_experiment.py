"""EPIC-054: useful-service protection, selected on independent offline work.

    python -m examples.msj_protected_service_experiment --regime one_or_all \
        --shape erlang2 --cost 0.2 --output works/msj_protected_service/one_or_all-erlang2-0.2.json

Selection uses only four completed tuning traces, never a test schedule. Tuned
rows alias the chosen fixed candidate: they are not extra scheduler executions.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from examples.msj_age_runtime_experiment import age_predictor, array_digest, initial_forecasts
from examples.msj_checkpoint_experiment import METRICS, REGIMES, checkpoint_metrics
from examples.msj_group_calibration_experiment import summarize
from examples.msj_packing_experiment import WORKLOADS, Workload, fit_history, make_bundle
from examples.msj_runtime_prediction_experiment import FEATURE_SLOPE, LOADS, SHAPES, fingerprint
from most_queue.sim.msj_checkpoint import MsjCheckpointSim
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob

FACTORS = (0, 1, 4, 16)
COST_LEVELS = (0.05, 0.2, 1.0)
CANDIDATES = tuple(f"sf:h{factor}" for factor in FACTORS)
BASELINE = CANDIDATES[0]
POLICIES = ("first_fit", "msf", "easy:km_age", "zero_cost_sf", *CANDIDATES)
DIAGNOSTICS = ("protected_preemptions", "protection_expirations")


@dataclass(frozen=True)
class StudySize:
    """Independent tuning/test replication sizes; defaults define the pilot."""

    jobs: int = 4000
    replications: int = 8
    tuning_jobs: int = 2000
    tuning_replications: int = 4

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            minimum = 100 if name.endswith("jobs") else 2
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.tuning_replications > 100:
            raise ValueError("tuning seed range must not overlap test seeds")


@dataclass(frozen=True)
class ReplaySettings:
    """Observable costs, fixed forecast models and the measurement cohort."""

    cost_factor: float
    warmup: int
    models: list


def replay(workload: Workload, trace: tuple[MsjTraceJob, ...], policy: str, settings: ReplaySettings) -> dict:
    """Replay a declared policy using episode age and costs, never future S."""
    c = r = q = 0.0
    factor, predictor = None, None
    if policy in CANDIDATES:
        factor = FACTORS[CANDIDATES.index(policy)]
        total_cost = settings.cost_factor * float(np.dot(workload.probabilities, workload.means))
        c = r = total_cost / 2
        q = factor * total_cost
        sim = MsjCheckpointSim(4, c, r, min_service_time=q)
    elif policy == "zero_cost_sf":
        sim = MsjCheckpointSim(4)
    elif policy in POLICIES:
        sim = MsjGeneralSim(4, policy.partition(":")[0])
        predictor = age_predictor(settings.models) if policy == "easy:km_age" else None
    else:
        raise ValueError("unknown policy")
    sim.set_servers(workload.needs)
    result = sim.run_trace(trace, settings.warmup, remaining_predictor=predictor)
    return {
        "factor": factor,
        "min_service_time": q,
        "checkpoint_time": c,
        "resume_time": r,
        **checkpoint_metrics(workload, trace, result, settings.warmup),
        **{name: getattr(result, name, 0) for name in DIAGNOSTICS},
    }


def _run_bundle(scenario, seed, jobs, policies):
    regime, shape, cost_factor = scenario
    workload, warmup = WORKLOADS[regime], jobs // 10
    history, cohort, arrivals = make_bundle(regime, shape, jobs + warmup, seed)
    models = fit_history(history, len(workload.needs))
    forecasts = initial_forecasts({"km": models}, cohort.classes)["km"]
    provenance = {
        "seed": seed,
        "history_hash": array_digest(history.classes, history.observed_times, history.completed),
        "test_hash": fingerprint(cohort),
        "unit_arrivals_hash": array_digest(arrivals),
        "forecasts_hash": array_digest(forecasts),
        "completed_fraction": float(history.completed.mean()),
        "initial_quantiles": [model.remaining_quantile(0, 0.9) for model in models],
    }
    runs = []
    for load in LOADS:
        rate = load * 4 / workload.mean_work
        trace = tuple(
            MsjTraceJob(float(a / rate), int(c), float(s), float(e))
            for a, c, s, e in zip(arrivals, cohort.classes, cohort.services, forecasts)
        )
        settings = ReplaySettings(cost_factor, warmup, models)
        for policy in policies:
            runs.append(
                {
                    "seed": seed,
                    "load": load,
                    "policy": policy,
                    "arrival_rate": rate,
                    "sample_offered_load": float(
                        rate * np.mean(np.array(workload.needs)[cohort.classes] * cohort.services) / 4
                    ),
                    **replay(workload, trace, policy, settings),
                }
            )
    return provenance, runs


def select_policy(tuning_runs: list[dict]) -> list[dict]:
    """Minimize mean tuning weighted T; break exact ties by smaller h.

    The function receives no test data. All candidates must use identical tuning
    seeds at each load; unpaired or duplicate rows are rejected.
    """
    choices = []
    for load in LOADS:
        samples = [[r for r in tuning_runs if r["load"] == load and r["policy"] == p] for p in CANDIDATES]
        seeds = [sorted(r["seed"] for r in rows) for rows in samples]
        if len(seeds[0]) < 2 or len(set(seeds[0])) != len(seeds[0]) or any(s != seeds[0] for s in seeds):
            raise ValueError("selection requires unique, matched tuning seeds for every candidate/load")
        scores = {p: float(np.mean([r["weighted_mean_t"] for r in rows])) for p, rows in zip(CANDIDATES, samples)}
        if not all(np.isfinite(score) for score in scores.values()):
            raise ValueError("nonfinite tuning objective")
        policy = min((scores[p], index, p) for index, p in enumerate(CANDIDATES))[2]
        choices.append({"load": load, "policy": policy, "factor": FACTORS[CANDIDATES.index(policy)], "scores": scores})
    return choices


def experiment(
    regime: str = "powers_of_two",
    shape: str = "lognormal_hetero",
    cost_factor: float = 0.2,
    *,
    size: StudySize = StudySize(),
) -> dict:
    """Tune independently, freeze the choice, then evaluate all held-out modes."""
    if regime not in REGIMES or shape not in SHAPES or isinstance(cost_factor, bool) or cost_factor not in COST_LEVELS:
        raise ValueError("unsupported resource regime, service shape or cost factor")
    scenario = regime, shape, cost_factor
    tuning_history, tuning_runs, history_runs, runs = [], [], [], []
    for seed in range(54000, 54000 + size.tuning_replications):
        history, rows = _run_bundle(scenario, seed, size.tuning_jobs, CANDIDATES)
        tuning_history.append(history)
        tuning_runs.extend(rows)
    selection = select_policy(tuning_runs)
    for seed in range(54100, 54100 + size.replications):
        history, rows = _run_bundle(scenario, seed, size.jobs, POLICIES)
        history_runs.append(history)
        for row in rows:
            runs.append({**row, "derived": False})
        for choice in selection:
            chosen = next(r for r in rows if r["load"] == choice["load"] and r["policy"] == choice["policy"])
            runs.append({**chosen, "policy": "tuned", "selected_policy": choice["policy"], "derived": True})
    workload = WORKLOADS[regime]
    metrics = (
        METRICS + DIAGNOSTICS + tuple(f"{metric}_k{need}" for metric in ("mean_t", "p99_t") for need in workload.needs)
    )
    return {
        "protocol": {
            "regime": regime,
            "shape": shape,
            "k": 4,
            **asdict(workload),
            **asdict(size),
            "cost_factor": cost_factor,
            "protection_factors": FACTORS,
            "policies": (*POLICIES, "tuned"),
            "feature_slope": FEATURE_SLOPE,
            "loads": LOADS,
            "mean_work": workload.mean_work,
            "mean_service": float(np.dot(workload.probabilities, workload.means)),
            "tuning_first_seed": 54000,
            "test_first_seed": 54100,
            "history_jobs": 4000,
            "warmup": size.jobs // 10,
            "tuning_warmup": size.tuning_jobs // 10,
            "scheduler_runs": size.tuning_replications * 8 + size.replications * 16,
            "note": "Same service law and independent streams as EPIC-053. q=h*(c+r) uses known constant costs, "
            "not hidden S. Protection resets at each useful start AFTER resume, not on overhead or arrival. "
            "Eligible jobs alone may preempt; overhead gate and original prefix retained. Expiries are real "
            "review events, not mandatory switches. Protected attempt counters are not saved preemptions. "
            "Select h by mean weighted T on separate fully completed tuning traces, ties choose smaller h. "
            "Selection frozen before test data are generated; tuned copies one candidate with no extra replay. "
            "Censored KM history for EASY is separate from offline tuning traces. Useful lambda is fixed, "
            "not renormalized for costs; every mode shares future work/estimates. No I/O/memory/lost work. "
            "Latency excludes arrival warmup; time averages exclude drain; counters/resource-time include both. "
            "Finite drain/backlog/throughput do not prove stability. Student 95% CIs are conditional on this "
            "tuning history, not retraining uncertainty, and have no multiplicity correction. "
            "No claimed optimality, real-trace calibration or Slurm emulation.",
        },
        "tuning_history": tuning_history,
        "tuning_runs": tuning_runs,
        "selection": selection,
        "history_runs": history_runs,
        "replications": runs,
        "unprotected_contrasts": summarize(runs, ("load",), metrics, "policy", lambda _: BASELINE),
        "first_fit_contrasts": summarize(runs, ("load",), metrics, "policy", lambda _: "first_fit"),
        "msf_contrasts": summarize(runs, ("load",), metrics, "policy", lambda _: "msf"),
    }


def main() -> None:
    """Write strict JSON and report held-out tuned-minus-unprotected delays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, default="powers_of_two")
    parser.add_argument("--shape", choices=SHAPES, default="lognormal_hetero")
    parser.add_argument("--cost", type=float, choices=COST_LEVELS, default=0.2)
    parser.add_argument("--jobs", type=int, default=4000)
    parser.add_argument("--replications", type=int, default=8)
    parser.add_argument("--tuning-jobs", type=int, default=2000)
    parser.add_argument("--tuning-replications", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    size = StudySize(args.jobs, args.replications, args.tuning_jobs, args.tuning_replications)
    result = experiment(args.regime, args.shape, args.cost, size=size)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for choice in result["selection"]:
        row = next(r for r in result["unprotected_contrasts"] if r["load"] == choice["load"] and r["policy"] == "tuned")
        print(choice["load"], "selected", choice["policy"], row["paired_delta"]["weighted_mean_t"])


if __name__ == "__main__":
    main()
