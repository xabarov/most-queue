"""EPIC-055: pinned SWF history, temporal holdout and common-workload replay.

Run from the repository root:
    python -m examples.real_trace_calibration_experiment --download \
        --output works/real_trace_calibration/sdsc-sp2.json

Raw data stays in .cache, retains its original notice and is not redistributed
under the package's MIT license. See docs/real_trace_calibration.md.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import scipy
from scipy.stats import t as student_t

from most_queue.random.service_calibration import FAMILIES, ServiceCalibration
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.utils.workload_trace import chronological_split, parse_swf, positive_integer

SOURCE_PAGE = "https://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/index.html"
MIRROR_COMMIT = "cd433e3d62a01705f255bd48cb55b393862800e7"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/DIR-LAB/deep-batch-scheduler/{MIRROR_COMMIT}" "/data/SDSC-SP2-1998-4.2-cln.swf"
)
SOURCE_SHA256 = "64bc6c621c97fe10e74cfc6a8c568ba544fddd99c4f810473027dca99efc2ed1"
POLICIES = ("fcfs", "first_fit", "msf", "adaptive_quickswap", "easy", "conservative")
METRICS = ("mean_w", "mean_t", "p95_t", "p99_t", "node_weighted_mean_t", "utilization", "idle_with_queue")
GROUP_UPPER = (1, 8, 32)


@dataclass(frozen=True)
class ExperimentConfig:
    """Prespecified protocol; warmup is additional to measured jobs per block."""

    jobs: int = 1000
    warmup: int = 200
    blocks: int = 3
    replications: int = 8
    first_seed: int = 55000
    train_fraction: float = 0.6

    def __post_init__(self):
        for name in ("jobs", "blocks", "replications", "first_seed", "warmup"):
            positive_integer(getattr(self, name), name, minimum=0 if name in ("warmup", "first_seed") else 1)
        if self.replications < 2:
            raise ValueError("at least two replications are required for Monte Carlo intervals")
        if not np.isfinite(self.train_fraction) or not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0,1)")


def verified_source(path: Path, download=False) -> bytes:
    """Read or explicitly download one pinned file; reject corruption/HTML.

    Existing files are never overwritten. A failed hash check does not create
    a cache entry. HTTPS source, size cap and timeout are fixed in code.
    """
    if path.exists():
        payload = path.read_bytes()
    elif download:
        with urlopen(SOURCE_URL, timeout=45) as response:
            payload = response.read(8_000_001)
        if len(payload) > 8_000_000:
            raise ValueError("source exceeds the download size limit")
    else:
        raise ValueError("source not cached; use --download after reviewing the source usage conditions")
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("SWF SHA-256 mismatch; refusing unverified data")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return payload


def resource_groups(needs):
    """Return fixed groups 1, 2..8, 9..32, >=33 without rounding K."""
    return np.searchsorted(GROUP_UPPER, needs, side="left")


def fit_history(training):
    """Use only the supplied completed history; sparse groups use pooled data."""
    pooled = ServiceCalibration.fit([job.runtime for job in training])
    groups = resource_groups([job.need for job in training])
    models, audit = [], []
    for group in range(4):
        values = [job.runtime for job, label in zip(training, groups) if label == group]
        fallback = len(values) < 20
        model = pooled if fallback else ServiceCalibration.fit(values)
        models.append(model)
        audit.append({"group": group, "group_count": len(values), "pooled_fallback": fallback, **model.parameters()})
    return models, audit


def fingerprint(rows):
    """Hash numerical inputs in a platform-independent byte representation."""
    return hashlib.sha256(np.asarray(rows, dtype="<f8").tobytes()).hexdigest()


def make_trace(jobs, services, models):
    """Keep arrivals and exact K; estimate from historical group p90 only."""
    needs = tuple(sorted({job.need for job in jobs}))
    classes = {need: idx for idx, need in enumerate(needs)}
    forecasts = [model.parameters()["p90"] for model in models]
    groups = resource_groups([job.need for job in jobs])
    origin = jobs[0].submit
    trace = tuple(
        MsjTraceJob(job.submit - origin, classes[job.need], float(service), forecasts[group])
        for job, service, group in zip(jobs, services, groups)
    )
    return needs, trace


def replay(capacity, needs, trace, policy, warmup):
    """Use the existing scheduler and extract finite-cohort/group metrics."""
    simulator = MsjGeneralSim(capacity, policy)
    simulator.set_servers(needs)
    result = simulator.run_trace(trace, warmup_jobs=warmup)
    weights = np.array([needs[job.cls] for job in trace[warmup:]])
    sojourns = np.array(result.sojourn_samples)
    labels = resource_groups(weights)
    groups = []
    for group in range(4):
        times = sojourns[labels == group]
        groups.append(
            {
                "count": int(times.size),
                "mean_t": float(times.mean()) if times.size else None,
                "p99_t": float(np.quantile(times, 0.99)) if times.size else None,
            }
        )
    return {
        "mean_w": result.w[0],
        "mean_t": result.v[0],
        "p95_t": result.v_quantiles[0.95],
        "p99_t": result.v_quantiles[0.99],
        "node_weighted_mean_t": float(np.average(sojourns, weights=weights)),
        "utilization": result.utilization,
        "idle_with_queue": result.idle_with_queue,
        "groups": groups,
        "reservation_violations": result.reservation_violations,
    }


def interval(values):
    """Exploratory 95% t interval for conditional Monte Carlo mean only."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "low": mean - half, "high": mean + half}


def summarize(rows):
    """Report conditional errors, paired contrasts and policy-choice regret."""
    summaries, decisions = [], []
    for block in sorted({row["block"] for row in rows}):
        observed = {row["policy"]: row for row in rows if row["block"] == block and row["family"] == "observed"}
        for family in FAMILIES:
            samples = {
                policy: sorted(
                    [
                        row
                        for row in rows
                        if row["block"] == block and row["family"] == family and row["policy"] == policy
                    ],
                    key=lambda row: row["seed"],
                )
                for policy in POLICIES
            }
            for policy in POLICIES:
                metrics = {}
                for metric in METRICS:
                    ref = observed[policy][metric]
                    if ref is None:
                        metrics[metric] = None
                        continue
                    estimate = interval([row[metric] for row in samples[policy]])
                    estimate["reference"] = ref
                    estimate["relative_error"] = (estimate["mean"] - ref) / ref if ref else None
                    estimate["difference_from_fcfs"] = interval(
                        [row[metric] - base[metric] for row, base in zip(samples[policy], samples["fcfs"])]
                    )
                    metrics[metric] = estimate
                summaries.append({"block": block, "family": family, "policy": policy, "metrics": metrics})
            for metric in ("mean_t", "p99_t"):
                predicted = {policy: float(np.mean([row[metric] for row in samples[policy]])) for policy in POLICIES}
                # Policy order resolves exact ties; regret is still zero on a reference tie.
                chosen = min(POLICIES, key=predicted.get)
                reference_values = {policy: observed[policy][metric] for policy in POLICIES}
                best = min(POLICIES, key=reference_values.get)
                flips, pairs = 0, 0
                for left, right in combinations(POLICIES, 2):
                    delta = observed[left][metric] - observed[right][metric]
                    if np.isclose(delta, 0, atol=1e-9, rtol=0):
                        continue
                    pairs += 1
                    flips += int(delta * (predicted[left] - predicted[right]) < 0)
                decisions.append(
                    {
                        "block": block,
                        "family": family,
                        "metric": metric,
                        "chosen": chosen,
                        "reference_best": best,
                        "reference_regret": observed[chosen][metric] - observed[best][metric],
                        "reference_relative_regret": observed[chosen][metric] / observed[best][metric] - 1,
                        "reversed_pairs": flips,
                        "reference_nontied_pairs": pairs,
                    }
                )
    return summaries, decisions


def experiment(source, config=ExperimentConfig()):
    """Fit once, replay prespecified chronological blocks, drain every cohort."""
    training, heldout, split = chronological_split(source, config.train_fraction)
    block_size = config.jobs + config.warmup
    if len(heldout) < block_size * config.blocks:
        raise ValueError("not enough held-out jobs for the prespecified blocks")
    models, fits = fit_history(training)
    rows, blocks = [], []
    for block in range(config.blocks):
        jobs = heldout[block * block_size : (block + 1) * block_size]
        groups = resource_groups([job.need for job in jobs])
        services = np.array([job.runtime for job in jobs])
        span = jobs[-1].submit - jobs[config.warmup].submit
        if span <= 0:
            raise ValueError("measured block must have positive arrival span")
        block_info = {
            "block": block,
            "first_submit": jobs[0].submit,
            "last_submit": jobs[-1].submit,
            "input_sha256": fingerprint([[j.job_id, j.submit, j.runtime, j.need] for j in jobs]),
            "service_models": [],
        }
        bundles = [("observed", None, services)]
        for seed in range(config.first_seed, config.first_seed + config.replications):
            rng = np.random.default_rng(np.random.SeedSequence([seed, block]))
            # Open-interval samples; exact endpoints are never silently clipped.
            uniforms = (rng.integers(0, 2**52, size=len(jobs), dtype=np.int64) + 0.5) / 2**52
            for family in FAMILIES:
                generated = np.empty(len(jobs))
                for group, model in enumerate(models):
                    mask = groups == group
                    generated[mask] = model.quantiles(family, uniforms[mask])
                bundles.append((family, seed, generated))
        for family, seed, generated in bundles:
            needs, trace = make_trace(jobs, generated, models)
            digest = fingerprint([[j.arrival, needs[j.cls], j.service, j.estimate] for j in trace])
            measured = generated[config.warmup :]
            work = sum(job.need * duration for job, duration in zip(jobs[config.warmup :], measured))
            block_info["service_models"].append(
                {
                    "family": family,
                    "seed": seed,
                    "trace_sha256": digest,
                    "mean_s": float(measured.mean()),
                    "p99_s": float(np.quantile(measured, 0.99)),
                    "offered_work_ratio": float(work / (source.capacity * span)),
                }
            )
            for policy in POLICIES:
                rows.append(
                    {
                        "block": block,
                        "family": family,
                        "seed": seed,
                        "policy": policy,
                        "trace_sha256": digest,
                        **replay(source.capacity, needs, trace, policy, config.warmup),
                    }
                )
        blocks.append(block_info)
    summaries, decisions = summarize(rows)
    return {
        "schema_version": 1,
        "config": asdict(config),
        "capacity": source.capacity,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "audit": source.audit,
        "split": split,
        "fits": fits,
        "blocks": blocks,
        "scheduler_runs": len(rows),
        "rows": rows,
        "summaries": summaries,
        "decisions": decisions,
        "interpretation": "Conditional completed-job replay, not reconstruction of the original cluster. "
        "Intervals measure Monte Carlo error conditional on one fixed history and arrival/K block; "
        "not population uncertainty, stability proofs, or checkpoint-overhead evidence.",
    }


def main():
    """Run the pinned research protocol; network access is explicitly opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache", type=Path, default=Path(".cache/real_trace/SDSC-SP2-1998-4.2-cln.swf"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--replications", type=int, default=8)
    args = parser.parse_args()
    config = ExperimentConfig(jobs=args.jobs, warmup=args.warmup, blocks=args.blocks, replications=args.replications)
    payload = verified_source(args.cache, args.download)
    source = parse_swf(payload.decode("ascii").splitlines(), capacity=128)
    result = experiment(source, config)
    result["source"] = {
        "page": SOURCE_PAGE,
        "download": SOURCE_URL,
        "sha256": SOURCE_SHA256,
        "mirror_commit": MIRROR_COMMIT,
        "capacity_assumption": "128 identical nodes, fixed",
        "attribution": "SDSC / Victor Hazlewood; SWF conversion Dror Feitelson; DIR-LAB mirror",
        "usage_conditions": "NPACI JOBLOG: educational, research and non-profit use; "
        "preserve the original copyright notice; data is not MIT-licensed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps({"output": str(args.output), "scheduler_runs": result["scheduler_runs"], "split": result["split"]})
    )


if __name__ == "__main__":
    main()
