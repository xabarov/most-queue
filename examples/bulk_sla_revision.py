"""Frozen single-server article revision experiments (2026-10-12).

Run with --panel patience/load/decision/inoue. JSON contains every refinement,
refusal and simulation replicate; old EPIC-066 outputs are never overwritten.
The independent cohort DES does not use any analytical transition builder.
"""

import argparse
import hashlib
import heapq
import json
import platform
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import gamma as gamma_distribution
from scipy.stats import t as student_t

from examples.batch_service_sla_exact_tail_comparison import inoue_phi
from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc
from most_queue.theory.batch.bulk_service_h2 import BulkServiceH2Calc
from most_queue.theory.batch.dynamic_batching import DynamicBatchingCalc

DEADLINES = (5.0, 10.0, 20.0)
ALPHA, TAU0 = 0.1438, 1.8874
H2_P = (1 + np.sqrt((1.5**2 - 1) / (1.5**2 + 1))) / 2


def mean_service(size):
    """Affine mean, in declared abstract time units (not calibrated LLM seconds)."""
    return ALPHA * size + TAU0


def make_calc(case, n):
    """Configure a library solver for a frozen scenario."""
    a, b, family, patience = case["a"], case["b"], case["family"], case["gamma"]
    if family == "H2":
        calc = BulkServiceH2Calc(a, b, n, gamma=patience)
        calc.set_servers(H2_P, lambda s: 2 * H2_P / mean_service(s), lambda s: 2 * (1 - H2_P) / mean_service(s))
    elif family == "M":
        calc = BulkServiceMM1Calc(a, b, n, gamma=patience)
        calc.set_servers(lambda s: 1 / mean_service(s))
    else:
        shape = int(family[1:])
        calc = BulkServiceErlangCalc(a, b, shape, n, gamma=patience)
        calc.set_servers(lambda s: shape / mean_service(s))
    calc.set_sources(case["lambda"])
    return calc


def fitted_tail(moments, deadline, atom=0.0):
    """Two-moment gamma fit; optionally preserve an independently known atom."""
    weight = 1 - atom
    mean, second = np.array(moments[:2]) / weight
    variance = second - mean * mean
    if variance <= 0 or not 0 < weight <= 1:
        raise ValueError("moment fit outside its positive-variance domain")
    return float(weight * gamma_distribution.sf(deadline, mean * mean / variance, scale=variance / mean))


def prediction(case):
    """Refine stationary N without mistaking convergence for certification."""
    attempts, previous = [], None
    result = {"case": case, "status": "unresolved", "attempts": attempts}
    for n in (200, 400, 800, 1600):
        started = time.perf_counter()
        try:
            calc = make_calc(case, n)
            chain = calc.get_tagged_wait()
            moments = chain.moments(2)
            tails = [chain.tail(d) for d in DEADLINES]
            qmass = calc._solve_pi().reshape(-1, n + 1).sum(axis=0)
            queue_mean = float(qmass @ np.arange(n + 1))
            flow_error = chain.abandoned - case["gamma"] * queue_mean / case["lambda"]
            row_error = float(np.max(np.abs(chain.sub @ np.ones(len(chain.h)) + chain.serve_rate + case["gamma"])))
            atom = chain.atom / chain.served
            values = np.array(moments + tails + [chain.abandoned])
            delta = None if previous is None else float(np.max(np.abs(values - previous)))
            entry = {
                "N": n,
                "queue_boundary_mass": chain.boundary_mass,
                "behind_cap": chain.cap,
                "behind_bound": chain.bound,
                "initial_omitted_mass": chain.omitted_mass,
                "states": len(chain.h),
                "row_residual": row_error,
                "abandon_flow_residual": flow_error,
                "moments": moments,
                "tails": tails,
                "served": chain.served,
                "abandoned": chain.abandoned,
                "atom_conditional": atom,
                "local_moment_bounds": chain.moment_error_bounds(2),
                "outcomes": [chain.outcomes(d) for d in DEADLINES],
                "ordinary_gamma_tails": [fitted_tail(moments, d) for d in DEADLINES],
                "atom_gamma_tails": [fitted_tail(moments, d, atom) for d in DEADLINES],
                "refinement_max_difference": delta,
                "mean_queue": queue_mean,
                "busy": float(1 - qmass[0] * 0 - calc._solve_pi()[: n + 1].sum()),
                "elapsed_s": time.perf_counter() - started,
            }
            attempts.append(entry)
            if delta is not None and delta <= 1e-7 and chain.boundary_mass <= 1e-10:
                result["status"] = "converged_diagnostic"
                result["selected"] = entry
                break
            previous = values
        except (ValueError, MemoryError, RuntimeError) as exc:
            attempts.append({"N": n, "refusal": str(exc), "elapsed_s": time.perf_counter() - started})
            result["status"] = "refused"
            break
    return result


def simulate(case, seed, customers=100_000, warmup=20_000):
    """Independent FIFO cohort DES, continuing arrivals through final resolution.

    Count arrivals, not completions, into a fixed cohort. A job's wait and full
    service duration are known at dispatch. Expired jobs leave individually.
    """
    rng = np.random.default_rng(seed)
    queue, patience_heap = OrderedDict(), []
    now, generated, resolved = 0.0, 0, 0
    arrival, completion = rng.exponential(1 / case["lambda"]), np.inf
    n_served, n_abandoned, zero = 0, 0, 0
    wait_sum, sojourn_sum = 0.0, 0.0
    late = np.zeros(len(DEADLINES), dtype=int)

    def tracked(identifier):
        return warmup <= identifier < warmup + customers

    def dispatch():
        nonlocal completion, resolved, n_served, wait_sum, sojourn_sum, zero
        if np.isfinite(completion) or len(queue) < case["a"]:
            return
        size = len(queue) if case["b"] is None else min(case["b"], len(queue))
        mean = mean_service(size)
        family = case["family"]
        if family == "D":
            service = mean
        elif family == "H2":
            weight = H2_P if rng.random() < H2_P else 1 - H2_P
            service = rng.exponential(mean / (2 * weight))
        elif family == "M":
            service = rng.exponential(mean)
        else:
            shape = int(family[1:])
            service = rng.gamma(shape, mean / shape)
        completion = now + service
        for _ in range(size):
            identifier, arrived = queue.popitem(last=False)
            if tracked(identifier):
                wait = now - arrived
                n_served += 1
                resolved += 1
                wait_sum += wait
                sojourn_sum += wait + service
                zero += int(wait == 0)
                late[:] += [int(wait > d) for d in DEADLINES]

    while resolved < customers:
        while patience_heap and patience_heap[0][1] not in queue:
            heapq.heappop(patience_heap)
        abandoned_at = patience_heap[0][0] if patience_heap else np.inf
        now = min(arrival, completion, abandoned_at)
        if now == arrival:
            queue[generated] = now
            if case["gamma"] > 0:
                heapq.heappush(patience_heap, (now + rng.exponential(1 / case["gamma"]), generated))
            generated += 1
            arrival = now + rng.exponential(1 / case["lambda"])
        elif now == completion:
            completion = np.inf
        else:
            _, identifier = heapq.heappop(patience_heap)
            del queue[identifier]
            if tracked(identifier):
                n_abandoned += 1
                resolved += 1
        dispatch()
    return {
        "seed": seed,
        "customers": customers,
        "warmup": warmup,
        "mean_wait": wait_sum / n_served,
        "mean_sojourn": sojourn_sum / n_served,
        "abandoned": n_abandoned / customers,
        "atom_conditional": zero / n_served,
        "conditional_tails": (late / n_served).tolist(),
        "on_time": ((n_served - late) / customers).tolist(),
        "late": (late / customers).tolist(),
        "extra_arrivals": generated - warmup - customers,
    }


def simulate_replicates(case, seed_start=12000):
    """Replicate-t intervals account for within-run dependence, not model error."""
    runs = [simulate(case, seed) for seed in range(seed_start, seed_start + 12)]
    summary = {}
    for key in ("mean_wait", "mean_sojourn", "abandoned", "atom_conditional", "conditional_tails", "on_time", "late"):
        samples = np.array([run[key] for run in runs])
        mean = samples.mean(axis=0)
        half = student_t.ppf(0.975, 11) * samples.std(axis=0, ddof=1) / np.sqrt(12)
        summary[key] = {"mean": mean.tolist(), "lower": (mean - half).tolist(), "upper": (mean + half).tolist()}
    return {"replicates": runs, "t95": summary}


def cases_for(panel):
    """Fixed grids; no tuning based on observed output."""
    cases = []
    if panel == "patience":
        for family in ("E3", "H2"):
            for patience in (0.0, 0.02, 0.05, 0.1, 0.25, 0.5):
                cases.append({"a": 2, "b": 8, "family": family, "gamma": patience, "lambda": 0.8})
    elif panel == "load":
        for family in ("E10", "E3", "M", "H2"):
            for load in (0.3, 0.6, 0.85):
                for threshold in (1, 2, 4):
                    cases.append(
                        {"a": threshold, "b": 8, "family": family, "gamma": 0.0, "lambda": load * 8 / mean_service(8)}
                    )
    elif panel == "decision":
        for family in ("E3", "H2"):
            for patience in (0.0, 0.1):
                for cap in range(2, 17):
                    for threshold in range(1, min(4, cap) + 1):
                        cases.append({"a": threshold, "b": cap, "family": family, "gamma": patience, "lambda": 0.8})
    elif panel == "inoue":
        for lam in (0.2, 0.8):
            for cap in (4, 8, 16, 32, None):
                cases.append({"a": 1, "b": cap, "family": "D", "gamma": 0.0, "lambda": lam})
    return cases


def main():
    """Execute one frozen panel; refuse to overwrite existing observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("patience", "load", "decision", "inoue"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/research"))
    args = parser.parse_args()
    target = args.output_dir / f"bulk-sla-revision-{args.panel}-2026.json"
    if target.exists():
        raise FileExistsError(f"preserve existing experiment: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__)] + list(Path("most_queue/theory/batch").glob("*.py"))
    protocol = Path("docs/research/bulk-sla-revision-protocol-2026.md")
    result = {
        "panel": args.panel,
        "date": "2026-10-12",
        "deadlines": DEADLINES,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "cases": [],
    }
    for index, case in enumerate(cases_for(args.panel)):
        start = time.perf_counter()
        if args.panel == "inoue":
            entry = {"case": case, "simulation": simulate_replicates(case, 13000)}
            if case["b"] is None:
                calc = DynamicBatchingCalc().set_sources(case["lambda"]).set_deterministic_service(ALPHA, TAU0)
                value = calc.run()
                entry["unbounded_mean_wait"] = value.w[0]
                entry["unbounded_mean_sojourn"] = value.v[0]
                entry["inoue_sojourn_upper_bound"] = inoue_phi(case["lambda"], ALPHA, TAU0)
        else:
            entry = prediction(case)
            if args.panel == "patience":
                entry["simulation"] = simulate_replicates(case)
        entry["elapsed_s"] = time.perf_counter() - start
        result["cases"].append(entry)
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
        print(args.panel, index + 1, case, entry.get("status", "simulated"), round(entry["elapsed_s"], 2), flush=True)


if __name__ == "__main__":
    main()
