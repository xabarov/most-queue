"""L6: finite SMDP reproduction, independent LP and unbounded DES comparisons."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.stats import poisson

from most_queue.sim.smdp_batching import BatchSmdpSim
from most_queue.theory.batch import BatchSmdpCalc


def serialize(result):
    """Exclude wall time, so seeded records are reproducible byte for byte."""
    output = asdict(result)
    output.pop("duration", None)
    output.pop("bias", None)
    output.pop("iteration_trace", None)
    return output


def calculate(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    n, rate, times, energies, weight=1, penalty=100, tolerance=1e-7
):
    """Build a fresh finite model."""
    model = BatchSmdpCalc(n, energy_weight=weight, overflow_cost=penalty)
    model.set_sources(rate)
    model.set_servers(times, energies)
    return model.run(tolerance=tolerance, max_iterations=20000)


def occupation_lp(
    n, rate, times, energies, weight, penalty
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Independent source-formula LP: x is decision frequency per physical time.

    No production epoch/operator/stationary helper is used. This oracle checks
    the finite model, not the infinite queue or its truncation error.
    """
    columns, rewards, periods = [], [], []
    for state in range(n + 2):
        for action in range(min(state, len(times)) + 1):
            row = np.zeros(n + 2)
            effective = min(state, n)
            if action == 0:
                row[min(state + 1, n + 1)] = 1
                period, reward = 1 / rate, effective / rate**2
            else:
                period = times[action - 1]
                base = effective - action
                row[base : n + 1] = poisson.pmf(np.arange(n - base + 1), rate * period)
                row[-1] = poisson.sf(n - base, rate * period)
                reward = effective * period / rate + period**2 / 2 + weight * energies[action - 1]
            if state == n + 1:
                reward += penalty * period
            row[state] -= 1
            columns.append(row)
            periods.append(period)
            rewards.append(reward)
    flow = np.array(columns).T
    equalities = np.vstack([flow[:-1], periods])
    target = np.r_[np.zeros(n + 1), 1.0]
    optimum = linprog(rewards, A_eq=equalities, b_eq=target, bounds=(0, None), method="highs")
    if not optimum.success:
        raise RuntimeError(optimum.message)
    return {
        "gain": float(optimum.fun),
        "flow_residual": float(max(abs(flow @ optimum.x))),
        "time_normalization_residual": float(abs(np.dot(periods, optimum.x) - 1)),
    }


def simulate(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    rate, times, energies, policy, weight, horizon, replications, initial_jobs=0
):
    """Paired exogenous streams: policies consume no arrival RNG draws."""
    runs = []
    for seed in range(9100, 9100 + replications):
        model = BatchSmdpSim(seed=seed, energy_weight=weight)
        model.set_sources(rate)
        model.set_servers(times, energies)
        result = model.run(horizon, policy, warmup=horizon / 5, initial_jobs=initial_jobs)
        assert result.initial_jobs + result.arrived == result.completed + result.unfinished
        assert abs(
            result.total_population_area - result.completed_residence - result.unfinished_residence
        ) < 1e-6 * max(1, result.total_population_area)
        runs.append({"seed": seed, "result": serialize(result)})
    costs = [r["result"]["average_cost"] for r in runs]
    return {
        "runs": runs,
        "cost_mean": float(np.mean(costs)),
        "cost_se": float(np.std(costs, ddof=1) / np.sqrt(replications)),
    }


def experiment(horizon=20000, replications=4):
    """Published numerical profile is not a new GPU measurement."""
    times = [0.3051 * b + 1.052 for b in range(1, 33)]
    energies = [19.90 * b + 19.60 for b in range(1, 33)]
    capacity = 32 / times[-1]
    table = []
    for rate_name, rate in [("exact_profile_capacity", 0.9 * capacity), ("rounded_paper_capacity", 0.9 * 2.96)]:
        for penalty, n, printed_gain, printed_delta in [
            (10000, 89, 66.1384, 9.36e-4),
            (1000, 78, 66.1383, 9.78e-4),
            (100, 70, 66.1377, 8.36e-4),
            (10, 161, 66.1374, 6.14e-12),
            (0, 192, 66.1374, 1.30e-14),
        ]:
            result = calculate(n, rate, times, energies, penalty=penalty, tolerance=0.01)
            table.append(
                {
                    "rate_definition": rate_name,
                    "rate": rate,
                    "penalty": penalty,
                    "state_max": n,
                    "printed_gain": printed_gain,
                    "printed_delta": printed_delta,
                    "gain_difference": result.average_cost - printed_gain,
                    "result": serialize(result),
                }
            )
    lp_result = calculate(70, 0.9 * capacity, times, energies)
    lp = occupation_lp(70, 0.9 * capacity, times, energies, 1, 100)
    assert abs(lp_result.average_cost - lp["gain"]) < 1e-5
    sensitivity = []
    for n in [32, 50, 70, 100]:
        for penalty in [0, 10, 100, 1000]:
            result = calculate(n, 0.9 * capacity, times, energies, penalty=penalty, tolerance=1e-5)
            sensitivity.append({"state_max": n, "penalty": penalty, "result": serialize(result)})
    source_des = []
    for name, policy in [
        ("finite_overflow_action", lp_result.policy),
        ("explicit_capacity_tail_repair", lp_result.policy[:-1] + [32]),
        ("work_conserving", lambda s: min(s, 32)),
        ("fixed_batch_32", lambda s: 32 if s >= 32 else 0),
    ]:
        source_des.append(
            {
                "name": name,
                "tail_action": int(lp_result.policy[-1]) if name == "finite_overflow_action" else 32,
                "result": simulate(0.9 * capacity, times, energies, policy, 1, horizon, replications),
            }
        )
    source_tail_stress = []
    for name, policy in [
        ("finite_overflow_action", lp_result.policy),
        ("explicit_capacity_tail_repair", lp_result.policy[:-1] + [32]),
    ]:
        source_tail_stress.append(
            {
                "name": name,
                "initial_jobs": 1000,
                "result": simulate(
                    0.9 * capacity, times, energies, policy, 1, horizon, replications, initial_jobs=1000
                ),
            }
        )
    synthetic = []
    for rate in [0.2, 1, 2.5]:
        for weight in [0, 4, 20]:
            result = calculate(40, rate, [0.4, 0.6], [1, 1.3], weight)
            comparisons = []
            for name, policy in [
                ("finite_policy", result.policy),
                ("capacity_tail_repair", result.policy[:-1] + [2]),
                ("work_conserving", lambda s: min(s, 2)),
                ("fixed_batch_2", lambda s: 2 if s >= 2 else 0),
            ]:
                comparisons.append(
                    {
                        "name": name,
                        "result": simulate(rate, [0.4, 0.6], [1, 1.3], policy, weight, horizon, replications),
                    }
                )
            synthetic.append(
                {"rate": rate, "energy_weight": weight, "finite": serialize(result), "comparisons": comparisons}
            )
    counter_model = BatchSmdpCalc(2, delay_weight=1e-8, energy_weight=100, overflow_cost=0)
    counter_model.set_sources(0.1)
    counter_model.set_servers([1, 1], [1, 1])
    counter = counter_model.run()
    counter_runs = []
    for time in [1000, 10000, 100000]:
        sim = BatchSmdpSim(seed=9200, delay_weight=1e-8, energy_weight=100)
        sim.set_sources(0.1)
        sim.set_servers([1, 1], [1, 1])
        counter_runs.append({"horizon": time, "result": serialize(sim.run(time, counter.policy))})
    return {
        "schema_version": 1,
        "source": "arXiv:2301.12865v3",
        "scope": "finite overflow optimum != infinite optimum; tail repairs are separate policies",
        "units": "source: milliseconds, millijoules, requests/ms, mJ/ms = W; synthetic: abstract time/energy",
        "source_times": times,
        "source_energies": energies,
        "exact_capacity": capacity,
        "horizon": horizon,
        "warmup": horizon / 5,
        "replications": replications,
        "solver": {
            "eta": "0.9 min(mean sojourns)",
            "max_iterations": 20000,
            "table_tolerance": 0.01,
            "sensitivity_tolerance": 1e-5,
            "other_tolerance": 1e-7,
        },
        "source_table": table,
        "source_lp": {"oracle": lp, "rvi": serialize(lp_result)},
        "sensitivity": sensitivity,
        "source_des": source_des,
        "source_tail_stress": source_tail_stress,
        "synthetic": synthetic,
        "always_wait_counter": {"finite": serialize(counter), "unbounded": counter_runs},
    }


def main():
    """Save records; validate CLI controls before computing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/research/smdp-batching-results-2026.json")
    parser.add_argument("--horizon", type=float, default=20000)
    parser.add_argument("--replications", type=int, default=4)
    args = parser.parse_args()
    if not np.isfinite(args.horizon) or args.horizon <= 0 or args.replications < 2:
        parser.error("positive finite horizon and at least two replications required")
    Path(args.output).write_text(
        json.dumps(experiment(args.horizon, args.replications), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Saved 10 table rows, 16 truncations, source LP, 42 DES comparisons and the waiting counterexample")


if __name__ == "__main__":
    main()
