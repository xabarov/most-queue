"""Reproducible article data, with independent replications and discarded warm-up."""

import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import t

from most_queue.random.distributions import GammaDistribution, H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.negative import DisasterScenario, NegativeServiceType, QsSimNegatives, RcsScenario
from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc

ROOT = Path(__file__).resolve().parent
REPS, WARMUP, SAMPLE = 8, 20000, 200000


def simulate(args):
    cv, mode, scenario, seed = args
    start = time.perf_counter()
    fixed = scenario == "fixed"
    delta = 0.02 if fixed else 0.3
    b = gamma_moments_by_mean_and_cv(2.1, cv)
    rs = {"remove": RcsScenario.REMOVE, "restart": RcsScenario.REQUEUE, "fixed": RcsScenario.REQUEUE_NO_RESAMPLING}[
        scenario
    ]
    ds = {
        "remove": DisasterScenario.CLEAR_SYSTEM,
        "restart": DisasterScenario.REQUEUE_ALL,
        "fixed": DisasterScenario.REQUEUE_ALL_NO_RESAMPLING,
    }[scenario]
    sim = QsSimNegatives(
        3,
        NegativeServiceType.RCS if mode == "rcs" else NegativeServiceType.DISASTER,
        verbose=False,
        rcs_scenario=rs,
        disaster_scenario=ds,
        seed=seed,
    )
    sim.set_positive_sources(1.0, "M")
    sim.set_negative_sources(delta, "M")
    if cv > 1:
        sim.set_servers(H2Distribution.get_params(b), "H")
    else:
        sim.set_servers(GammaDistribution.get_params(b[:2]), "Gamma")
    while sim.total < WARMUP:
        sim.run_one_step()
    n0, k0, served0 = sim.total, sim.taked, sim.served
    w0, v0 = np.array(sim.w) * k0, np.array(sim.v) * n0
    while sim.total < n0 + SAMPLE:
        sim.run_one_step()
    w = (np.array(sim.w) * sim.taked - w0) / (sim.taked - k0)
    v = (np.array(sim.v) * sim.total - v0) / (sim.total - n0)
    return dict(
        cv=cv,
        mode=mode,
        scenario=scenario,
        seed=seed,
        departures=sim.total - n0,
        waiting_observations=sim.taked - k0,
        w=w[:2].tolist(),
        v=v[:2].tolist(),
        sw=float(np.sqrt(w[1] - w[0] ** 2)),
        sv=float(np.sqrt(v[1] - v[0] ** 2)),
        q=(sim.served - served0) / (sim.total - n0),
        seconds=time.perf_counter() - start,
    )


def calculate(cv, mode, scenario, levels):
    cls = MGnNegativeRCSCalc if mode == "rcs" else MGnNegativeDisasterCalc
    c = cls(
        3,
        calc_params=TakahashiTakamiParams(N=levels, tolerance=1e-12),
        requeue_on_disaster=scenario != "remove",
        repeat_without_resampling=scenario == "fixed",
    )
    c.set_sources(1.0, 0.02 if scenario == "fixed" else 0.3)
    c.set_servers(gamma_moments_by_mean_and_cv(2.1, cv))
    start = time.perf_counter()
    r = c.run()
    return dict(
        w=[float(np.real(x)) for x in r.w[:2]],
        v=[float(np.real(x)) for x in r.v[:2]],
        sw=float(np.sqrt(np.real(r.w[1] - r.w[0] ** 2))),
        sv=float(np.sqrt(np.real(r.v[1] - r.v[0] ** 2))),
        q=r.q,
        tail=r.p[-1],
        levels=levels,
        seconds=time.perf_counter() - start,
        restart_rate=getattr(c, "effective_restart_rate", None),
    )


if __name__ == "__main__":
    cases = [
        (cv, mode, scenario)
        for scenario in ["remove", "restart", "fixed"]
        for cv in [0.8, 1.2]
        for mode in ["rcs", "disaster"]
    ]
    tasks = [(*case, 202611070 + 100 * i + j) for i, case in enumerate(cases) for j in range(REPS)]
    rawfile = ROOT / "replications.jsonl"
    done = []
    if rawfile.exists():
        done = [json.loads(line) for line in rawfile.read_text().splitlines()]
    completed = {r["seed"] for r in done}
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(simulate, a) for a in tasks if a[-1] not in completed]
        for f in concurrent.futures.as_completed(futures):
            row = f.result()
            done.append(row)
            with rawfile.open("a") as out:
                out.write(json.dumps(row) + "\n")
            print(f"{len(done)}/{len(tasks)} {row['cv']} {row['mode']} {row['scenario']}", flush=True)
    results = []
    for cv, mode, scenario in cases:
        rows = [r for r in done if (r["cv"], r["mode"], r["scenario"]) == (cv, mode, scenario)]
        sim = {}
        for key in ["w", "v", "sw", "sv", "q"]:
            vals = np.array([r[key][0] if key in ["w", "v"] else r[key] for r in rows])
            sim[key] = dict(
                mean=float(vals.mean()), half_ci95=float(t.ppf(0.975, REPS - 1) * vals.std(ddof=1) / np.sqrt(REPS))
            )
        calc = calculate(cv, mode, scenario, 160)
        larger = calculate(cv, mode, scenario, 240)
        change = max(abs(calc[k][0] - larger[k][0]) for k in ["w", "v"])
        results.append(
            dict(
                cv=cv,
                mode=mode,
                scenario=scenario,
                simulation=sim,
                calculation=calc,
                truncation_change=change,
                replication_seeds=[r["seed"] for r in rows],
            )
        )
    (ROOT / "experiment-results.json").write_text(
        json.dumps(
            dict(repetitions=REPS, warmup_departures=WARMUP, sampled_departures=SAMPLE, results=results),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Saved experiment-results.json", flush=True)
