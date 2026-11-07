"""Positive-PH comparisons prescribed independently of simulation estimates."""

import concurrent.futures
import json
import resource
from pathlib import Path

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107.ph_queue import calculate, hyperexponential
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph
from works.negative_queues.revision_v7_20261107.study_cases import cases

ROOT = Path(__file__).resolve().parent / "step4"


def erlang_cases():
    return [
        dict(id=f"erlang-{mode}", mean=2.1, cv=0.5, n=3, lam=0.6, delta=0.3, action="restart", mode=mode)
        for mode in ["rcs", "disaster"]
    ]


def run(task):
    case, j, levels, behind = task
    path = ROOT / f"{case['id']}-j{j}-n{levels}-b{behind}.json"
    if path.exists():
        return path.name, json.loads(path.read_text())
    if case["cv"] < 1:
        service = gamma_ph(case["mean"], case["cv"], j)
    else:
        h = H2Distribution.get_params_clx(gamma_moments_by_mean_and_cv(case["mean"], case["cv"]))
        service = hyperexponential([h.p1, 1 - h.p1], [h.mu1, h.mu2])
    result = calculate(
        service, levels=levels, behind=behind, **{k: case[k] for k in ["n", "lam", "delta", "action", "mode"]}
    )
    result.update(case=case, quadrature=j, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path.name, result


def main():
    ROOT.mkdir(exist_ok=True)
    tasks = []
    for action in ["remove", "restart"]:
        for mode in ["rcs", "disaster"]:
            for cv in [0.8, 1.2]:
                case = dict(
                    id=f"base-{action}-{mode}-cv{cv}", mean=2.1, cv=cv, n=3, lam=1, delta=0.3, action=action, mode=mode
                )
                grids = [(2, 160, 40), (4, 160, 40), (8, 160, 40), (8, 240, 80)] if cv < 1 else [(0, 240, 80)]
                tasks.extend((case, j, n, b) for j, n, b in grids)
    tasks.extend((case, 8, 240, 80) for case in cases() if case["cv"] == 0.8 and case["action"] != "fixed")
    tasks.extend((case, 4, n, b) for case in erlang_cases() for n, b in [(160, 40), (240, 80)])
    with concurrent.futures.ProcessPoolExecutor(max_workers=3, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for f in concurrent.futures.as_completed(futures):
            name, r = f.result()
            print(name, {k: r[k] for k in ["w", "v", "seconds", "peak_rss_kib"]}, flush=True)


if __name__ == "__main__":
    main()
