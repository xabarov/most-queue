"""Numerical parameter study using the already validated joint PH equations."""

import concurrent.futures
import json
import resource
from pathlib import Path

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.refinement_20261107.fixed_work_phase import calculate as shared_fixed
from works.negative_queues.revision_v7_20261107.ph_queue import calculate, fixed_work, hyperexponential
from works.negative_queues.revision_v7_20261107.study_cases import cases

ROOT = Path(__file__).resolve().parent / "step3"


def run(task):
    case, k, levels, behind = task
    path = ROOT / f"{case['id']}-k{k}-n{levels}-b{behind}.json"
    if path.exists():
        return path.name, json.loads(path.read_text())
    options = {key: case[key] for key in ["n", "lam", "delta", "mode"]}
    if case["action"] == "fixed" and case["mode"] == "disaster":
        result = shared_fixed(
            cv=case["cv"],
            classes=3,
            stages=k,
            levels=levels,
            tolerance=1e-9,
            method="krylov",
            max_iter=100000,
            **options,
        )
    else:
        if case["action"] == "fixed":
            service = fixed_work(case["cv"], 3, k, case["mean"])
        else:
            h = H2Distribution.get_params_clx(gamma_moments_by_mean_and_cv(case["mean"], case["cv"]))
            service = hyperexponential([h.p1, 1 - h.p1], [h.mu1, h.mu2])
        result = calculate(service, action=case["action"], levels=levels, behind=behind, **options)
    result.update(case=case, stages=k, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path.name, result


def main():
    ROOT.mkdir(exist_ok=True)
    tasks = []
    for case in cases():
        if case["action"] != "fixed":
            tasks.extend((case, 0, n, b) for n, b in [(160, 40), (240, 80)])
        else:
            levels, behind = (120, 32) if case["n"] == 2 else (240, 40)
            if case["mode"] == "disaster":
                behind = 0
            tasks.extend((case, k, levels, behind) for k in [4, 8, 12])
            if case["n"] == 3:
                tasks.append((case, 8, 400, behind))
    with concurrent.futures.ProcessPoolExecutor(max_workers=3, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run, task) for task in tasks]
        for f in concurrent.futures.as_completed(futures):
            name, result = f.result()
            print(name, {k: result[k] for k in ["w", "v", "seconds", "peak_rss_kib"]}, flush=True)


if __name__ == "__main__":
    main()
