"""Additional refinement grids, fixed independently of simulation comparisons."""

import concurrent.futures
import fcntl
import json
import resource
from pathlib import Path

from works.negative_queues.refinement_20261107.fixed_work_phase import calculate as stationary_work
from works.negative_queues.revision_v7_20261107.ph_queue import calculate, fixed_work

ROOT = Path(__file__).resolve().parent / "step2"


def run(case):
    mode, cv, j, k, levels, behind = case
    name = f"{mode}-cv{cv}-j{j}-k{k}-n{levels}-b{behind}.json"
    with (ROOT / (name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return calculate_case(case)


def calculate_case(case):
    mode, cv, j, k, levels, behind = case
    name = f"{mode}-cv{cv}-j{j}-k{k}-n{levels}-b{behind}.json"
    path = ROOT / name
    if path.exists():
        return name, json.loads(path.read_text())
    if mode == "rcs":
        result = calculate(fixed_work(cv, j, k), mode=mode, action="fixed", delta=0.02, levels=levels, behind=behind)
    else:
        result = stationary_work(
            cv=cv, mode=mode, classes=j, stages=k, levels=levels, tolerance=1e-9, method="krylov", max_iter=100000
        )
    result.update(cv=cv, classes=j, stages=k, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return name, result


def main():
    ROOT.mkdir(exist_ok=True)
    cases = [("rcs", cv, j, k, 100, 24) for cv in [0.8, 1.2] for j in [3, 4] for k in [12, 16]]
    cases += [("rcs", 1.2, 4, 8, 160, 24), ("rcs", 1.2, 4, 8, 100, 48)]
    cases += [("disaster", cv, j, 16, 100, 0) for cv in [0.8, 1.2] for j in [3, 4]]
    cases += [("disaster", 1.2, 4, 16, 160, 0)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run, c) for c in cases]
        for f in concurrent.futures.as_completed(futures):
            name, result = f.result()
            print(name, {k: result[k] for k in ["w", "v", "seconds", "peak_rss_kib"]}, flush=True)


if __name__ == "__main__":
    main()
