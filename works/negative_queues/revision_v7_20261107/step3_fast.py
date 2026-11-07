"""Complete the shared-work study with independently assembled stationary blocks."""

import concurrent.futures
import json
from pathlib import Path
from works.negative_queues.revision_v7_20261107.shared_work_fast import calculate
from works.negative_queues.revision_v7_20261107.study_cases import cases

ROOT = Path(__file__).resolve().parent / "step3_fast"


def run(task):
    k, levels = task
    case = cases()[11]
    name = f"case11-k{k}-n{levels}-b0.json"
    path = ROOT / name
    if path.exists():
        return name, json.loads(path.read_text())
    r = calculate(case["cv"], 3, k, levels, n=case["n"], lam=case["lam"], delta=case["delta"])
    r["case"] = case
    r["stages"] = k
    path.write_text(json.dumps(r, indent=2) + "\n")
    return name, r


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3, max_tasks_per_child=1) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run, task) for task in [(8, 240), (12, 240), (8, 400)]]):
            name, r = f.result()
            print(name, {k: r[k] for k in ["w", "v", "seconds", "peak_rss_kib", "balance_residual"]}, flush=True)
