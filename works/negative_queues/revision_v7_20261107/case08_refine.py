"""Follow-up numerical sensitivity; retain the original prespecified estimate."""

import concurrent.futures, json, resource
from pathlib import Path
from works.negative_queues.revision_v7_20261107.ph_queue import calculate, fixed_work
from works.negative_queues.revision_v7_20261107.study_cases import cases

ROOT = Path(__file__).resolve().parent / "case08_refinement"


def run(task):
    j, k = task
    c = cases()[8]
    path = ROOT / f"j{j}-k{k}.json"
    if path.exists():
        return path.name, json.loads(path.read_text())
    r = calculate(
        fixed_work(c["cv"], j, k),
        n=c["n"],
        lam=c["lam"],
        delta=c["delta"],
        mode="rcs",
        action="fixed",
        levels=120,
        behind=32,
    )
    r.update(classes=j, stages=k, case=c, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    path.write_text(json.dumps(r, indent=2) + "\n")
    return path.name, r


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3, max_tasks_per_child=1) as pool:
        for f in concurrent.futures.as_completed(
            [pool.submit(run, t) for t in [(j, k) for j in [4, 5] for k in [4, 8, 12, 16]] + [(3, 16)]]
        ):
            name, r = f.result()
            print(name, r["w"], r["v"], r["seconds"], flush=True)
