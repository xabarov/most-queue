"""Complete the two outstanding RCS models before the wider validation study."""

import concurrent.futures
import json
import resource
from pathlib import Path

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107.ph_queue import calculate, fixed_work, hyperexponential

ROOT = Path(__file__).resolve().parent / "step1"


def run(case):
    cv, j, k, levels, behind, action = case
    name = f"{action}-cv{cv}-j{j}-k{k}-n{levels}-b{behind}.json"
    path = ROOT / name
    if path.exists():
        return name, json.loads(path.read_text())
    if action == "fixed":
        service = fixed_work(cv, j, k)
    else:
        h = H2Distribution.get_params_clx(gamma_moments_by_mean_and_cv(2.1, cv))
        service = hyperexponential([h.p1, 1 - h.p1], [h.mu1, h.mu2])
    result = calculate(
        service, mode="rcs", action=action, delta=0.02 if action == "fixed" else 0.3, levels=levels, behind=behind
    )
    result.update(cv=cv, classes=j, stages=k, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return name, result


def main():
    ROOT.mkdir(exist_ok=True)
    cases = [(cv, 3, k, 100, 24, "fixed") for cv in [0.8, 1.2] for k in [2, 4, 8]]
    cases += [(cv, 4, k, 100, 24, "fixed") for cv in [0.8, 1.2] for k in [2, 4, 8]]
    cases += [(cv, 3, 4, 160, 48, "fixed") for cv in [0.8, 1.2]]
    cases += [(cv, 0, 0, levels, behind, "remove") for cv in [0.8, 1.2] for levels, behind in [(160, 80), (240, 120)]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run, c) for c in cases]
        for f in concurrent.futures.as_completed(futures):
            name, result = f.result()
            print(name, {k: result[k] for k in ["w", "v", "little_residuals", "seconds", "peak_rss_kib"]}, flush=True)


if __name__ == "__main__":
    main()
