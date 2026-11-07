"""v9: grids for the quadrature correction of the retained-work model.

The correction compares the retained-work approximation (J work classes,
K Erlang stages) with an accurate M/G/n reference at delta=0 on the same
grid and truncation:
    m_corr(delta) = m_{J,K}(delta) + [m_ref(0) - m_{J,K}(0)].
Each task writes its own JSON; existing files are skipped.

Usage (from repo root):
  PYTHONPATH=. OPENBLAS_NUM_THREADS=1 .venv/bin/python \
      works/negative_queues/revision_v9_20261108/correction_grids.py <group> [<workers>]
"""

import json
import os
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107.ph_queue import calculate, fixed_work, hyperexponential
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph

OUT = Path(__file__).resolve().parent / "correction"
MEAN = 2.1

# (label, n, lam, cv, J, Ks, levels)
SETTINGS = {
    "main-cv0.8": (3, 1.0, 0.8, 4, (8, 12, 16), 100),
    "main-cv1.2": (3, 1.0, 1.2, 4, (8, 12, 16), 100),
    "case08-10": (2, 0.5 * 2 / MEAN, 0.8, 3, (4, 8, 12), 120),
    "case09": (3, 0.85 * 3 / MEAN, 1.2, 3, (4, 8, 12), 240),
    "case11": (3, 0.8 * 3 / MEAN, 1.2, 3, (4, 8, 12), 240),
}


def reference(n, lam, cv, levels):
    if cv < 1:
        service, name = gamma_ph(MEAN, cv, 16), "positive Gamma PH J=16"
    else:
        h = H2Distribution.get_params(gamma_moments_by_mean_and_cv(MEAN, cv))
        service, name = hyperexponential([h.p1, 1 - h.p1], [h.mu1, h.mu2]), "H2"
    return service, name


def task(spec):
    label, kind, n, lam, cv, j, k, levels = spec
    path = OUT / f"{label}-{kind}-j{j}-k{k}-n{levels}.json"
    if path.exists():
        return str(path), "skipped"
    started = time.perf_counter()
    if kind == "ref":
        service, name = reference(n, lam, cv, levels)
        result = calculate(service, n=n, lam=lam, delta=0.0, mode="rcs", action="restart", levels=levels, behind=0)
    else:
        service = fixed_work(cv, j, k, MEAN)
        name = service.name
        result = calculate(service, n=n, lam=lam, delta=0.0, mode="rcs", action="fixed", levels=levels, behind=0)
    record = {
        "label": label,
        "kind": kind,
        "model": name,
        "n": n,
        "lam": lam,
        "cv": cv,
        "delta": 0.0,
        "classes": j,
        "stages": k,
        "levels": levels,
        "w": [float(x) for x in result["w"]],
        "v": [float(x) for x in result["v"]],
        "tail": float(result.get("tail", float("nan"))),
        "balance_residual": float(result.get("balance_residual", float("nan"))),
        "seconds": time.perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
    }
    path.write_text(json.dumps(record, indent=1))
    return str(path), f"{record['seconds']:.1f}s w1={record['w'][0]:.8f}"


def tasks(group):
    if group == "refs":
        return [(lab, "ref", n, lam, cv, 0, 0, max(levels, 240)) for lab, (n, lam, cv, _, _, levels) in SETTINGS.items()]
    if group == "small":
        lab = "case08-10"
        n, lam, cv, j, ks, levels = SETTINGS[lab]
        return [(lab, "fixed", n, lam, cv, j, k, levels) for k in ks]
    if group == "jcheck":  # consistency of J-differences at n=3, K=8
        n, lam, cv, _, _, levels = SETTINGS["main-cv0.8"]
        return [("main-cv0.8", "fixed", n, lam, cv, j, 8, levels) for j in (3, 5)]
    n, lam, cv, j, ks, levels = SETTINGS[group]
    return [(group, "fixed", n, lam, cv, j, k, levels) for k in ks]


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    group = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    todo = tasks(group)
    with ProcessPoolExecutor(workers) as ex:
        for path, status in ex.map(task, todo):
            print(os.path.basename(path), status, flush=True)
