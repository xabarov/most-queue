"""Independent block assembly check of shared retained-work stationary moments."""

import json
import resource
import time
from pathlib import Path

import numpy as np

from works.negative_queues.revision_v7_20261107.ph_queue import fixed_work, occurrence_blocks, stationary


def calculate(cv, classes, stages, levels=100, n=3, lam=1, delta=0.02, tolerance=1e-12, progress=False):
    begin = time.perf_counter()
    service = fixed_work(cv, classes, stages)
    parts = occurrence_blocks(service, n)
    if progress:
        print("Phase blocks built:", time.perf_counter() - begin, flush=True)
    pi, _, diagnostics = stationary(
        parts, n, lam, delta, "disaster", "fixed", levels, tolerance=tolerance, progress=progress
    )
    p = np.array([row.sum() for row in pi])
    queue = np.maximum(np.arange(levels) - n, 0)
    w = [p @ queue / lam, p @ (queue * (queue - 1)) / lam**2]
    sub = service.generator + delta * (service.reset - np.eye(service.size))
    t1 = np.linalg.solve(-sub, np.ones(service.size))
    t2 = np.linalg.solve(-sub, 2 * t1)
    own = service.alpha @ t1, service.alpha @ t2
    v = [p @ np.arange(levels) / lam, w[1] + 2 * w[0] * own[0] + own[1]]
    return dict(
        cv=cv,
        classes=classes,
        stages=stages,
        levels=levels,
        phase_states=len(parts["states"][n]),
        total_states=sum(len(parts["states"][min(i, n)]) for i in range(levels)),
        n=n,
        lam=lam,
        delta=delta,
        mode="disaster",
        w=w,
        v=v,
        sw=float(np.sqrt(w[1] - w[0] ** 2)),
        sv=float(np.sqrt(v[1] - v[0] ** 2)),
        p=p.tolist(),
        tail=p[-1],
        service_moments=list(own),
        service_mean_residual=float(abs(v[0] - w[0] - own[0])),
        seconds=time.perf_counter() - begin,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        solver="independent block assembly, normalized embedded-chain balance without power polishing",
        **diagnostics,
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = calculate(1.2, 4, 8)
    old = json.loads((root.parent / "refinement_20261107/full_work/cv1.2-j4-k8-n100.json").read_text())
    difference = {k: (np.array(result[k]) - old[k]).tolist() for k in ["w", "v"]}
    print(result["seconds"], difference, result["balance_residual"], flush=True)
    (root / "stationary-independent-check.json").write_text(
        json.dumps(dict(result=result, difference=difference), indent=2) + "\n"
    )
