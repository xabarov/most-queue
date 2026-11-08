"""Cost-accuracy data for ph_gamma_notes.md: wall-clock time of one calculation
(stationary probabilities + w1, v1) for n=3, lambda=1, b1=2.1, delta=0.3 on the
CV grid: Takahashi-Takami with H2 (n+1 microstates per level), the joint phase
model with the PH-representation of the Gamma law with J=8 nodes (kappa+8
phases) and with J=16 nodes (kappa+16 phases). Output: notes-cost.json.
Run from the repository root with PYTHONPATH=., one process."""

import json
import time
from math import comb
from pathlib import Path

from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107 import ph_queue
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph
from works.negative_queues.revision_v9_20261108 import tt_tagged

ROOT = Path(__file__).resolve().parent
B1, N = 2.1, 3


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def main():
    rows = []
    for cv in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        b = gamma_moments_by_mean_and_cv(B1, cv)
        for mode in ("rcs", "disaster"):
            rec = {"cv": cv, "mode": mode}
            r, sec = timed(lambda: tt_tagged.calculate(b, N, 1.0, 0.3, mode, levels=200, behind=60))
            rec["tt"] = {"seconds": sec, "phases": 2, "level_states": N + 1, "w1": r["w"][0]}
            for nodes in (8, 16):
                s = gamma_ph(B1, cv, nodes)
                r, sec = timed(lambda: ph_queue.calculate(s, n=N, lam=1.0, delta=0.3, mode=mode, action="remove",
                                                          levels=200, behind=60))
                rec[f"ph{nodes}"] = {"seconds": sec, "phases": s.size, "level_states": comb(N + s.size - 1, s.size - 1),
                                     "w1": r["w"][0]}
            print(json.dumps(rec), flush=True)
            rows.append(rec)
    (ROOT / "notes-cost.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
