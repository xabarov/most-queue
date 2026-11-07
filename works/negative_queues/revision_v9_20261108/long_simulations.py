"""v9: long independent simulation series (32 x 2e6 departures, warm-up 2e5).

Uses the independent v7 event simulator unchanged. Seeds: seed0 + replication.
Each case is written to long_sim/<id>.json; existing files are skipped.
Usage: PYTHONPATH=. .venv/bin/python works/negative_queues/revision_v9_20261108/long_simulations.py <workers>
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import t as student
from works.negative_queues.revision_v7_20261107.simulation import run

OUT = Path(__file__).resolve().parent / "long_sim"
REPS, SAMPLE = 32, 2_000_000
M = 2.1
CASES = [  # id, case, seed0
    ("main-fixed-rcs-cv0.8", dict(n=3, lam=1.0, delta=0.02, cv=0.8, action="fixed", mode="rcs"), 2026110810000),
    ("main-fixed-disaster-cv0.8", dict(n=3, lam=1.0, delta=0.02, cv=0.8, action="fixed", mode="disaster"), 2026110811000),
    ("main-fixed-rcs-cv1.2", dict(n=3, lam=1.0, delta=0.02, cv=1.2, action="fixed", mode="rcs"), 2026110812000),
    ("main-fixed-disaster-cv1.2", dict(n=3, lam=1.0, delta=0.02, cv=1.2, action="fixed", mode="disaster"), 2026110813000),
    ("case10", dict(n=2, lam=0.5 * 2 / M, delta=0.04, cv=0.8, action="fixed", mode="disaster"), 2026110814000),
    ("case09", dict(n=3, lam=0.85 * 3 / M, delta=0.04, cv=1.2, action="fixed", mode="rcs"), 2026110815000),
    ("case11", dict(n=3, lam=0.8 * 3 / M, delta=0.04, cv=1.2, action="fixed", mode="disaster"), 2026110816000),
]

def summarize(runs):
    out = {}
    for k in ("w", "v", "sw", "sv", "q"):
        x = np.array([r[k] for r in runs])
        out[k] = {"mean": float(x.mean()), "half_ci95": float(student.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))}
    return out

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    for cid, case, seed0 in CASES:
        path = OUT / f"{cid}.json"
        if path.exists():
            continue
        case = dict(case, mean=M, id=cid)
        with ProcessPoolExecutor(workers) as ex:
            runs = list(ex.map(run, [case] * REPS, [seed0 + i for i in range(REPS)], [SAMPLE] * REPS, [SAMPLE // 10] * REPS))
        path.write_text(json.dumps({"case": case, "seed0": seed0, "reps": REPS, "sample": SAMPLE, "warmup": SAMPLE // 10,
                                    "summary": summarize(runs), "runs": runs}, indent=1))
        print(cid, json.dumps(summarize(runs)), flush=True)
