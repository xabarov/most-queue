"""v9, figure 3 data and computational cost (n=3, lambda=1, b1=2.1 unless noted).

Part 1: removal (RCS) and clearing by Takahashi-Takami + tagged equations for
delta from 0.05 to 1.5, CV=0.8 and 1.2: w1, v1, q. The H2 law is matched to
three Gamma moments, as in the article tables.
Part 2: wall-clock time and state counts of one removal-RCS calculation
(stationary probabilities + waiting and sojourn moments) for n=3, 5, 8:
TT + tagged equations (H2), the sparse joint model with the same H2, and the
joint model with the 9-phase positive Gamma representation (CV=0.8).
Output: delta-effect.json. Run from the repository root with PYTHONPATH=.
"""

import json
import time
from pathlib import Path

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107 import ph_queue
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph
from works.negative_queues.revision_v9_20261108 import tt_tagged

ROOT = Path(__file__).resolve().parent
B1 = 2.1
DELTAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]


def part1():
    out = {"deltas": DELTAS}
    for cv in (0.8, 1.2):
        b = gamma_moments_by_mean_and_cv(B1, cv)
        rows = {}
        for mode in ("rcs", "disaster"):
            rows[mode] = []
            for delta in DELTAS:
                r = tt_tagged.calculate(b, 3, 1.0, delta, mode, levels=240, behind=80)
                rows[mode].append({"delta": delta, "w1": r["w"][0], "v1": r["v"][0], "q": r["q"]})
                print(cv, mode, delta, round(r["w"][0], 4), round(r["q"], 4), flush=True)
        out[str(cv)] = rows
    return out


def timed(fn):
    t = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t


def part2():
    out = []
    for n in (3, 5, 8):
        lam = 0.7 * n / B1
        b = gamma_moments_by_mean_and_cv(B1, 0.8)
        h = H2Distribution.get_params(b)
        rec = {"n": n, "lam": lam}
        r, sec = timed(lambda: tt_tagged.calculate(b, n, lam, 0.3, "rcs", levels=240, behind=80))
        rec["tt"] = {"seconds": sec, "stationary_states": r["stationary_states"],
                     "tag_states": r["maximum_tag_block_states"], "w1": r["w"][0], "v1": r["v"][0]}
        y, mu = [complex(*a) for a in r["h2"]["y"]], [complex(*a) for a in r["h2"]["mu"]]
        r, sec = timed(lambda: ph_queue.calculate(ph_queue.hyperexponential(y, mu), n=n, lam=lam, delta=0.3,
                                                  mode="rcs", action="remove", levels=240, behind=80))
        rec["sparse_h2"] = {"seconds": sec, "stationary_states": r["stationary_states"],
                            "tag_states": r["maximum_tag_block_states"], "w1": r["w"][0], "v1": r["v"][0]}
        if n <= 5:
            r, sec = timed(lambda: ph_queue.calculate(gamma_ph(B1, 0.8, 8), n=n, lam=lam, delta=0.3, mode="rcs",
                                                      action="remove", levels=240, behind=80))
            rec["sparse_gamma9"] = {"seconds": sec, "stationary_states": r["stationary_states"],
                                    "tag_states": r["maximum_tag_block_states"], "w1": r["w"][0], "v1": r["v"][0]}
        print(json.dumps(rec), flush=True)
        out.append(rec)
    return out


if __name__ == "__main__":
    data = {"delta": part1(), "cost": part2()}
    (ROOT / "delta-effect.json").write_text(json.dumps(data, indent=1))
