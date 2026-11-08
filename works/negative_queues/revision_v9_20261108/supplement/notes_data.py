"""Data for ph_gamma_notes.md (supplementary note on the PH-representation of the
Gamma law and on the H2 substitution error).

Part A: H2 parameters fitted to three Gamma moments on a CV grid (where the
weight becomes negative, where the parameters become complex).
Part B: law-substitution error on a CV grid: Takahashi-Takami with H2 against
the PH-representation of the Gamma law with 16 quadrature nodes (n=3, lambda=1,
b1=2.1, delta=0.3; removal RCS and clearing).
Part C: convergence of w1 with the number of quadrature nodes J (CV=0.8, RCS
removal and clearing) and the maximum CDF deviation of the J-node law.
Output: notes-data.json. Run from the repository root with PYTHONPATH=.
One process; each sparse 17-phase calculation takes up to a minute.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.stats import gamma as gamma_law

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107 import ph_queue
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph
from works.negative_queues.revision_v9_20261108 import tt_tagged

ROOT = Path(__file__).resolve().parent
B1 = 2.1


def h2_grid():
    out = []
    for cv in np.round(np.arange(0.40, 2.01, 0.02), 2):
        b = gamma_moments_by_mean_and_cv(B1, float(cv))
        p = H2Distribution.get_params_clx(b) if cv < 1 else H2Distribution.get_params(b)
        y1, mu1, mu2 = complex(p.p1), complex(p.mu1), complex(p.mu2)
        out.append({"cv": float(cv), "y1": [y1.real, y1.imag], "mu1": [mu1.real, mu1.imag],
                    "mu2": [mu2.real, mu2.imag]})
    return out


def cdf_deviation(cv, nodes, grid):
    s = gamma_ph(B1, cv, nodes)
    a, T, t0 = s.alpha, s.generator, s.exit
    shape = 1 / cv**2
    exact = gamma_law.cdf(grid, shape, scale=B1 / shape)
    approx = np.array([1 - a @ expm(T * t) @ np.ones(len(a)) for t in grid])
    return float(np.max(np.abs(approx - exact))), s.size


def substitution_grid():
    rows = []
    for cv in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        b = gamma_moments_by_mean_and_cv(B1, cv)
        ref = gamma_ph(B1, cv, 16)
        for mode in ("rcs", "disaster"):
            t = time.perf_counter()
            tt = tt_tagged.calculate(b, 3, 1.0, 0.3, mode, levels=200, behind=60)
            g = ph_queue.calculate(ref, n=3, lam=1.0, delta=0.3, mode=mode, action="remove", levels=200, behind=60)
            rows.append({"cv": cv, "mode": mode, "phases_ref": ref.size,
                         "w1_h2": tt["w"][0], "v1_h2": tt["v"][0], "w1_gamma": g["w"][0], "v1_gamma": g["v"][0],
                         "dw_percent": 100 * (tt["w"][0] / g["w"][0] - 1),
                         "dv_percent": 100 * (tt["v"][0] / g["v"][0] - 1),
                         "seconds": time.perf_counter() - t})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def convergence():
    grid = np.linspace(0.0, 12 * B1, 1201)
    rows = []
    for nodes in range(1, 11):
        dev, size = cdf_deviation(0.8, nodes, grid)
        s = gamma_ph(B1, 0.8, nodes)
        rec = {"nodes": nodes, "phases": size, "cdf_deviation": dev}
        for mode in ("rcs", "disaster"):
            g = ph_queue.calculate(s, n=3, lam=1.0, delta=0.3, mode=mode, action="remove", levels=200, behind=60)
            rec[f"w1_{mode}"] = g["w"][0]
        print(json.dumps(rec), flush=True)
        rows.append(rec)
    return rows


if __name__ == "__main__":
    data = {"h2": h2_grid(), "convergence": convergence(), "substitution": substitution_grid()}
    (ROOT / "notes-data.json").write_text(json.dumps(data, indent=1))
    print("written", ROOT / "notes-data.json")
