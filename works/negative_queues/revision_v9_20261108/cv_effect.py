"""v9, figure 2 data: effect of the coefficient of variation (n=3, lambda=1, b1=2.1).

Panel a: mean time a customer occupies a server when all servers are busy,
relative to b1 (closed forms checked in diagnostics/effective_occupation.py).
Panel b: mean waiting time w1 for removal and clearing (Takahashi-Takami, H2 by
three Gamma moments, as in the article), restarts with a new length (joint
phase model, positive Gamma PH with 9 phases for CV<1, H2 for CV>1) and the
system without negative events. CV=1 is the exponential law.
Output: cv-effect.json. Run from the repository root with PYTHONPATH=.
"""

import json
from pathlib import Path

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107 import ph_queue
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph
from works.negative_queues.revision_v9_20261108 import tt_tagged

ROOT = Path(__file__).resolve().parent
B1, N, LAM, DELTA, DELTA_FIXED = 2.1, 3, 1.0, 0.3, 0.02
GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0]


def law(cv):
    if abs(cv - 1) < 1e-12:
        return ph_queue.hyperexponential([1.0], [1 / B1])
    if cv < 1:
        return gamma_ph(B1, cv, 8)
    h = H2Distribution.get_params(gamma_moments_by_mean_and_cv(B1, cv))
    return ph_queue.hyperexponential([h.p1, 1 - h.p1], [h.mu1, h.mu2])


def occupation(s):
    S, a = s.generator.real, s.alpha.real
    one = np.ones(s.size)
    ex = -S @ one
    beta = lambda r: a @ np.linalg.solve(r * np.eye(s.size) - S, ex)
    mgf = lambda r: a @ np.linalg.solve(-r * np.eye(s.size) - S, ex)
    out = {
        "rcs_remove": (1 - beta(DELTA / N)) / (DELTA / N),
        "clearing": (1 - beta(DELTA)) / DELTA,
        "rcs_restart": (1 - beta(DELTA / N)) / (DELTA / N * beta(DELTA / N)),
        "common_restart": (1 - beta(DELTA)) / (DELTA * beta(DELTA)),
        "common_retained": (mgf(DELTA_FIXED) - 1) / DELTA_FIXED,
    }
    return {k: float(v) / B1 for k, v in out.items()}


def waiting(cv):
    s = law(cv)
    out = {}
    for mode in ("rcs", "disaster"):
        if abs(cv - 1) < 1e-12:
            r = ph_queue.calculate(s, n=N, lam=LAM, delta=DELTA, mode=mode, action="remove", levels=240, behind=80)
        else:
            r = tt_tagged.calculate(gamma_moments_by_mean_and_cv(B1, cv), N, LAM, DELTA, mode, levels=240, behind=80)
        out["rcs_remove" if mode == "rcs" else "clearing"] = r["w"][0]
        r = ph_queue.calculate(s, n=N, lam=LAM, delta=DELTA, mode=mode, action="restart", levels=240, behind=80)
        out["rcs_restart" if mode == "rcs" else "common_restart"] = r["w"][0]
    out["no_events"] = ph_queue.calculate(s, n=N, lam=LAM, delta=0.0, action="restart", levels=240, behind=0)["w"][0]
    return out


def main():
    data = {"grid": GRID, "occupation": {}, "waiting": {}}
    for cv in GRID:
        data["occupation"][str(cv)] = occupation(law(cv))
        data["waiting"][str(cv)] = waiting(cv)
        print(cv, {k: round(v, 4) for k, v in data["waiting"][str(cv)].items()}, flush=True)
    (ROOT / "cv-effect.json").write_text(json.dumps(data, indent=1))


if __name__ == "__main__":
    main()
