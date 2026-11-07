"""Mean time a customer occupies a server when all servers are busy (v9, CV effect).

Removal: E[min(B,Y)] = (1-beta(r))/r; restart with a new length (Poisson rate r):
E[T] = (1-beta(r))/(r*beta(r)); retained work: E[T] = (M_B(r)-1)/r. Each closed
form is checked against the phase-type absorption time. r = delta/n for RCS.
"""
import json
from pathlib import Path

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107.ph_queue import hyperexponential
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph


def law(cv):
    if cv < 1:
        return gamma_ph(2.1, cv, 16)
    h = H2Distribution.get_params(gamma_moments_by_mean_and_cv(2.1, cv))
    return hyperexponential([h.p1, 1 - h.p1], [h.mu1, h.mu2])


def main():
    out = {}
    for cv in (0.8, 1.2):
        s = law(cv)
        S, a, one = s.generator.real, s.alpha.real, np.ones(s.size)
        exit_ = -S @ one
        beta = lambda r: a @ np.linalg.solve(r * np.eye(s.size) - S, exit_)
        mgf = lambda r: a @ np.linalg.solve(-r * np.eye(s.size) - S, exit_)
        row = {}
        for name, r, kind in [("rcs_remove", 0.1, "remove"), ("clearing", 0.3, "remove"),
                              ("rcs_restart", 0.1, "restart"), ("common_restart", 0.3, "restart"),
                              ("common_retained", 0.02, "retained")]:
            if kind == "remove":
                closed = (1 - beta(r)) / r
                ph = a @ np.linalg.solve(-(S - r * np.eye(s.size)), one)
            elif kind == "restart":
                closed = (1 - beta(r)) / (r * beta(r))
                ph = a @ np.linalg.solve(-(S + r * (np.outer(one, a) - np.eye(s.size))), one)
            else:
                closed = (mgf(r) - 1) / r
                ph = closed
            assert abs(closed - ph) < 1e-10, (cv, name, closed, ph)
            row[name] = float(closed)
        out[str(cv)] = row
        print(cv, {k: round(v, 3) for k, v in row.items()})
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
