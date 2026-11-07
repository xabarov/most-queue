"""v9: Takahashi-Takami stationary probabilities + tagged-customer moment equations.

Stationary microstate probabilities of the two-phase model are obtained by the
Takahashi-Takami iterations of most_queue (complex arithmetic, so the H2 fitted
to three moments may have a negative weight or complex parameters). For the
clearing model the TT scheme uses auxiliary states D_i; their mass is deleted
and the ordinary states are renormalized (physical process with instantaneous
clearing).

Waiting and sojourn moments are then computed from the same probabilities by
the linear first-step equations for the tagged customer (v7 ph_queue code):
no numerical differentiation of transforms, and the RCS rate delta/M(t) changes
with the current number of busy servers during the tagged service.

The microstate order coincides: level i, j = number of busy servers in phase 2.
"""

from contextlib import contextmanager

import numpy as np

from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc
from works.negative_queues.revision_v7_20261107 import ph_queue


def tt_solve(b, n, lam, delta, mode, levels=200, tolerance=1e-12):
    cls = MGnNegativeRCSCalc if mode == "rcs" else MGnNegativeDisasterCalc
    calc = cls(n=n, calc_params=TakahashiTakamiParams(tolerance=tolerance, N=levels))
    calc.set_sources(l_pos=lam, l_neg=delta)
    calc.set_servers(b=b)
    library = calc.run()
    if mode == "rcs":
        pi = [np.array(calc.Y[i][0, : min(i, n) + 1], dtype=complex) for i in range(calc.N)]
    else:
        pi = [np.array([calc.Y[0][0, 0]], dtype=complex)]
        pi += [np.array(calc.Y[i][0, 1 : min(i, n) + 2], dtype=complex) for i in range(1, calc.N)]
    mass = sum(a.sum() for a in pi)
    pi = [a / mass for a in pi]
    service = ph_queue.hyperexponential(list(calc.y), list(calc.mu))
    return calc, library, service, pi


@contextmanager
def _given_stationary(pi):
    original = ph_queue.stationary

    def replacement(parts, n, lam, delta, mode, action, levels, **_):
        assert len(pi) == levels, (len(pi), levels)
        for i, a in enumerate(pi):
            assert len(a) == len(parts["states"][min(i, n)])
        blocks = ph_queue.physical_blocks(parts, n, delta, mode, action)
        inside, down, replacement_block = blocks
        # Residual of the physical balance equations for the supplied vector.
        residual = 0.0
        clearing = action == "remove" and mode == "disaster"
        for i, prob in enumerate(pi):
            m = min(i, n)
            value = inside[m].T @ prob
            if i < levels - 1:
                value = value - lam * prob
            if i + 1 < levels:
                value = value + (down[m + 1] if i < n else replacement_block).T @ pi[i + 1]
            if i:
                value = value + lam * (parts["admission"][m - 1].T @ pi[i - 1] if i <= n else pi[i - 1])
            if clearing and i == 0:
                value = value + delta * sum(a.sum() for a in pi[1:])
            residual = max(residual, float(np.max(np.abs(value))))
        return list(pi), blocks, dict(balance_residual=residual, stationary_products=0, stationary_source="TT")

    ph_queue.stationary = replacement
    try:
        yield
    finally:
        ph_queue.stationary = original


def calculate(b, n, lam, delta, mode, levels=200, behind=60, tolerance=1e-12):
    calc, library, service, pi = tt_solve(b, n, lam, delta, mode, levels, tolerance)
    with _given_stationary(pi):
        result = ph_queue.calculate(service, n=n, lam=lam, delta=delta, mode=mode, action="remove",
                                    levels=len(pi), behind=behind)
    result["tt_library"] = {"w": [float(np.real(x)) for x in library.w[:2]],
                            "v": [float(np.real(x)) for x in library.v[:2]],
                            "note": "moments by numerical differentiation of LST; RCS sojourn with constant rate"}
    result["h2"] = {"y": [complex(x) for x in calc.y], "mu": [complex(x) for x in calc.mu]}
    result["h2"] = {k: [[v.real, v.imag] for v in vals] for k, vals in result["h2"].items()}
    result["tt_iterations"] = int(getattr(calc, "num_of_iter_", -1))
    return result
