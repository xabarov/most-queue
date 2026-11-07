"""Independent finite CTMC reference; no production transition blocks are reused."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.theory.calc_params import TakahashiTakamiParams
from most_queue.theory.negative.mgn_disaster import MGnNegativeDisasterCalc
from most_queue.theory.negative.mgn_rcs import MGnNegativeRCSCalc


def stationary(n, lam, delta, y, mu, mode, levels=80):
    states = [(i, j) for i in range(levels) for j in range(min(i, n) + 1)]
    index = {s: k for k, s in enumerate(states)}
    q = lil_matrix((len(states), len(states)), dtype=complex)
    for row, (i, j) in enumerate(states):

        def add(target, rate):
            if rate and target != (i, j):
                q[row, index[target]] += rate
                q[row, row] -= rate

        if i < levels - 1:
            if i < n:
                add((i + 1, j), lam * y[0])
                add((i + 1, j + 1), lam * y[1])
            else:
                add((i + 1, j), lam)
        if i == 0:
            continue
        if mode == "disaster":
            add((0, 0), delta)
        busy = min(i, n)
        hazard = delta / busy if mode == "rcs" else 0
        d1 = (busy - j) * (mu[0] + hazard)
        d2 = j * (mu[1] + hazard)
        if i <= n:
            if busy > j:
                add((i - 1, j), d1)
            if j:
                add((i - 1, j - 1), d2)
        else:
            add((i - 1, j), d1 * y[0] + d2 * y[1])
            if j < n:
                add((i - 1, j + 1), d1 * y[1])
            if j:
                add((i - 1, j - 1), d2 * y[0])
    a = q.T.tolil()
    a[-1, :] = 1
    rhs = np.zeros(len(states), complex)
    rhs[-1] = 1
    pi = spsolve(a.tocsc(), rhs)
    p = np.zeros(levels, complex)
    for prob, (i, j) in zip(pi, states):
        p[i] += prob
    return states, pi, p.real, float(np.max(np.abs(pi @ q.tocsr())))


def compare(n=3, cv=1.2, delta=0.3, gamma=None, levels=80):
    b = gamma_moments_by_mean_and_cv(0.7 * n, cv)
    rows = []
    for mode, cls in [("rcs", MGnNegativeRCSCalc), ("disaster", MGnNegativeDisasterCalc)]:
        calc = cls(n=n, calc_params=TakahashiTakamiParams(N=levels, tolerance=1e-12))
        calc.set_sources(1.0, delta)
        calc.set_servers(b)
        if gamma is not None and mode == "disaster":
            calc.gamma = gamma
        res = calc.run()
        initial_p = np.array(res.p).copy()
        states, pi, p, residual = stationary(n, 1.0, delta, calc.y, calc.mu, mode, levels)
        little_w = sum(max(i - n, 0) * pk for i, pk in enumerate(p))
        little_v = sum(i * pk for i, pk in enumerate(p))
        q_service = sum(prob * ((min(i, n) - j) * calc.mu[0] + j * calc.mu[1]) for prob, (i, j) in zip(pi, states)).real
        rows.append(
            dict(
                mode=mode,
                n=n,
                cv=cv,
                delta=delta,
                gamma=gamma,
                levels=levels,
                w=[float(np.real(x)) for x in res.w[:2]],
                v=[float(np.real(x)) for x in res.v[:2]],
                q=float(res.q),
                reference_q=float(q_service),
                little_w=little_w,
                little_v=little_v,
                stationary_error=float(max(abs(np.array(res.p) - p))),
                repeat_get_p_error=float(max(abs(np.array(calc.get_p()) - initial_p))),
                residual=residual,
                tail=float(p[-1]),
                w_relative_error=float(np.real(res.w[0] / little_w - 1)),
                v_relative_error=float(np.real(res.v[0] / little_v - 1)),
            )
        )
    return rows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reference-checks.json")
    args = parser.parse_args()
    rows = compare()
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    Path(args.output).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
