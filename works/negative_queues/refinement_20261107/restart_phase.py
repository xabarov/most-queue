"""Direct phase equations for repeat-with-resampling, including shared resets.

Real H2 parameters define a CTMC. Complex moment fits are an algebraic
approximation. Stationary levels and arrivals behind the tag are truncated
separately; callers must repeat with larger limits.
"""

import math
import time

import numpy as np
from scipy.sparse import bmat, lil_matrix
from scipy.sparse.linalg import splu, spsolve


def phase_changes(busy, y, delta, mode):
    """Generator of phase changes at a fixed number of busy servers."""
    c = np.zeros((busy + 1, busy + 1), complex)
    if not busy:
        return c
    for j in range(busy + 1):
        if mode == "rcs":
            if j < busy:
                c[j, j + 1] = delta * (busy - j) * y[1] / busy
            if j:
                c[j, j - 1] = delta * j * y[0] / busy
        else:
            for k in range(busy + 1):
                if j != k:
                    c[j, k] = delta * math.comb(busy, k) * y[1] ** k * y[0] ** (busy - k)
        c[j, j] = -sum(c[j])
    return c


def completions(busy, y, mu, replace):
    d = np.zeros((busy + 1, busy + (1 if replace else 0)), complex)
    for j in range(busy + 1):
        for phase, count in [(0, busy - j), (1, j)]:
            if count:
                if replace:
                    for new_phase in [0, 1]:
                        d[j, j - phase + new_phase] += count * mu[phase] * y[new_phase]
                else:
                    d[j, j - phase] += count * mu[phase]
    return d


def stationary(n, lam, delta, y, mu, mode, levels):
    sizes = [min(i, n) + 1 for i in range(levels)]
    blocks = [[None] * levels for _ in range(levels)]
    for i, size in enumerate(sizes):
        busy = min(i, n)
        diagonal = phase_changes(busy, y, delta, mode)
        if i:
            down = completions(busy, y, mu, replace=i > n)
            blocks[i][i - 1] = down
            diagonal -= np.diag(down.sum(axis=1))
        if i < levels - 1:
            up = np.zeros((size, sizes[i + 1]), complex)
            for j in range(size):
                if i < n:
                    up[j, j : j + 2] = lam * np.asarray(y)
                else:
                    up[j, j] = lam
            blocks[i][i + 1] = up
            diagonal -= np.diag(up.sum(axis=1))
        blocks[i][i] = diagonal
    generator = bmat(blocks, format="csr")
    balance = generator.T.tolil()
    balance[-1, :] = 1
    rhs = np.zeros(sum(sizes), complex)
    rhs[-1] = 1
    weights = spsolve(balance.tocsc(), rhs)
    split = np.split(weights, np.cumsum(sizes)[:-1])
    return split, float(np.max(np.abs(weights @ generator)))


def service_tag(n, lam, delta, y, mu, mode, max_others, order):
    """Tag in service: (number of others, other phase-2 servers, tag phase)."""
    states = [(k, j, t) for k in range(max_others + 1) for j in range(min(k, n - 1) + 1) for t in [0, 1]]
    index = {state: row for row, state in enumerate(states)}
    q = lil_matrix((len(states), len(states)), dtype=complex)
    for row, (k, j, tag) in enumerate(states):
        busy = min(k, n - 1)

        def add(target, rate):
            if target != (k, j, tag) and rate:
                q[row, index[target]] += rate
                q[row, row] -= rate

        q[row, row] -= mu[tag]  # Successful completion absorbs the tag.
        if k < max_others:
            if k < n - 1:
                for phase in [0, 1]:
                    add((k + 1, j + phase, tag), lam * y[phase])
            else:
                add((k + 1, j, tag), lam)
        for phase, count in [(0, busy - j), (1, j)]:
            if not count:
                continue
            if k <= n - 1:
                add((k - 1, j - phase, tag), count * mu[phase])
            else:
                for new in [0, 1]:
                    add((k - 1, j - phase + new, tag), count * mu[phase] * y[new])
        if mode == "rcs":
            rate = delta / (busy + 1)
            add((k, j, 1 - tag), rate * y[1 - tag])
            if j < busy:
                add((k, j + 1, tag), rate * (busy - j) * y[1])
            if j:
                add((k, j - 1, tag), rate * j * y[0])
        else:
            for jnew in range(busy + 1):
                probability = math.comb(busy, jnew) * y[1] ** jnew * y[0] ** (busy - jnew)
                for tnew in [0, 1]:
                    add((k, jnew, tnew), delta * probability * y[tnew])
    solver = splu(-q.tocsc())
    moments = [np.ones(len(states), complex)]
    for r in range(1, order + 1):
        moments.append(solver.solve(r * moments[-1]))
    return states, index, moments


def calculate(n, lam, delta, y, mu, mode, levels=160, behind=160, order=4):
    """Raw W,V moments from stationary weights and first-step equations."""
    started = time.perf_counter()
    if mode not in ("rcs", "disaster") or lam <= 0 or delta < 0 or n < 1:
        raise ValueError("Invalid phase-model parameters.")
    if levels <= n or behind < 0 or order < 1:
        raise ValueError("Invalid truncation or moment order.")
    y, mu = np.asarray(y, complex), np.asarray(mu, complex)
    rate = delta / n if mode == "rcs" else delta
    beta = sum(y * mu / (mu + rate))
    mean = (1 - beta) / (rate * beta) if rate else sum(y / mu)
    if abs(mean.imag) > 1e-8 or mean.real <= 0 or lam * mean.real >= n:
        raise ValueError("The saturated phase model is not stable.")
    pi, residual = stationary(n, lam, delta, y, mu, mode, levels)
    p = np.array([sum(row) for row in pi])
    c = phase_changes(n, y, delta, mode)
    down = completions(n, y, mu, replace=True)
    leaving = completions(n, y, mu, replace=False)
    h = c - np.diag(down.sum(axis=1))
    inv_wait = np.linalg.inv(-h)
    waiting = np.zeros((order + 1, n + 1), complex)
    waiting[0] = 1
    w = np.zeros(order, complex)
    for level in range(n, levels):
        current = np.ones_like(waiting)
        for r in range(1, order + 1):
            current[r] = inv_wait @ (r * current[r - 1] + down @ waiting[r])
        for r in range(1, order + 1):
            w[r - 1] += pi[level] @ current[r]
        waiting = current

    _, index, tagged = service_tag(n, lam, delta, y, mu, mode, n - 1 + behind, order)
    v = np.zeros(order, complex)
    for level in range(n):
        for j, prob in enumerate(pi[level]):
            for phase in [0, 1]:
                for r in range(1, order + 1):
                    v[r - 1] += prob * y[phase] * tagged[r][index[level, j, phase]]
    # On the first freeing of a server, start the tag with an independent phase.
    start = np.ones((behind + 1, order + 1, n + 1), complex)
    for b in range(behind + 1):
        for r in range(1, order + 1):
            service = np.array([sum(y[t] * tagged[r][index[n - 1 + b, j, t]] for t in [0, 1]) for j in range(n)])
            start[b, r] = leaving @ service
    inverse = np.linalg.inv(lam * np.eye(n + 1) - h)
    previous = None
    for ahead in range(levels - n):
        current = np.ones((behind + 1, order + 1, n + 1), complex)
        for b in range(behind, -1, -1):
            for r in range(1, order + 1):
                rhs = r * current[b, r - 1]
                rhs += start[b, r] if ahead == 0 else down @ previous[b, r]
                if b < behind:
                    rhs += lam * current[b + 1, r]
                current[b, r] = (inverse if b < behind else inv_wait) @ rhs
        for r in range(1, order + 1):
            v[r - 1] += pi[n + ahead] @ current[0, r]
        previous = current
    imaginary = max(np.max(np.abs(p.imag)), np.max(np.abs(w.imag)), np.max(np.abs(v.imag)))
    if not np.all(np.isfinite([*p, *w, *v])) or imaginary > 1e-7 or p.real.min() < -1e-9:
        raise FloatingPointError("Invalid phase solution; the algebraic fit or truncation requires review.")
    little = [
        sum(max(i - n, 0) * pk for i, pk in enumerate(p)).real / lam,
        sum(i * pk for i, pk in enumerate(p)).real / lam,
    ]
    return dict(
        w=w.real.tolist(),
        v=v.real.tolist(),
        p=p.real.tolist(),
        sw=float(np.sqrt((w[1] - w[0] ** 2).real)) if order > 1 else None,
        sv=float(np.sqrt((v[1] - v[0] ** 2).real)) if order > 1 else None,
        q=1.0,
        levels=levels,
        behind=behind,
        tail=float(p[-1].real),
        balance_residual=residual,
        little_residuals=[abs(w[0].real - little[0]), abs(v[0].real - little[1])],
        seconds=time.perf_counter() - started,
    )
