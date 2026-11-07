"""Joint finite-state approximation of fixed-work restart queues.

Each job retains a work class throughout all repeats. An Erlang clock approximates
deterministic execution of that class's work, and a common reset simultaneously
returns every busy job to phase zero of its OWN class. Both the work quadrature
and Erlang order must be refined; finite-order results are not the exact PDMP.
"""

import argparse
import itertools
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, bicgstab, lgmres
from scipy.special import gamma, roots_genlaguerre, roots_laguerre

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv


def work_quadrature(mean, cv, classes):
    if cv < 1:
        shape, scale = 1 / cv**2, mean * cv**2
        x, weights = roots_genlaguerre(classes, shape - 1)
        return scale * x, weights / gamma(shape)
    h = H2Distribution.get_params(gamma_moments_by_mean_and_cv(mean, cv))
    # Compress an accurate quadrature of the true H2 law by Lanczos. The small
    # Gaussian rule matches its first 2*classes-1 moments, not a Gamma substitute.
    x, w = roots_laguerre(max(32, classes + 2))
    nodes = np.r_[x / h.mu1, x / h.mu2]
    weights = np.r_[h.p1 * w, (1 - h.p1) * w]
    v = np.sqrt(weights)
    v /= np.linalg.norm(v)
    basis, diagonal, off = [], [], []
    for k in range(classes):
        basis.append(v)
        z = nodes * v
        diagonal.append(float(v @ z))
        for q in basis:
            z -= (q @ z) * q
        if k < classes - 1:
            norm = np.linalg.norm(z)
            off.append(norm)
            v = z / norm
    nodes, vectors = eigh_tridiagonal(diagonal, off)
    return nodes, vectors[0] ** 2


def service_moments(work, weights, stages, delta, order=3):
    """Mixture over retained classes; phase durations resample within a class."""
    result = np.zeros(order, np.longdouble)
    for b, weight in zip(work, weights):
        b, r = np.longdouble(b), np.longdouble(delta)
        if not delta:
            raw = [b**k * math.prod(1 + j / stages for j in range(k)) for k in range(1, order + 1)]
        else:
            scale = b / stages
            c = [(1 + scale * r) ** (-stages)]
            for k in range(1, order + 1):
                c.append(-c[-1] * (stages + k - 1) * scale / (k * (1 + scale * r)))
            num = [r * c[k] + (c[k - 1] if k else 0) for k in range(order + 1)]
            den = [r * c[k] + (1 if k == 1 else 0) for k in range(order + 1)]
            q = []
            for k in range(order + 1):
                q.append((num[k] - sum(den[j] * q[k - j] for j in range(1, k + 1))) / den[0])
            raw = [(-1) ** k * math.factorial(k) * q[k] for k in range(1, order + 1)]
        result += np.longdouble(weight) * np.array(raw)
    return np.asarray(result, float)


def blocks(n, lam, delta, work, weights, stages, mode):
    phases = len(work) * stages
    states = [list(itertools.combinations_with_replacement(range(phases), m)) for m in range(n + 1)]
    indices = [{s: i for i, s in enumerate(group)} for group in states]
    internals, downs, ups = [], [], []

    def sparse(entries, rows, columns):
        if not entries:
            return coo_matrix((rows, columns)).tocsr()
        rr, cc, vv = zip(*entries)
        return coo_matrix((vv, (rr, cc)), shape=(rows, columns)).tocsr()

    for m, group in enumerate(states):
        internal, down, up = [], [], []
        for row, state in enumerate(group):
            if m < n:
                for c, w in enumerate(weights):
                    target = tuple(sorted((*state, c * stages)))
                    up.append((row, indices[m + 1][target], lam * w))
            for p, count in Counter(state).items():
                rest = list(state)
                rest.remove(p)
                c, step = divmod(p, stages)
                rate = count * stages / work[c]
                if step < stages - 1:
                    target = tuple(sorted((*rest, p + 1)))
                    internal.append((row, indices[m][target], rate))
                else:
                    down.append((row, indices[m - 1][tuple(rest)], rate))
                if mode == "rcs" and step:
                    target = tuple(sorted((*rest, c * stages)))
                    internal.append((row, indices[m][target], delta * count / m))
            if mode == "disaster" and m:
                target = tuple(sorted(p // stages * stages for p in state))
                if target != state:
                    internal.append((row, indices[m][target], delta))
        internals.append(sparse(internal, len(group), len(group)))
        downs.append(sparse(down, len(group), len(states[m - 1]) if m else 1))
        ups.append(sparse(up, len(group), len(states[m + 1]) if m < n else len(group)))
    # A completion above level n immediately installs a new job, with an
    # independent work class. Compose removal and admission, without a time step.
    replacement = downs[n] @ (ups[n - 1] / lam)
    return states, internals, downs, ups, replacement


def solve_jump_chain(low_rates, high_rates, inside, down, up, replacement, n, lam, tolerance, progress):
    sizes = [len(r) for r in low_rates]
    offsets = np.r_[0, np.cumsum(sizes)]
    total = int(offsets[-1] + high_rates.size)
    calls = 0

    def apply_jump(vector):
        lo = [vector[offsets[i] : offsets[i + 1]] / low_rates[i] for i in range(n)]
        hi = vector[offsets[-1] :].reshape(high_rates.shape) / high_rates
        result = np.empty_like(vector)
        for i in range(n):
            value = inside[i].T @ lo[i]
            if i:
                value += up[i - 1].T @ lo[i - 1]
            value += down[i + 1].T @ (lo[i + 1] if i + 1 < n else hi[0])
            result[offsets[i] : offsets[i + 1]] = value
        high = (inside[n].T @ hi.T).T
        high[0] += up[n - 1].T @ lo[n - 1]
        high[1:] += lam * hi[:-1]
        high[:-1] += (replacement.T @ hi[1:].T).T
        result[offsets[-1] :] = high.ravel()
        return result

    def matvec(vector):
        nonlocal calls
        calls += 1
        if progress and calls % 500 == 0:
            print("Krylov products:", calls, flush=True)
        result = vector - apply_jump(vector)
        result[0] += vector.sum()
        return result

    a = LinearOperator((total, total), matvec=matvec, dtype=float)
    rhs = np.zeros(total)
    rhs[0] = 1
    solution, info = bicgstab(a, rhs, rtol=min(tolerance * 0.01, 1e-12), atol=1e-14, maxiter=5000)
    if info:
        solution, info = lgmres(
            a, rhs, x0=solution, rtol=min(tolerance * 0.01, 1e-12), atol=1e-14, maxiter=300, inner_m=40
        )
    if info or not np.all(np.isfinite(solution)):
        raise RuntimeError(f"Stationary Krylov solve failed: {info}")
    solution /= solution.sum()
    residual = np.abs(solution - apply_jump(solution)).sum()
    if solution.min() < -1e-10 or residual > max(1e-8, 100 * tolerance):
        raise FloatingPointError(f"Invalid stationary vector: min={solution.min()}, residual={residual}")
    return (
        [solution[offsets[i] : offsets[i + 1]] for i in range(n)],
        solution[offsets[-1] :].reshape(high_rates.shape),
        calls,
    )


def calculate(
    cv=1.2,
    mode="disaster",
    n=3,
    lam=1,
    delta=0.02,
    classes=4,
    stages=4,
    levels=80,
    tolerance=1e-10,
    max_iter=40000,
    progress=False,
    method="power",
):
    started = time.perf_counter()
    work, weights = work_quadrature(2.1, cv, classes)
    states, inside, down, up, replacement = blocks(n, lam, delta, work, weights, stages, mode)
    d = len(states[n])
    high_count = levels - n
    if high_count < 2:
        raise ValueError("At least two saturated levels are required.")
    if mode == "disaster" and lam * service_moments(work, weights, stages, delta)[0] >= n:
        raise ValueError("Unstable approximating model.")
    low_rates = []
    for m in range(n):
        low_rates.append(np.asarray(inside[m].sum(axis=1) + down[m].sum(axis=1)).ravel() + lam)
    service_rate = np.asarray(down[n].sum(axis=1)).ravel()
    rate = np.asarray(inside[n].sum(axis=1)).ravel() + service_rate
    high_rates = np.tile(rate + lam, (high_count, 1))
    high_rates[-1] -= lam
    low = [np.zeros(len(s)) for s in states[:n]]
    low[0][0] = 1
    high = np.zeros((high_count, d))
    linear_products = 0
    if method == "krylov":
        low, high, linear_products = solve_jump_chain(
            low_rates, high_rates, inside, down, up, replacement, n, lam, tolerance, progress
        )
    previous = None
    error = math.inf
    for iteration in range(1, max_iter + 1):
        lo = [p / r for p, r in zip(low, low_rates)]
        hi = high / high_rates
        incoming = [inside[m].T @ lo[m] for m in range(n)]
        for m in range(n):
            if m:
                incoming[m] += up[m - 1].T @ lo[m - 1]
            incoming[m] += down[m + 1].T @ (lo[m + 1] if m + 1 < n else hi[0])
        high_in = (inside[n].T @ hi.T).T
        high_in[0] += up[n - 1].T @ lo[n - 1]
        high_in[1:] += lam * hi[:-1]
        high_in[:-1] += (replacement.T @ hi[1:].T).T
        if iteration % 25 == 0:
            jump_residual = float(
                sum(np.abs(a - b).sum() for a, b in zip(incoming, low)) + np.abs(high_in - high).sum()
            )
        low = [0.5 * (old + new) for old, new in zip(low, incoming)]
        high = 0.5 * (high + high_in)
        if iteration % 25:
            continue
        pi_low = [p / r for p, r in zip(low, low_rates)]
        pi_high = high / high_rates
        total = sum(p.sum() for p in pi_low) + pi_high.sum()
        p = np.r_[[a.sum() / total for a in pi_low], pi_high.sum(axis=1) / total]
        q = np.maximum(np.arange(levels) - n, 0)
        metrics = np.array([p @ q / lam, p @ (q * (q - 1)) / lam**2, p @ np.arange(levels) / lam])
        if previous is not None:
            error = float(np.max(np.abs(metrics - previous) / (1 + np.abs(metrics))))
        previous = metrics
        if progress and iteration % 500 == 0:
            print(f"J={classes} K={stages} iteration={iteration} error={error:.3g} W={metrics[0]:.8g}", flush=True)
        if iteration >= (50 if method == "krylov" else 500) and error < tolerance and jump_residual < 100 * tolerance:
            break
    else:
        raise RuntimeError(f"Embedded-chain iteration failed: error={error}, iterations={iteration}")
    w1, w2, v1 = metrics
    result = dict(
        cv=cv,
        mode=mode,
        n=n,
        lam=lam,
        delta=delta,
        classes=classes,
        stages=stages,
        levels=levels,
        phase_states=d,
        total_states=sum(len(s) for s in states[:n]) + high_count * d,
        iterations=iteration,
        solver=method,
        linear_products=linear_products,
        iteration_change=error,
        w=[float(w1), float(w2)],
        v=[float(v1)],
        embedded_chain_residual_l1=jump_residual,
        sw=float(np.sqrt(w2 - w1 * w1)),
        tail=float(p[-1]),
        p=p.tolist(),
        work=work.tolist(),
        weights=weights.tolist(),
        seconds=time.perf_counter() - started,
    )
    if mode == "disaster":
        service = service_moments(work, weights, stages, delta)
        v2 = w2 + 2 * w1 * service[0] + service[1]
        result.update(
            v=[float(v1), float(v2)],
            sv=float(np.sqrt(v2 - v1 * v1)),
            service_moments=service.tolist(),
            service_mean_residual=float(abs(v1 - w1 - service[0])),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=float, default=1.2)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--levels", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calculate(cv=args.cv, classes=args.classes, stages=args.stages, levels=args.levels, progress=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k not in ["p", "work", "weights"]}, flush=True)
