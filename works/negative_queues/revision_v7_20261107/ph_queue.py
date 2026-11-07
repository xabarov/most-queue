"""Joint PH queues with exact tagged-customer equations for each finite model.

The stationary level cutoff and arrivals behind the tag are separate limits.
PH service may be probabilistic or an explicitly identified algebraic H2 fit.
Retained-work models keep the work class when resetting its Erlang phase.
"""

import itertools
import math
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import bicgstab, lgmres, LinearOperator, splu, spilu, spsolve

from works.negative_queues.refinement_20261107.fixed_work_phase import solve_jump_chain, work_quadrature


@dataclass
class Service:
    alpha: np.ndarray
    generator: np.ndarray
    reset: np.ndarray
    groups: list
    name: str

    @property
    def size(self):
        return len(self.alpha)

    @property
    def exit(self):
        return -self.generator.sum(axis=1)

    def moments(self, order=3):
        values = [np.ones(self.size)]
        for k in range(1, order + 1):
            values.append(np.linalg.solve(-self.generator, k * values[-1]))
        return [float(np.real(self.alpha @ a)) for a in values[1:]]


def hyperexponential(y, mu):
    y, mu = np.asarray(y, complex), np.asarray(mu, complex)
    return Service(y, -np.diag(mu), np.tile(y, (len(y), 1)), [list(range(len(y)))], "H2")


def fixed_work(cv, classes, stages, mean=2.1):
    work, weights = work_quadrature(mean, cv, classes)
    size = classes * stages
    alpha, generator, reset = np.zeros(size), np.zeros((size, size)), np.zeros((size, size))
    groups = []
    for c, (b, weight) in enumerate(zip(work, weights)):
        group = list(range(c * stages, (c + 1) * stages))
        groups.append(group)
        alpha[group[0]] = weight
        for step, p in enumerate(group):
            generator[p, p] = -stages / b
            if step + 1 < stages:
                generator[p, p + 1] = stages / b
            reset[p, group[0]] = 1
    return Service(alpha, generator, reset, groups, f"retained work J={classes} K={stages}")


def occurrence_blocks(service, n):
    """Unordered busy phases; generators retain absorption on the diagonal."""
    size, alpha, s, reset = service.size, service.alpha, service.generator, service.reset
    dtype = np.result_type(alpha, s)
    states = [list(itertools.combinations_with_replacement(range(size), m)) for m in range(n + 1)]
    indices = [{state: k for k, state in enumerate(group)} for group in states]
    internal, down, admission, single, common, removal = [], [], [], [], [], []

    def matrix(entries, rows, cols):
        if not entries:
            return sp.csr_matrix((rows, cols), dtype=dtype)
        rr, cc, vv = zip(*entries)
        return sp.coo_matrix((vv, (rr, cc)), shape=(rows, cols), dtype=dtype).tocsr()

    nonzero_s = [[q for q in range(size) if q != p and s[p, q] != 0] for p in range(size)]
    nonzero_r = [[q for q in range(size) if reset[p, q] != 0] for p in range(size)]
    starts = np.flatnonzero(alpha)
    for m, group in enumerate(states):
        se, de, ae, re, ce, rm = [], [], [], [], [], []
        for row, state in enumerate(group):
            for p, count in Counter(state).items():
                rest = list(state)
                rest.remove(p)
                se.append((row, row, count * s[p, p]))
                for q in nonzero_s[p]:
                    se.append((row, indices[m][tuple(sorted((*rest, q)))], count * s[p, q]))
                de.append((row, indices[m - 1][tuple(rest)], count * service.exit[p]))
                rm.append((row, indices[m - 1][tuple(rest)], count))
                for q in nonzero_r[p]:
                    if q != p:
                        re.append((row, indices[m][tuple(sorted((*rest, q)))], count * reset[p, q]))
                        re.append((row, row, -count * reset[p, q]))
            if m < n:
                for p in starts:
                    ae.append((row, indices[m + 1][tuple(sorted((*state, p)))], alpha[p]))
            for target in itertools.product(*(nonzero_r[p] for p in state)):
                probability = math.prod(reset[p, q] for p, q in zip(state, target))
                ce.append((row, indices[m][tuple(sorted(target))], probability))
        d = len(group)
        internal.append(matrix(se, d, d))
        down.append(matrix(de, d, len(states[m - 1]) if m else 1))
        admission.append(matrix(ae, d, len(states[m + 1]) if m < n else d))
        single.append(matrix(re, d, d))
        common.append(matrix(ce, d, d))
        removal.append(matrix(rm, d, len(states[m - 1]) if m else 1))
    return dict(
        states=states, internal=internal, down=down, admission=admission, single=single, common=common, removal=removal
    )


def physical_blocks(parts, n, delta, mode, action):
    inside, down = [], []
    for m, states in enumerate(parts["states"]):
        g, d = parts["internal"][m].copy(), parts["down"][m].copy()
        if m:
            if action == "remove":
                g -= delta * sp.eye(len(states))
                if mode == "rcs":
                    d += delta / m * parts["removal"][m]
            elif mode == "rcs":
                g += delta / m * parts["single"][m]
            else:
                g += delta * (parts["common"][m] - sp.eye(len(states)))
        inside.append(g.tocsr())
        down.append(d.tocsr())
    replacement = down[n] @ parts["admission"][n - 1]
    return inside, down, replacement


def stationary(parts, n, lam, delta, mode, action, levels, tolerance=1e-10, progress=False, clearing_resolvent=True):
    inside, down, replacement = physical_blocks(parts, n, delta, mode, action)
    sizes = [len(parts["states"][min(i, n)]) for i in range(levels)]
    clearing = action == "remove" and mode == "disaster"
    probabilistic = not any(np.iscomplexobj(g.data) for g in inside)
    # Large positive QBDs use the same matrix-free stationarity engine as v6.
    if sum(sizes) > 30000 and probabilistic and not clearing:
        off = [g - sp.diags(g.diagonal()) for g in inside]
        low_rates = [-inside[m].diagonal() + lam for m in range(n)]
        high_rates = np.tile(-inside[n].diagonal() + lam, (levels - n, 1))
        high_rates[-1] -= lam
        up = [lam * a for a in parts["admission"]]
        low, high, calls = solve_jump_chain(
            low_rates, high_rates, off, down, up, replacement, n, lam, tolerance, progress
        )
        pi = [a / r for a, r in zip(low, low_rates)] + list(high / high_rates)
        norm = sum(a.sum() for a in pi)
        pi = [a / norm for a in pi]
        residual = 0.0
        for i, prob in enumerate(pi):
            m = min(i, n)
            value = inside[m].T @ prob
            if i < levels - 1:
                value -= lam * prob
                value += (down[m + 1] if i < n else replacement).T @ pi[i + 1]
            if i:
                value += lam * (parts["admission"][m - 1].T @ pi[i - 1] if i <= n else pi[i - 1])
            residual = max(residual, float(np.max(np.abs(value))))
    else:
        use_resolvent = clearing and delta > 0 and clearing_resolvent
        blocks = [[None] * levels for _ in range(levels)]
        for i in range(levels):
            m = min(i, n)
            g = inside[m].copy()
            if i:
                blocks[i][i - 1] = down[m] if i <= n else replacement
            if use_resolvent and i == 0:
                g -= delta * sp.eye(sizes[i])
            if clearing and i and not use_resolvent:
                reset = sp.csr_matrix(np.full((sizes[i], 1), delta))
                blocks[i][0] = reset if blocks[i][0] is None else blocks[i][0] + reset
            if i < levels - 1:
                blocks[i][i + 1] = lam * (parts["admission"][m] if i < n else sp.eye(sizes[i]))
                g -= lam * sp.eye(sizes[i])
            blocks[i][i] = g
        generator = sp.bmat(blocks, format="csr")
        rhs = np.zeros(sum(sizes), dtype=generator.dtype)
        if use_resolvent:
            # generator=Q0-delta*I; stationary resetting to empty gives
            # (delta*I-Q0).T*pi=delta*e0. Avoid a dense normalization row.
            rhs[0] = delta
            weights = spsolve(-generator.T.tocsc(), rhs)
            balance = weights @ generator
            balance[0] += delta * weights.sum()
            residual = float(np.max(np.abs(balance)))
        else:
            a = generator.T.tolil()
            a[-1, :] = 1
            rhs[-1] = 1
            weights = spsolve(a.tocsc(), rhs)
            residual = float(np.max(np.abs(weights @ generator)))
        pi = list(np.split(weights, np.cumsum(sizes)[:-1]))
        calls = 0
    return pi, (inside, down, replacement), dict(balance_residual=residual, stationary_products=calls)


def factor(matrix, direct=False):
    """Reusable direct/iterative solver, with an independently checked residual."""
    a = matrix.tocsc()
    if direct or a.shape[0] < 18000 or np.iscomplexobj(a.data):
        lu = splu(a)
        return lambda rhs, initial=None: lu.solve(rhs)
    diagonal = a.diagonal()
    preconditioner = LinearOperator(a.shape, matvec=lambda x: x / diagonal, dtype=a.dtype)

    def solve(rhs, initial=None):
        x, info = bicgstab(a, rhs, x0=initial, M=preconditioner, rtol=3e-11, atol=1e-12, maxiter=2000)
        if info:
            ilu = spilu(a, drop_tol=2e-4, fill_factor=12)
            stronger = LinearOperator(a.shape, matvec=ilu.solve, dtype=a.dtype)
            x, info = lgmres(a, rhs, x0=x, M=stronger, rtol=3e-11, atol=1e-12, maxiter=100)
        error = np.linalg.norm(a @ x - rhs, np.inf) / max(1, np.linalg.norm(rhs, np.inf))
        if info or error > 1e-7:
            raise RuntimeError(f"Tagged equation did not converge: info={info}, residual={error}")
        return x

    return solve


def tag_service(service, parts, n, lam, delta, mode, action, max_others, order):
    """Solve separately for invariant retained-work classes of the tag."""
    result = [[None] * (order + 1) for _ in range(max_others + 1)]
    groups = service.groups if action != "remove" else [list(range(service.size))]
    maximum_size = 0
    for group in groups:
        tag_s = sp.csr_matrix(service.generator[np.ix_(group, group)])
        tag_r = sp.csr_matrix(service.reset[np.ix_(group, group)])
        identity = sp.eye(len(group))
        dimensions = [len(group) * len(parts["states"][min(k, n - 1)]) for k in range(max_others + 1)]
        blocks = [[None] * (max_others + 1) for _ in range(max_others + 1)]
        for k in range(max_others + 1):
            m = min(k, n - 1)
            d = len(parts["states"][m])
            g = sp.kron(identity, parts["internal"][m]) + sp.kron(tag_s, sp.eye(d))
            down = parts["down"][m].copy()
            if action == "remove":
                g -= delta * sp.eye(dimensions[k])
                if mode == "rcs" and m:
                    down += delta / (m + 1) * parts["removal"][m]
            elif mode == "rcs":
                g += delta / (m + 1) * (sp.kron(identity, parts["single"][m]) + sp.kron(tag_r - identity, sp.eye(d)))
            else:
                g += delta * (sp.kron(tag_r, parts["common"][m]) - sp.eye(dimensions[k]))
            if k and m:
                if k > n - 1:
                    down = down @ parts["admission"][m - 1]
                blocks[k][k - 1] = sp.kron(identity, down)
            if k < max_others:
                up = parts["admission"][m] if k < n - 1 else sp.eye(d)
                blocks[k][k + 1] = lam * sp.kron(identity, up)
                g -= lam * sp.eye(dimensions[k])
            blocks[k][k] = g
        q = sp.bmat(blocks, format="csr")
        maximum_size = max(maximum_size, q.shape[0])
        solve = factor(-q)
        values = np.ones(q.shape[0], dtype=q.dtype)
        rate = delta / n if mode == "rcs" else delta
        saturated = tag_s.toarray() - rate * np.eye(len(group))
        if action != "remove":
            saturated += rate * tag_r.toarray()
        baseline = np.ones(len(group), dtype=q.dtype)
        for r in range(1, order + 1):
            baseline = np.linalg.solve(-saturated, r * baseline)
            initial = np.concatenate([np.repeat(baseline, dimension // len(group)) for dimension in dimensions])
            values = solve(r * values, initial)
            for k, value in enumerate(np.split(values, np.cumsum(dimensions)[:-1])):
                contribution = service.alpha[group] @ value.reshape(len(group), -1)
                result[k][r] = contribution if result[k][r] is None else result[k][r] + contribution
    return result, maximum_size


def calculate(service, n=3, lam=1, delta=0.3, mode="rcs", action="restart", levels=120, behind=40, order=2):
    started = time.perf_counter()
    if (
        n < 1
        or order < 2
        or levels <= n
        or behind < 0
        or action not in ("restart", "fixed", "remove")
        or mode not in ("rcs", "disaster")
    ):
        raise ValueError("Invalid model or truncation parameters")
    rate = delta / n if mode == "rcs" else delta
    saturated = service.generator.copy()
    if action == "remove":
        saturated -= rate * np.eye(service.size)
    else:
        saturated += rate * (service.reset - np.eye(service.size))
    effective_mean = service.alpha @ np.linalg.solve(-saturated, np.ones(service.size))
    if not (action == "remove" and mode == "disaster" and delta > 0):
        if abs(effective_mean.imag) > 1e-8 or effective_mean.real <= 0 or lam * effective_mean.real >= n:
            raise ValueError("Saturated phase model is not stable")
    parts = occurrence_blocks(service, n)
    t_blocks = time.perf_counter()
    pi, (inside, down, replacement), diagnostics = stationary(parts, n, lam, delta, mode, action, levels)
    t_stationary = time.perf_counter()
    phase_size = len(parts["states"][n])
    solve_wait = factor(-inside[n], direct=True)
    solve_arrival = factor(lam * sp.eye(phase_size) - inside[n], direct=True)
    # W is absorbed on service admission, or on a clearing catastrophe.
    w = np.zeros(order, dtype=service.generator.dtype)
    previous = np.zeros((order + 1, phase_size), dtype=w.dtype)
    previous[0] = 1
    for ahead in range(levels - n):
        current = np.ones_like(previous)
        for r in range(1, order + 1):
            current[r] = solve_wait(r * current[r - 1] + replacement @ previous[r])
            w[r - 1] += pi[n + ahead] @ current[r]
        previous = current
    t_wait = time.perf_counter()
    tagged, maximum_tag_size = tag_service(service, parts, n, lam, delta, mode, action, n - 1 + behind, order)
    t_tag = time.perf_counter()
    v = np.zeros_like(w)
    for level in range(n):
        for r in range(1, order + 1):
            v[r - 1] += pi[level] @ tagged[level][r]
    start = np.ones((behind + 1, order + 1, phase_size), dtype=w.dtype)
    for b in range(behind + 1):
        for r in range(1, order + 1):
            start[b, r] = down[n] @ tagged[n - 1 + b][r]
    previous = None
    for ahead in range(levels - n):
        current = np.ones((behind + 1, order + 1, phase_size), dtype=w.dtype)
        for b in range(behind, -1, -1):
            for r in range(1, order + 1):
                rhs = r * current[b, r - 1]
                rhs += start[b, r] if ahead == 0 else replacement @ previous[b, r]
                if b < behind:
                    rhs += lam * current[b + 1, r]
                current[b, r] = (solve_arrival if b < behind else solve_wait)(rhs)
        for r in range(1, order + 1):
            v[r - 1] += pi[n + ahead] @ current[0, r]
        previous = current
    p = np.array([a.sum() for a in pi])
    imaginary = max(np.max(np.abs(a.imag)) for a in [p, w, v])
    if imaginary > 1e-7 or p.real.min() < -1e-9 or not np.isfinite(np.r_[p, w, v]).all():
        raise FloatingPointError("Invalid phase-model result")
    if abs(p.sum() - 1) > 1e-8 or any((a[1] - a[0] ** 2).real < -1e-10 for a in [w, v]):
        raise FloatingPointError("Invalid normalization or variance")
    little = [np.maximum(np.arange(levels) - n, 0) @ p / lam, np.arange(levels) @ p / lam]
    success = (
        sum(prob @ (parts["down"][min(i, n)] @ np.ones(parts["down"][min(i, n)].shape[1])) for i, prob in enumerate(pi))
        / lam
    )
    return dict(
        w=w.real.tolist(),
        v=v.real.tolist(),
        sw=float(np.sqrt((w[1] - w[0] ** 2).real)),
        sv=float(np.sqrt((v[1] - v[0] ** 2).real)),
        p=p.real.tolist(),
        q=float(success.real) if action == "remove" else 1.0,
        tail=float(p[-1].real),
        little_residuals=[float(abs(w[0] - little[0])), float(abs(v[0] - little[1]))],
        phases=service.size,
        phase_states=phase_size,
        stationary_states=sum(len(a) for a in pi),
        maximum_tag_block_states=maximum_tag_size,
        levels=levels,
        behind=behind,
        model=service.name,
        n=n,
        lam=lam,
        delta=delta,
        mode=mode,
        action=action,
        seconds=time.perf_counter() - started,
        timings=dict(
            blocks=t_blocks - started,
            stationary=t_stationary - t_blocks,
            waiting=t_wait - t_stationary,
            tag_service=t_tag - t_wait,
            tag_waiting=time.perf_counter() - t_tag,
        ),
        **diagnostics,
    )
