"""Block-Jacobi preconditioning of the unchanged embedded-chain equations."""

import numpy as np
from scipy import sparse as sp
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, spilu, bicgstab, lgmres


def solve(low_rates, high_rates, inside, down, up, replacement, n, lam, tolerance, progress):
    sizes = [len(r) for r in low_rates]
    offsets = np.r_[0, np.cumsum(sizes)]
    total = int(offsets[-1] + high_rates.size)
    calls = 0
    factors = [spilu((sp.diags(low_rates[m]) - inside[m]).T.tocsc(), drop_tol=1e-4, fill_factor=3) for m in range(n)]
    regular = spilu((sp.diags(high_rates[0]) - inside[n]).T.tocsc(), drop_tol=1e-4, fill_factor=3)
    boundary = spilu((sp.diags(high_rates[-1]) - inside[n]).T.tocsc(), drop_tol=1e-4, fill_factor=3)
    if progress:
        print("Incomplete block factors prepared", flush=True)

    def jump(vector):
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

    # Coarse levels remove the slow queue-length modes left by block Jacobi.
    conditional = []
    for m in range(n + 1):
        if m == 0:
            conditional.append(np.ones(1))
            continue
        rates = low_rates[m] if m < n else high_rates[0]
        replace = down[m] @ (up[m - 1] / lam)
        closed = (inside[m] + replace).T.tocsr()
        mass = np.ones(len(rates)) / len(rates)
        for iteration in range(1000):
            new = 0.5 * (mass + closed @ (mass / rates) + lam * mass / rates)
            new /= new.sum()
            error = np.abs(new - mass).sum()
            mass = new
            if iteration > 50 and error < 1e-8:
                break
        time_mass = mass / rates
        time_mass /= time_mass.sum()
        conditional.append(time_mass)
    all_rates = low_rates + list(high_rates)
    basis = [conditional[min(i, n)] * r for i, r in enumerate(all_rates)]
    basis = [a / a.sum() for a in basis]
    coarse = np.zeros((len(basis), len(basis)))
    for i, z in enumerate(basis):
        m = min(i, n)
        occupancy = z / all_rates[i]
        coarse[i, i] = 1 - float(np.asarray(inside[m].sum(axis=1)).ravel() @ occupancy)
        if i:
            coarse[i - 1, i] = -float(np.asarray((down[m] if i <= n else replacement).sum(axis=1)).ravel() @ occupancy)
        if i + 1 < len(basis):
            coarse[i + 1, i] = -lam * occupancy.sum()
        coarse[0, i] += 1
    coarse_factor = lu_factor(coarse)
    if progress:
        print("Coarse queue correction prepared", flush=True)

    def matvec(vector):
        nonlocal calls
        calls += 1
        if progress and calls % 100 == 0:
            print("Preconditioned products:", calls, flush=True)
        r = vector - jump(vector)
        r[0] += vector.sum()
        return r

    def precondition(vector):
        result = np.empty_like(vector)
        for i in range(n):
            result[offsets[i] : offsets[i + 1]] = low_rates[i] * factors[i].solve(vector[offsets[i] : offsets[i + 1]])
        hi = vector[offsets[-1] :].reshape(high_rates.shape)
        out = result[offsets[-1] :].reshape(high_rates.shape)
        # A separate sparse triangular solve per level avoids very large dense RHS temporaries.
        for i in range(len(hi) - 1):
            out[i] = high_rates[i] * regular.solve(hi[i])
        out[-1] = high_rates[-1] * boundary.solve(hi[-1])
        residual = vector - matvec(result)
        compressed = np.r_[
            [residual[offsets[i] : offsets[i + 1]].sum() for i in range(n)],
            residual[offsets[-1] :].reshape(high_rates.shape).sum(axis=1),
        ]
        correction = lu_solve(coarse_factor, compressed)
        for i in range(n):
            result[offsets[i] : offsets[i + 1]] += correction[i] * basis[i]
        out += np.asarray(basis[n:]) * correction[n:, None]
        return result

    a = LinearOperator((total, total), matvec=matvec, dtype=float)
    m = LinearOperator(a.shape, matvec=precondition, dtype=float)
    rhs = np.zeros(total)
    rhs[0] = 1
    x, info = bicgstab(a, rhs, M=m, rtol=min(tolerance * 0.01, 1e-12), atol=1e-14, maxiter=1500)
    if info:
        x, info = lgmres(a, rhs, x0=x, M=m, rtol=min(tolerance * 0.01, 1e-12), atol=1e-14, maxiter=150, inner_m=30)
    if info or not np.isfinite(x).all():
        raise RuntimeError(f"Preconditioned stationarity failed: {info}")
    x /= x.sum()
    residual = np.abs(x - jump(x)).sum()
    if x.min() < -1e-10 or residual > max(1e-8, 100 * tolerance):
        raise FloatingPointError((x.min(), residual))
    return [x[offsets[i] : offsets[i + 1]] for i in range(n)], x[offsets[-1] :].reshape(high_rates.shape), calls
