"""Independent reset-queue experiments and direct phase-state equations."""

import argparse
import concurrent.futures
import json
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import t

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv

ROOT = Path(__file__).resolve().parent


def phase_reference(cv, mode, delta=0.3, n=3, levels=100):
    """Stationarity for resampling: reset phases, without removing a customer."""
    b = gamma_moments_by_mean_and_cv(0.7 * n, cv)
    h = H2Distribution.get_params_clx(b)
    y, mu = [h.p1, 1 - h.p1], [h.mu1, h.mu2]
    states = [(i, j) for i in range(levels) for j in range(min(i, n) + 1)]
    indices = {s: k for k, s in enumerate(states)}
    q = lil_matrix((len(states), len(states)), dtype=complex)
    for row, (i, j) in enumerate(states):
        m = min(i, n)

        def add(target, rate):
            if rate and target != (i, j):
                q[row, indices[target]] += rate
                q[row, row] -= rate

        if i < levels - 1:
            if i < n:
                add((i + 1, j), y[0])
                add((i + 1, j + 1), y[1])
            else:
                add((i + 1, j), 1)
        if not m:
            continue
        for phase, count in [(0, m - j), (1, j)]:
            if not count:
                continue
            left = j - phase
            if i <= n:
                add((i - 1, left), count * mu[phase])
            else:
                for new_phase in [0, 1]:
                    add((i - 1, left + new_phase), count * mu[phase] * y[new_phase])
        if mode == "rcs":
            if j < m:
                add((i, j + 1), delta * (m - j) / m * y[1])
            if j:
                add((i, j - 1), delta * j / m * y[0])
        else:
            for k in range(m + 1):
                add((i, k), delta * math.comb(m, k) * y[1] ** k * y[0] ** (m - k))
    a = q.T.tolil()
    a[-1, :] = 1
    rhs = np.zeros(len(states), complex)
    rhs[-1] = 1
    pi = spsolve(a.tocsc(), rhs)
    p = np.zeros(levels, complex)
    for prob, (i, j) in zip(pi, states):
        p[i] += prob
    # While the tag waits, arrivals join behind it; all servers stay busy.
    # Factorial queue-length moments give the waiting moments under Poisson
    # arrivals and FCFS admission to service, including dependent services.
    w = [
        sum(
            math.prod(range(max(i - n, 0) - k + 1, max(i - n, 0) + 1)) * pk for i, pk in enumerate(p) if i - n >= k
        ).real
        for k in [1, 2]
    ]
    v1 = sum(i * pk for i, pk in enumerate(p)).real
    return dict(
        cv=cv,
        mode=mode,
        levels=levels,
        delta=delta,
        w=w,
        v1=v1,
        sw=float(np.sqrt(w[1] - w[0] ** 2)),
        residual=float(np.max(np.abs(pi @ q.tocsr()))),
        tail=float(p[-1].real),
        min_p=float(p.real.min()),
        imaginary_p=float(np.max(np.abs(p.imag))),
    )


def independent_simulation(args):
    """No production simulator code: arrivals, completions and restart clocks."""
    cv, mode, scenario, independent, seed, sample = args
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    fixed = scenario == "fixed"
    rate = 0.02 if fixed else 0.3
    n, warm = 3, 50000
    h = H2Distribution.get_params(gamma_moments_by_mean_and_cv(2.1, cv)) if cv > 1 else None

    def service():
        if h:
            return rng.exponential(1 / (h.mu1 if rng.random() < h.p1 else h.mu2))
        return rng.gamma(1 / cv**2, 2.1 * cv**2)

    def effective():
        duration, elapsed = service(), 0.0
        while True:
            restart = rng.exponential(1 / rate)
            if restart >= duration:
                return elapsed + duration
            elapsed += restart
            if not fixed:
                duration = service()

    completions = [math.inf] * n
    jobs = [None] * n  # arrival, first start, original requirement
    queue = deque()
    now, arrived, departed = 0.0, 0, 0
    arrival, reset = rng.exponential(), (math.inf if independent else rng.exponential(1 / rate))
    sums = np.zeros(7)  # W,W²,V,V²,S,S²,WS
    occup = np.zeros(2)
    sample_time = 0.0

    def begin(slot, arr, clock):
        b = effective() if independent else service()
        jobs[slot] = [arr, clock, b]
        completions[slot] = clock + b

    while departed < warm + sample:
        slot = min(range(n), key=completions.__getitem__)
        clock = min(arrival, reset, completions[slot])
        if departed >= warm:
            dt = clock - now
            busy = sum(j is not None for j in jobs)
            occup += [dt * len(queue), dt * (len(queue) + busy)]
            sample_time += dt
        now = clock
        if clock == arrival:
            arrived += 1
            arrival += rng.exponential()
            free = next((k for k, j in enumerate(jobs) if j is None), None)
            if free is None:
                queue.append(now)
            else:
                begin(free, now, now)
        elif clock == reset:
            reset += rng.exponential(1 / rate)
            busy = [k for k, j in enumerate(jobs) if j is not None]
            selected = [busy[int(rng.integers(len(busy)))]] if mode == "rcs" and busy else busy
            for k in selected:
                completions[k] = now + (jobs[k][2] if fixed else service())
        else:
            arr, start, _ = jobs[slot]
            w, v, s = start - arr, now - arr, now - start
            if departed >= warm:
                sums += [w, w * w, v, v * v, s, s * s, w * s]
            departed += 1
            jobs[slot] = None
            completions[slot] = math.inf
            if queue:
                begin(slot, queue.popleft(), now)
    means = sums / sample
    return dict(
        cv=cv,
        mode=mode,
        scenario=scenario,
        independent=independent,
        seed=seed,
        warmup=warm,
        sample=sample,
        w=means[0],
        v=means[2],
        sw=float(np.sqrt(means[1] - means[0] ** 2)),
        sv=float(np.sqrt(means[3] - means[2] ** 2)),
        service_mean=means[4],
        service_second=means[5],
        covariance_ws=means[6] - means[0] * means[4],
        little_w=occup[0] / sample_time,
        little_v=occup[1] / sample_time,
        seconds=time.perf_counter() - started,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=500000)
    args = parser.parse_args()
    cases = [
        (cv, "disaster", scenario, independent)
        for cv in [0.8, 1.2]
        for scenario in ["fixed", "restart"]
        for independent in [False, True]
    ]
    cases += [(cv, "rcs", "restart", False) for cv in [0.8, 1.2]]
    path = ROOT / "independent-replications.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {r["seed"] for r in rows}
    tasks = [(*case, 202611071000 + 100 * i + j, args.sample) for i, case in enumerate(cases) for j in range(8)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(independent_simulation, task) for task in tasks if task[-2] not in done]
        for f in concurrent.futures.as_completed(pending):
            row = f.result()
            rows.append(row)
            with path.open("a") as out:
                out.write(json.dumps(row) + "\n")
            print(
                f"{len(rows)}/{len(tasks)} {row['cv']} {row['mode']} {row['scenario']} iid={row['independent']}",
                flush=True,
            )
    results = []
    keys = ["w", "v", "sw", "sv", "service_mean", "service_second", "covariance_ws", "little_w", "little_v"]
    for case in cases:
        group = [r for r in rows if (r["cv"], r["mode"], r["scenario"], r["independent"]) == case]
        summary = {}
        for key in keys:
            values = np.array([r[key] for r in group])
            summary[key] = dict(
                mean=float(values.mean()),
                half_ci95=float(t.ppf(0.975, len(group) - 1) * values.std(ddof=1) / len(group) ** 0.5),
            )
        results.append(dict(zip(["cv", "mode", "scenario", "independent"], case)) | {"estimates": summary})
    (ROOT / "independent-results.json").write_text(json.dumps(results, indent=2))
    references = [
        phase_reference(cv, mode, levels=levels)
        for cv in [0.8, 1.2]
        for mode in ["rcs", "disaster"]
        for levels in [100, 160]
    ]
    (ROOT / "phase-reference.json").write_text(json.dumps(references, indent=2))


if __name__ == "__main__":
    main()
