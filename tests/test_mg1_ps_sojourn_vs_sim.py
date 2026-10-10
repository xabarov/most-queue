"""
Cross-validate the higher conditional sojourn-time moments in M/G/1-PS
(EPIC-077, roadmap item R3 -- Yashkov, arXiv:math/0512281) against simulation.

How the comparison is made matters here. Binning simulated jobs by size and
comparing with ``v_n(bin centre)`` is BIASED, because ``E[size | size in bin]``
is not the bin centre -- and the bias is large enough to show up as many sigma.
It is immediately visible on the first moment, which is the exact insensitive
``x/(1-rho)``, so it cannot be mistaken for a theory error if one looks.

Instead every simulated job is paired with the theory evaluated at ITS OWN
size: the sample mean of ``V^n`` against the sample mean of ``v_n(size_j)`` over
the same jobs. The size-sampling error then cancels, and what is left tests
``v_n(.)`` pointwise across the whole size range.

The first moment doubles as a built-in control: it is exact and insensitive, so
whatever deviation it shows is the simulator's and calibrates how much of the
deviation in the higher moments to attribute to the same cause.
"""

import numpy as np
import pytest

from most_queue.random.utils.params import ErlangParams, H2Params
from most_queue.theory.fifo.mg1_ps import MG1PSCalc

NUM_JOBS = 120_000
SEEDS = 4
WARMUP = 0.25


def _simulate(lam, sample_size, n_jobs, seed):
    """
    Egalitarian PS, written from scratch so that each job's (size, sojourn) pair
    is recorded -- the library's ProcessorSharingSim only accumulates moments.
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    jobs = []  # [remaining work, arrival time, size]
    sizes, sojourns = [], []
    next_arrival = rng.exponential(1 / lam)
    while len(sizes) < n_jobs:
        count = len(jobs)
        if count == 0:
            t = next_arrival
            size = sample_size(rng)
            jobs.append([size, t, size])
            next_arrival = t + rng.exponential(1 / lam)
            continue
        min_rem = min(job[0] for job in jobs)
        completion = t + min_rem * count
        if next_arrival < completion:
            step = (next_arrival - t) / count
            for job in jobs:
                job[0] -= step
            t = next_arrival
            size = sample_size(rng)
            jobs.append([size, t, size])
            next_arrival = t + rng.exponential(1 / lam)
        else:
            for job in jobs:
                job[0] -= min_rem
            t = completion
            idx = min(range(count), key=lambda i: jobs[i][0])
            _, arrival, size = jobs.pop(idx)
            sizes.append(size)
            sojourns.append(t - arrival)
    warm = int(n_jobs * WARMUP)
    return np.array(sizes[warm:]), np.array(sojourns[warm:])


def _paired_deviation(lam, params, sampler, num=3):
    """Sigma of ``mean(V^n) - mean(v_n(size))`` from zero, per moment."""
    calc = MG1PSCalc()
    calc.set_sources(lam)
    calc.set_servers_params(params)

    diffs = [[] for _ in range(num)]
    for seed in range(SEEDS):
        sizes, sojourns = _simulate(lam, sampler, NUM_JOBS, seed)
        grid = np.linspace(sizes.min(), float(np.quantile(sizes, 0.9995)), 300)
        table = np.array([calc.get_conditional_sojourn_moments(float(u), num) for u in grid])
        for n in range(num):
            predicted = np.interp(sizes, grid, table[:, n])
            diffs[n].append(float((sojourns ** (n + 1)).mean() - predicted.mean()))
    out = []
    for n in range(num):
        arr = np.array(diffs[n])
        out.append(arr.mean() / (arr.std(ddof=1) / np.sqrt(SEEDS)))
    return out


@pytest.mark.parametrize(
    "lam, params, sampler, label",
    [
        (0.5, H2Params(mu1=1.0, mu2=9.0, p1=1.0), lambda r: r.exponential(1.0), "M/M/1-PS"),
        (0.3, ErlangParams(r=3, mu=3.0), lambda r: r.gamma(3, 1 / 3.0), "M/E3/1-PS"),
        (
            0.5,
            H2Params(mu1=0.5, mu2=3.0, p1=0.4),
            lambda r: r.exponential(1 / 0.5) if r.random() < 0.4 else r.exponential(1 / 3.0),
            "M/H2/1-PS",
        ),
    ],
)
def test_conditional_moments_vs_sim(lam, params, sampler, label):
    """
    Moderate loads on purpose: at rho close to 1 a PS simulation that starts
    empty needs far more than a test budget to settle, and the deviation it then
    shows is its own (the first moment, which is exact, drifts by the same
    amount). Those loads are covered in the epic with a much larger run.
    """
    sigmas = _paired_deviation(lam, params, sampler)
    assert all(abs(s) < 3.0 for s in sigmas), f"{label}: paired deviations {sigmas}"


def test_first_moment_control_is_tight():
    """
    The exact insensitive first moment must agree closely -- if it does not, the
    simulation budget is the problem and the higher-moment comparison above
    carries no information.
    """
    sigmas = _paired_deviation(0.5, ErlangParams(r=2, mu=2.0), lambda r: r.gamma(2, 1 / 2.0), num=1)
    assert abs(sigmas[0]) < 3.0
