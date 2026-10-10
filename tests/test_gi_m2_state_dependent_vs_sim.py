"""
Cross-validate GI/M/2 with a busy-server-dependent service rate (EPIC-078,
roadmap item R4 -- Bhat, AISM 18:211-221, 1966) against discrete-event
simulation.

Note on what this adds. The unit tests already pin the implementation to an
EXACT CTMC (Erlang interarrivals make the system a finite-phase Markov chain),
which is a sharper reference than any simulation. What simulation adds is an
end-to-end check of the modelling itself -- that a queue actually run under
these rules behaves the way the transforms say -- and in particular of the
waiting time, which the CTMC comparison does not cover.

One trap, learned the hard way and worth repeating in code: normalising the
collected state counts by the TRUNCATED totals inflates every probability by
``1 / P(N < truncation)``. At cv = 1.5 that was a few per cent, enough to look
like a 13-sigma theory error. Totals here count every arrival and all elapsed
time regardless of truncation.
"""

from collections import deque

import numpy as np
import pytest

from most_queue.random.distributions import GammaDistribution
from most_queue.theory.fifo.gi_m2_state_dependent import GiM2StateDependentCalc

ARRIVAL_RATE = 1.2
NUM_JOBS = 200_000
SEEDS = 4
P_NUM = 12


def _simulate(gamma_params, mu, mu_single, n_jobs, seed, warmup=0.2):
    """
    Two servers, renewal arrivals, exponential service whose rate depends on the
    occupancy. Memorylessness means the rate change when the second server
    becomes busy needs no special handling -- the remaining service is simply
    redrawn at the new rate.
    """
    rng = np.random.default_rng(seed)
    shape, scale = gamma_params.alpha, 1.0 / gamma_params.mu
    t = 0.0
    in_system = 0
    queued = deque()  # arrival times of customers not yet in service
    waits = []
    seen = np.zeros(P_NUM)
    area = np.zeros(P_NUM)
    arrivals = 0
    elapsed = 0.0
    collecting = False
    warm_target = int(n_jobs * warmup)
    next_arrival = rng.gamma(shape, scale)

    while len(waits) < n_jobs:
        rate = 0.0 if in_system == 0 else (mu_single if in_system == 1 else 2.0 * mu)
        departure = t + rng.exponential(1.0 / rate) if rate > 0 else float("inf")
        step = min(departure, next_arrival)
        if collecting:
            elapsed += step - t
            if in_system < P_NUM:
                area[in_system] += step - t
        t = step

        if departure < next_arrival:
            in_system -= 1
            if queued:  # a server just freed up for the head of the queue
                waits.append(t - queued.popleft())
        else:
            if collecting:
                arrivals += 1
                if in_system < P_NUM:
                    seen[in_system] += 1
            if in_system >= 2:
                queued.append(t)
            else:
                waits.append(0.0)
            in_system += 1
            next_arrival = t + rng.gamma(shape, scale)
            if not collecting and len(waits) >= warm_target:
                collecting = True

    return (
        np.array(waits[warm_target:]),
        seen / max(arrivals, 1),
        area / max(elapsed, 1e-12),
    )


def _deviations(cv, mu, mu_single):
    mean_a = 1.0 / ARRIVAL_RATE
    gamma_params = GammaDistribution.get_params_by_mean_and_cv(mean_a, cv)
    moments = GammaDistribution.calc_theory_moments(gamma_params, 4)
    calc = GiM2StateDependentCalc()
    calc.set_sources(moments)
    calc.set_servers(mu=mu, mu_single=mu_single)

    waits, pis, ps = [], [], []
    for seed in range(SEEDS):
        w, pi, p = _simulate(gamma_params, mu, mu_single, NUM_JOBS, seed)
        waits.append([w.mean(), (w > 0).mean()])
        pis.append(pi)
        ps.append(p)
    waits, pis, ps = np.array(waits), np.array(pis), np.array(ps)

    def sigma(sample, reference):
        err = sample.std(ddof=1) / np.sqrt(SEEDS)
        return abs(reference - sample.mean()) / max(err, 1e-15)

    out = [sigma(waits[:, 0], calc.get_w(1)[0]), sigma(waits[:, 1], calc.get_wait_prob())]
    out += [sigma(ps[:, j], calc.get_p(P_NUM)[j]) for j in range(4)]
    out += [sigma(pis[:, j], calc.get_pi(P_NUM)[j]) for j in range(4)]
    return out


@pytest.mark.parametrize(
    "cv, mu, mu_single, label",
    [
        (0.5, 1.0, 1.6, "acceleration: the idle server helps the busy one"),
        (1.5, 1.0, 0.6, "slowdown: a lone server works slower"),
        (0.8, 1.0, 1.0, "ordinary GI/M/2"),
    ],
)
def test_states_and_waiting_vs_sim(cv, mu, mu_single, label):
    sigmas = _deviations(cv, mu, mu_single)
    assert all(s < 4.0 for s in sigmas), f"{label}: deviations {['%.2f' % s for s in sigmas]}"
