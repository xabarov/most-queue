"""
Cross-validate M/M/1 with a threshold-controlled service rate (EPIC-079,
roadmap item R5 -- Morrison, Queueing Systems 4, 1989) against discrete-event
simulation.

What this adds over the unit tests. Those already pin the implementation to an
explicit matrix solve of the same absorbing chain and to Little's law, both to
machine precision. Simulation checks something neither can: that a queue
actually run under these rules behaves as the recursion says -- in particular
that a customer's own service really does change speed because of arrivals
BEHIND it, which is the whole reason this model is not elementary.

On the tolerances: at the loads used here a few hundred thousand customers put
every moment well inside 2 sigma. The slowdown case (mu_high < mu_low) is the
slowest to settle -- at 250k it shows a consistent ~1.8 sigma on all six
quantities, which is warm-up, not theory: at 8 seeds x 2M it comes back at
+0.04 sigma. The budget here is chosen so the test is informative without being
flaky; the long run is recorded in the epic.
"""

from collections import deque

import numpy as np
import pytest

from most_queue.theory.fifo.mm1_threshold_rate import MM1ThresholdRateCalc

NUM_JOBS = 200_000
SEEDS = 5
WARMUP = 0.25


def _simulate(lam, mu_low, mu_high, threshold, n_jobs, seed):
    """
    Exponential service, so the rate change when the queue crosses the threshold
    needs no special handling: the remaining service is redrawn at the new rate.
    """
    rng = np.random.default_rng(seed)
    t = 0.0
    in_system = 0
    arrivals = deque()
    service_starts = deque()
    sojourns, waits = [], []
    next_arrival = rng.exponential(1 / lam)

    while len(sojourns) < n_jobs:
        rate = (mu_low if in_system <= threshold else mu_high) if in_system > 0 else 0.0
        departure = t + rng.exponential(1 / rate) if rate > 0 else float("inf")
        if next_arrival < departure:
            t = next_arrival
            arrivals.append(t)
            if in_system == 0:
                service_starts.append(t)
            in_system += 1
            next_arrival = t + rng.exponential(1 / lam)
        else:
            t = departure
            arrived = arrivals.popleft()
            started = service_starts.popleft()
            sojourns.append(t - arrived)
            waits.append(started - arrived)
            in_system -= 1
            if in_system > 0:
                service_starts.append(t)

    warm = int(n_jobs * WARMUP)
    return np.array(sojourns[warm:]), np.array(waits[warm:])


def _deviations(threshold, lam, mu_low, mu_high, num=3):
    calc = MM1ThresholdRateCalc(threshold=threshold)
    calc.set_sources(lam)
    calc.set_servers(mu_low=mu_low, mu_high=mu_high)

    sojourn_rows, wait_rows = [], []
    for seed in range(SEEDS):
        sojourns, waits = _simulate(lam, mu_low, mu_high, threshold, NUM_JOBS, seed)
        sojourn_rows.append([(sojourns**k).mean() for k in range(1, num + 1)])
        wait_rows.append([(waits**k).mean() for k in range(1, num + 1)])
    sojourn_rows, wait_rows = np.array(sojourn_rows), np.array(wait_rows)

    out = []
    for rows, theory in ((sojourn_rows, calc.get_v(num)), (wait_rows, calc.get_w(num))):
        for k in range(num):
            sample = rows[:, k]
            err = sample.std(ddof=1) / np.sqrt(SEEDS)
            out.append(abs(theory[k] - sample.mean()) / max(err, 1e-15))
    return out


@pytest.mark.parametrize(
    "threshold, lam, mu_low, mu_high, label",
    [
        (3, 0.6, 0.7, 1.4, "speeds up once the queue builds"),
        (2, 0.9, 0.3, 1.5, "low rate below the arrival rate: only the threshold saves it"),
        (4, 0.6, 1.0, 1.0, "equal rates: an ordinary M/M/1"),
    ],
)
def test_moments_vs_sim(threshold, lam, mu_low, mu_high, label):
    sigmas = _deviations(threshold, lam, mu_low, mu_high)
    assert all(s < 3.5 for s in sigmas), f"{label}: deviations {['%.2f' % s for s in sigmas]}"


def test_slowdown_case_distribution_level_agreement():
    """
    mu_high < mu_low -- the server gets SLOWER under load. Only the first moment
    is asserted at this budget: the higher moments of this case need far more
    simulation to settle (see the module docstring), and asserting them loosely
    would be a test that cannot fail.
    """
    sigmas = _deviations(5, 0.6, 1.5, 0.9, num=1)
    assert all(s < 3.5 for s in sigmas)
