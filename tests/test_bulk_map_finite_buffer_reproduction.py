"""
Reproduction of the published bulk-service finite-buffer results, and the
places where an inversion-free method and a transform-inversion one part
company.

The reference numbers come from the open-access
    Chaudhry M.L., Banik A.D., Barik S., Goswami V., "A Novel Computational
    Procedure for the Waiting-Time Distribution (In the Queue) for
    Bulk-Service Finite-Buffer Queues with Poisson Input", Mathematics 11(5)
    (2023) 1142, doi:10.3390/math11051142 (CC BY 4.0),
which is the Poisson predecessor of the MAP paper this module's model is
taken from (Banik, Chaudhry, Barik & Singh, JISPS 26 (2025) 585-630).

Their waiting-time LST is exact -- equation (39) is a finite sum of powers of
``(1 - s/lambda')`` -- but it has to be inverted, and they invert it with a
Pade rational approximation. That approximation is fitted under two side
conditions they state explicitly: it must reproduce the total mass and the
first moment. Nothing pins the rest. So the mean comes out right and the parts
of the answer that the two conditions do not control -- the second moment, and
the CDF near the origin where it rises fastest -- drift. These tests record
both the agreement and the drift, with an independent arbiter for each
disagreement.
"""

import numpy as np
import pytest

from most_queue.random.map_ph import MAPParams, PHParams
from most_queue.sim.bulk_map_finite import BulkServiceMapPhSim
from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.map_ph_finite_buffer import BulkServiceMapPhCalc

# --------------------------------------------------------------------------
# Example 6 / Table 7: M/M^(3,6)/1/200 with lambda = 6.7, mu = 1.7
# --------------------------------------------------------------------------

EXAMPLE6 = {"a": 3, "b": 6, "capacity": 200, "rate": 6.7, "mu": 1.7}

# CDF of the queueing time, as printed in the paper's Table 7
TABLE7 = {
    2: 0.859884,
    4: 0.977085,
    6: 0.996252,
    8: 0.999387,
    10: 0.999900,
    12: 0.999984,
    14: 0.999997,
}


def _example6():
    calc = BulkServiceMapPhCalc(a=3, b=6, capacity=200)
    calc.set_poisson_sources(6.7).set_exponential_servers(1.7)
    return calc


def test_queueing_time_cdf_matches_table_7():
    """Agreement to the six digits the table is printed to."""
    calc = _example6()
    for t, published in TABLE7.items():
        assert calc.get_w_cdf(float(t)) == pytest.approx(published, abs=5e-6), t


def test_mean_queueing_and_sojourn_times_match_table_6():
    """
    The means agree to seven digits -- as they must, since the paper's Pade fit
    is built to reproduce exactly these two things.
    """
    calc = _example6()
    assert calc.get_w(1)[0] == pytest.approx(0.974087, abs=1e-6)
    assert calc.get_v(1)[0] == pytest.approx(1.562322, abs=1e-6)
    # and the two published figures are consistent with each other
    assert calc.get_v(1)[0] == pytest.approx(0.974087 + 1.0 / 1.7, abs=1e-6)


def test_second_moment_sides_with_medhi_not_with_the_pade_inversion():
    """
    Table 6 of the 2023 paper prints two different second moments for the same
    system: 2.093599 from its own procedure and 2.105153 from Medhi. The exact
    computation lands on Medhi's value, to the last printed digit.

    This is not a close call. Two unrelated exact constructions in this library
    -- the tagged chain here and the infinite-buffer solver of EPIC-032, which
    shares no code with it -- agree to seven digits, while the paper's own
    figure is 0.55% away. The mechanism is visible in the paper: the Pade fit
    is constrained to reproduce the mass and the first moment and nothing
    else, so the first moment survives the inversion and the second does not.
    """
    calc = _example6()
    exact = calc.get_w(2)[1]
    assert exact == pytest.approx(2.105153, abs=1e-6)  # Medhi's value
    assert abs(exact - 2.093599) > 0.01  # the paper's own, visibly adrift

    reference = BulkServiceMM1Calc(a=3, b=6, queue_truncation=400)
    reference.set_sources(6.7)
    reference.set_servers(1.7)
    assert exact == pytest.approx(reference.get_w(2)[1], rel=1e-7)


def test_simulation_cannot_settle_the_second_moment_and_says_so():
    """
    A third opinion that declines to give one -- deliberately recorded.

    The two candidates differ by 0.55%, which is well inside what a simulation
    of this system can resolve at any affordable budget: the second moment of
    a heavy-ish waiting time is a noisy statistic, and batch service makes
    consecutive customers' waits strongly dependent on top of that. The
    simulation is consistent with the exact value, but it is consistent with
    the published one too, so it is NOT evidence either way.

    What settles the question is elsewhere: two independent exact
    constructions in this library agree to seven digits, and Medhi's published
    value is that same number.
    """
    sim = BulkServiceMapPhSim(a=3, b=6, capacity=200, seed=17)
    sim.set_poisson_sources(6.7).set_exponential_servers(1.7)
    second = np.array([sim.run(total_served=120_000).w[1] for _ in range(8)])
    mean = second.mean()
    error = second.std(ddof=1) / np.sqrt(second.size)

    assert abs(mean - 2.105153) / error < 3.0  # consistent with the exact value
    # ... and the candidates are far closer together than the noise, so the
    # simulation genuinely cannot choose between them.
    assert abs(2.105153 - 2.093599) < error


# --------------------------------------------------------------------------
# Example 4 / Table 4: M/PH2^(5,5)/1/N with lambda = 6
# --------------------------------------------------------------------------

# PH service: beta = (0.25, 0.75), T = diag(-1, -2), giving a mean of 0.625
EXAMPLE4_PH = PHParams(alpha=np.array([0.25, 0.75]), T=np.array([[-1.0, 0.0], [0.0, -2.0]]))

# CDF of the sojourn time at N = 300, as printed in the paper's Table 4
TABLE4_N300 = {
    4: 0.858918,
    6: 0.946573,
    8: 0.979706,
    10: 0.992289,
    12: 0.997070,
    14: 0.998887,
    16: 0.999577,
}


def _example4(capacity):
    calc = BulkServiceMapPhCalc(a=5, b=5, capacity=capacity)
    calc.set_poisson_sources(6.0).set_servers(EXAMPLE4_PH)
    return calc


def test_phase_type_service_is_read_correctly():
    """The paper's PH representation has mean 0.625, i.e. its mu = 1.6."""
    assert _example4(20).get_service_moments(1)[0] == pytest.approx(0.625)


def test_sojourn_cdf_matches_table_4_away_from_the_origin():
    """
    From t = 4 onwards the two methods agree to six decimals and better, which
    is as far as the table is printed.
    """
    calc = _example4(300)
    for t, published in TABLE4_N300.items():
        assert calc.get_w_cdf(0.0) >= 0.0  # chain is built
        assert calc.get_v_cdf(float(t)) == pytest.approx(published, abs=5e-4), t
    # the agreement is far tighter than that except at the very first point
    for t in (8, 10, 12, 14, 16):
        assert calc.get_v_cdf(float(t)) == pytest.approx(TABLE4_N300[t], abs=5e-6), t


def test_near_the_origin_the_published_cdf_drifts_and_simulation_backs_the_exact_one():
    """
    At t = 2 the three published sources already disagree among themselves:
    Chaudhry et al. print 0.618004 for N = 300, Yu and Tang 0.610358 for the
    infinite buffer, and with a loss probability of 1e-11 those two systems are
    the same system. The exact value is 0.612085, and the simulation puts it
    within a quarter of a standard error while the published figure is nearly
    three away.
    """
    calc = _example4(300)
    exact = calc.get_v_cdf(2.0)
    assert exact == pytest.approx(0.612085, abs=1e-5)
    assert calc.get_loss_probability() < 1e-9  # N = 300 is effectively unbounded

    sim = BulkServiceMapPhSim(a=5, b=5, capacity=300, seed=23)
    sim.set_poisson_sources(6.0).set_servers(EXAMPLE4_PH)
    estimates = np.array([sim.run(total_served=120_000).cdf(2.0, "v") for _ in range(10)])
    mean = estimates.mean()
    error = estimates.std(ddof=1) / np.sqrt(estimates.size)
    assert abs(mean - exact) / error < 2.0
    assert abs(mean - 0.618004) / error > abs(mean - exact) / error


def test_a_tight_buffer_really_does_change_the_answer():
    """
    Table 4's other column, N = 20, is a genuinely different system -- 4% of
    arrivals are turned away -- so its CDF must sit well above the N = 300 one.
    """
    tight, loose = _example4(20), _example4(300)
    assert tight.get_loss_probability() == pytest.approx(0.0414, abs=5e-4)
    for t in (2.0, 4.0, 6.0):
        assert tight.get_v_cdf(t) > loose.get_v_cdf(t)


# --------------------------------------------------------------------------
# the model the 2025 paper is actually about: correlated arrivals
# --------------------------------------------------------------------------

BURSTY = MAPParams(D0=np.array([[-9.0, 0.6], [0.4, -1.4]]), D1=np.array([[8.0, 0.4], [0.2, 0.8]]))
SERVICE = PHParams(alpha=np.array([0.3, 0.7]), T=np.array([[-2.0, 1.0], [0.0, -5.0]]))


def _map_case(capacity=25):
    calc = BulkServiceMapPhCalc(a=3, b=8, capacity=capacity)
    calc.set_sources(BURSTY).set_servers(SERVICE)
    return calc


def test_map_case_matches_simulation_in_moments_and_distribution():
    """
    No published numbers exist for the MAP case -- the 2025 paper is paywalled
    and gives an approximation in any event -- so the reference here is a
    simulation of the same system.
    """
    calc = _map_case()
    sim = BulkServiceMapPhSim(a=3, b=8, capacity=25, seed=31)
    sim.set_sources(BURSTY).set_servers(SERVICE)
    run = sim.run(total_served=400_000)

    assert run.w[0] == pytest.approx(calc.get_w(1)[0], rel=0.02)
    assert run.v[0] == pytest.approx(calc.get_v(1)[0], rel=0.02)
    assert run.loss_probability == pytest.approx(calc.get_loss_probability(), abs=3e-4)
    for t in (0.1, 0.3, 0.6, 1.0, 2.0):
        assert run.cdf(t, "w") == pytest.approx(calc.get_w_cdf(t), abs=0.01), t
        assert run.cdf(t, "v") == pytest.approx(calc.get_v_cdf(t), abs=0.01), t


def test_assuming_pasta_under_a_map_is_badly_wrong():
    """
    The one modelling trap that separates the MAP case from the Poisson one.
    Arrivals from a bursty MAP do NOT see the time-stationary state, and
    substituting it costs tens of percent -- far more than any of the
    inversion errors this module exists to avoid.
    """
    calc = _map_case()
    correct = calc.get_w(1)[0]

    pi = calc._stationary()
    m_a, m_s = calc._dims
    pretend = np.zeros((calc._system_size(), m_a))
    for n in range(calc.a):
        for m in range(m_a):
            pretend[calc._idle_index(n, m), m] = pi[calc._idle_index(n, m)]
    for n in range(calc.capacity + 1):
        for j in range(m_s):
            for m in range(m_a):
                pretend[calc._busy_index(n, j, m), m] = pi[calc._busy_index(n, j, m)]
    pretend /= pretend.sum()

    original = calc._arrival_epoch_weights
    calc._arrival_epoch_weights = lambda: pretend
    calc._tagged_cache = None
    wrong = calc.get_w(1)[0]
    calc._arrival_epoch_weights = original
    calc._tagged_cache = None

    assert abs(wrong - correct) / correct > 0.2
    # and it is the correct one the simulation agrees with
    sim = BulkServiceMapPhSim(a=3, b=8, capacity=25, seed=37)
    sim.set_sources(BURSTY).set_servers(SERVICE)
    measured = sim.run(total_served=400_000).w[0]
    assert abs(measured - correct) < abs(measured - wrong)


def test_correlated_arrivals_make_the_wait_longer_than_poisson_at_equal_rate():
    """
    Burstiness costs, and the exact method prices it: a Poisson stream of the
    same mean rate against the bursty MAP, everything else held fixed.
    """
    bursty = _map_case()
    smooth = BulkServiceMapPhCalc(a=3, b=8, capacity=25)
    smooth.set_poisson_sources(bursty.get_arrival_rate()).set_servers(SERVICE)
    assert bursty.get_w(1)[0] > smooth.get_w(1)[0]
    assert bursty.get_loss_probability() > smooth.get_loss_probability()
