"""
Unit tests for phase-type distributions and MAPs (most_queue.random.map_ph).
"""

import numpy as np

from most_queue.random.distributions import ErlangDistribution, ExpDistribution, H2Distribution
from most_queue.random.map_ph import MAP, PHDistribution
from most_queue.random.utils.params import ErlangParams, H2Params

MU = 1.3
ERLANG = ErlangParams(r=3, mu=2.0)
H2 = H2Params(p1=0.4, mu1=2.0, mu2=0.5)


def test_ph_moments_match_exponential():
    """PH built from Exp(mu) reproduces the exponential raw moments."""
    ph = PHDistribution.from_exp(MU)
    assert np.allclose(PHDistribution.calc_theory_moments(ph, 4), ExpDistribution.calc_theory_moments(MU, 4))


def test_ph_moments_match_erlang():
    """PH chain of r phases reproduces the Erlang raw moments."""
    ph = PHDistribution.from_erlang(ERLANG)
    assert np.allclose(PHDistribution.calc_theory_moments(ph, 4), ErlangDistribution.calc_theory_moments(ERLANG, 4))


def test_ph_moments_match_h2():
    """Two-phase parallel PH reproduces the hyperexponential raw moments."""
    ph = PHDistribution.from_h2(H2)
    assert np.allclose(PHDistribution.calc_theory_moments(ph, 4), H2Distribution.calc_theory_moments(H2, 4))


def test_ph_pdf_cdf_sanity():
    """CDF tends to 1, PDF is non-negative and consistent with the CDF slope."""
    ph = PHDistribution.from_h2(H2)
    assert PHDistribution.get_cdf(ph, 50.0) > 0.999999
    x, dx = 0.7, 1e-6
    slope = (PHDistribution.get_cdf(ph, x + dx) - PHDistribution.get_cdf(ph, x - dx)) / (2 * dx)
    assert np.isclose(slope, PHDistribution.get_pdf(ph, x), rtol=1e-5)


def test_ph_sampling_moments():
    """Sampled mean and second moment agree with theory."""
    ph = PHDistribution(PHDistribution.from_erlang(ERLANG), generator=np.random.default_rng(42))
    samples = np.array([ph.generate() for _ in range(60_000)])
    m = PHDistribution.calc_theory_moments(ph.params, 2)
    assert np.isclose(samples.mean(), m[0], rtol=0.02)
    assert np.isclose((samples**2).mean(), m[1], rtol=0.05)


def test_map_poisson_is_poisson():
    """One-phase MAP: exponential interarrivals, zero lag correlation."""
    m = MAP.poisson(MU)
    assert np.allclose(MAP.calc_theory_moments(m, 3), ExpDistribution.calc_theory_moments(MU, 3))
    assert np.isclose(MAP.arrival_rate(m), MU)
    assert abs(MAP.lag_correlation(m, 1)) < 1e-12


def test_map_ph_renewal_uncorrelated():
    """Renewal MAP from an H2: H2 interarrival moments, zero lag correlation."""
    m = MAP.from_ph_renewal(PHDistribution.from_h2(H2))
    assert np.allclose(MAP.calc_theory_moments(m, 3), H2Distribution.calc_theory_moments(H2, 3))
    for k in (1, 2, 5):
        assert abs(MAP.lag_correlation(m, k)) < 1e-10


def test_mmpp_bursty_positive_correlation():
    """Two-state MMPP with distinct rates is positively correlated at lag 1."""
    m = MAP.mmpp([5.0, 0.5], np.array([[-0.1, 0.1], [0.2, -0.2]]))
    corr1 = MAP.lag_correlation(m, 1)
    assert corr1 > 0.05
    # correlation decays with lag
    assert MAP.lag_correlation(m, 5) < corr1


def test_mmpp_rate_and_sampling():
    """MMPP fundamental rate matches theory; sampled mean interval = 1/lambda."""
    m = MAP.mmpp([5.0, 0.5], np.array([[-0.1, 0.1], [0.2, -0.2]]))
    lam = MAP.arrival_rate(m)
    pi = MAP.phase_stationary(m)
    assert np.isclose(lam, pi[0] * 5.0 + pi[1] * 0.5)

    src = MAP(m, generator=np.random.default_rng(7))
    samples = np.array([src.generate() for _ in range(120_000)])
    assert np.isclose(samples.mean(), 1.0 / lam, rtol=0.03)
    # empirical lag-1 correlation close to theoretical
    emp = np.corrcoef(samples[:-1], samples[1:])[0, 1]
    assert np.isclose(emp, MAP.lag_correlation(m, 1), atol=0.03)


if __name__ == "__main__":
    test_ph_moments_match_exponential()
    test_ph_moments_match_erlang()
    test_ph_moments_match_h2()
    test_ph_pdf_cdf_sanity()
    test_ph_sampling_moments()
    test_map_poisson_is_poisson()
    test_map_ph_renewal_uncorrelated()
    test_mmpp_bursty_positive_correlation()
    test_mmpp_rate_and_sampling()
