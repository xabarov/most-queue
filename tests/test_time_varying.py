"""
Tests for the non-stationary Mt/M/c calculator (PSA / MOL) and its paired
loss-system simulator.
"""

import numpy as np
import pytest

from most_queue.sim.time_varying import TimeVaryingMMcSim
from most_queue.theory.time_varying import TimeVaryingMMcCalc, erlang_b, erlang_c


def test_erlang_b_known_values():
    # a=1 offered load, c=1 server -> B = 1/2
    assert erlang_b(1.0, 1) == pytest.approx(0.5, abs=1e-9)
    # a=2, c=2 -> B = (2^2/2!) / (1 + 2 + 2^2/2!) = 2 / 5 = 0.4
    assert erlang_b(2.0, 2) == pytest.approx(0.4, abs=1e-9)


def test_erlang_c_reduces_and_bounds():
    # overload a>=c -> saturated, P(wait)=1
    assert erlang_c(3.0, 2) == pytest.approx(1.0, abs=1e-9)
    # single server: Erlang C == rho
    assert erlang_c(0.5, 1) == pytest.approx(0.5, abs=1e-9)
    # Erlang C >= Erlang B always
    for a, c in [(1.0, 2), (3.0, 5), (0.7, 3)]:
        assert erlang_c(a, c) >= erlang_b(a, c) - 1e-12


def test_constant_rate_matches_stationary_erlang_b():
    """A constant lambda(t) must give PSA == MOL == the stationary Erlang B value."""
    mu, c = 1.0, 3
    a = 2.0
    calc = TimeVaryingMMcCalc(n=c, kind="loss")
    calc.set_sources(lambda t: a * mu)
    calc.set_servers(mu)
    t_grid = np.linspace(0, 10, 20)
    res = calc.run(t_grid, mol_warmup=20.0)
    expected = erlang_b(a, c)
    assert np.allclose(res.psa, expected, atol=1e-9)
    assert np.allclose(res.mol, expected, atol=1e-3)


def test_psa_is_pointwise_erlang():
    mu, c = 1.0, 4

    def lam(t):
        return 3.0 * (1 + 0.5 * np.sin(t))

    calc = TimeVaryingMMcCalc(n=c, kind="delay")
    calc.set_sources(lam)
    calc.set_servers(mu)
    t_grid = np.linspace(0, 6, 15)
    res = calc.run(t_grid, mol_warmup=10.0)
    for t, p in zip(t_grid, res.psa):
        assert p == pytest.approx(erlang_c(lam(t) / mu, c), abs=1e-9)


def test_mol_beats_psa_under_fast_variation():
    """MOL should track the true blocking probability better than PSA when
    lambda(t) varies fast relative to the service rate."""
    mu, c = 1.0, 5
    w = 4.0
    period = 2 * np.pi / w
    lam0, amp = 4.0, 0.6

    def lam(t):
        return lam0 * (1 + amp * np.sin(w * t))

    lam_max = lam0 * (1 + amp) * 1.001
    nb = 40
    t_grid = np.array([(i + 0.5) / nb * period for i in range(nb)])

    calc = TimeVaryingMMcCalc(n=c, kind="loss")
    calc.set_sources(lam)
    calc.set_servers(mu)
    res = calc.run(t_grid, mol_warmup=8.0 / mu)

    sim = TimeVaryingMMcSim(n=c, period=period, n_buckets=nb, seed=42)
    sim.set_sources(lam, lam_max)
    sim.set_servers(mu)
    _, prob = sim.run(horizon=period * 3000)
    prob = np.array(prob)

    err_psa = np.mean(np.abs(res.psa - prob))
    err_mol = np.mean(np.abs(res.mol - prob))
    assert err_mol < err_psa
    assert err_mol < 0.06


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        TimeVaryingMMcCalc(n=2, kind="bogus")
