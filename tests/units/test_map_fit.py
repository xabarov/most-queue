"""
Unit tests for MMPP-2 fitting (most_queue.random.map_fit).
"""

import numpy as np
import pytest

from most_queue.random.map_fit import fit_map_from_trace, fit_mmpp2, map_statistics, trace_statistics
from most_queue.random.map_ph import MAP

KNOWN_MMPPS = [
    MAP.mmpp([2.0, 0.4], np.array([[-0.2, 0.2], [0.3, -0.3]])),
    MAP.mmpp([5.0, 0.5], np.array([[-0.1, 0.1], [0.05, -0.05]])),
    MAP.mmpp([1.5, 0.8], np.array([[-0.5, 0.5], [0.5, -0.5]])),
]


@pytest.mark.parametrize("mmpp", KNOWN_MMPPS)
def test_fit_roundtrip(mmpp):
    """Fitting to a known MMPP's own statistics reproduces those statistics."""
    target = map_statistics(mmpp)
    fitted = fit_mmpp2(*target)
    achieved = map_statistics(fitted)
    assert np.allclose(target, achieved, atol=1e-3), (target, achieved)


def test_fit_produces_valid_map():
    """The fitted object is a valid MAP (D0+D1 is a generator, D1 >= 0)."""
    fitted = fit_mmpp2(rate=1.0, scv=3.0, lag1=0.2)
    d0, d1 = np.asarray(fitted.D0), np.asarray(fitted.D1)
    assert np.allclose((d0 + d1) @ np.ones(2), 0.0, atol=1e-9)
    assert np.all(d1 >= -1e-12)


def test_fit_from_trace():
    """Fitting from a simulated trace recovers the empirical statistics."""
    src = MAP(KNOWN_MMPPS[0], generator=np.random.default_rng(7))
    trace = [src.generate() for _ in range(200_000)]
    fitted = fit_map_from_trace(trace)
    emp = trace_statistics(trace)
    got = map_statistics(fitted)
    assert np.allclose(emp, got, atol=2e-3), (emp, got)
    # and the empirical stats are close to the true generating MMPP
    true = map_statistics(KNOWN_MMPPS[0])
    assert np.isclose(emp[0], true[0], rtol=0.02)
    assert np.isclose(emp[1], true[1], rtol=0.05)


def test_infeasible_targets_rejected():
    """MMPP cannot represent SCV < 1 or negative lag-1 correlation."""
    with pytest.raises(ValueError, match="SCV"):
        fit_mmpp2(rate=1.0, scv=0.5, lag1=0.1)
    with pytest.raises(ValueError, match="lag-1"):
        fit_mmpp2(rate=1.0, scv=2.0, lag1=-0.2)
    with pytest.raises(ValueError, match="rate"):
        fit_mmpp2(rate=-1.0, scv=2.0, lag1=0.1)


def test_near_poisson():
    """SCV≈1, lag1≈0 fits to a near-Poisson MMPP."""
    fitted = fit_mmpp2(rate=2.0, scv=1.0, lag1=0.0)
    rate, scv, lag1 = map_statistics(fitted)
    assert np.isclose(rate, 2.0, rtol=1e-3)
    assert np.isclose(scv, 1.0, atol=1e-2)
    assert abs(lag1) < 1e-2


if __name__ == "__main__":
    for mp in KNOWN_MMPPS:
        test_fit_roundtrip(mp)
    test_fit_produces_valid_map()
    test_fit_from_trace()
    test_infeasible_targets_rejected()
    test_near_poisson()
    print("all map_fit tests passed")
