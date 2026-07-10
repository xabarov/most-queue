"""
Tests for the learning-augmented (prediction-based) scheduling degradation
analysis: SPJF mean response time as prediction quality degrades, bracketed by
SRPT / SJF / blind FB.
"""

import warnings

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.theory.srpt import MG1SjfCalc, MG1SpjfCalc, PerfectPredictor, prediction_degradation_curve

warnings.filterwarnings("ignore")


def _h2(cv):
    return H2Distribution.get_params(gamma_moments_by_mean_and_cv(1.0, cv))


def test_spjf_perfect_reduces_to_sjf():
    """SPJF with a perfect predictor equals SJF."""
    hp = _h2(1.5)
    spjf = MG1SpjfCalc()
    spjf.set_sources(0.7)
    spjf.set_servers(hp, "H")
    spjf.set_predictor(PerfectPredictor())
    v_spjf = spjf.run().v[0]

    sjf = MG1SjfCalc()
    sjf.set_sources(0.7)
    sjf.set_servers(hp, "H")
    assert np.isclose(v_spjf, sjf.run().v[0], rtol=1e-6)


def test_degradation_curve_monotone_and_bracketed():
    """
    The SPJF mean response grows with prediction noise, starts at ~SJF for small
    noise, and SRPT is the lower bound.
    """
    hp = _h2(1.5)
    curve = prediction_degradation_curve(0.7, hp, "H", sigmas=(0.1, 0.3, 0.6, 1.0, 2.0))

    # monotone non-decreasing in noise
    assert all(b >= a - 1e-9 for a, b in zip(curve.spjf, curve.spjf[1:]))
    # smallest noise ~ SJF (perfect-prediction non-preemptive)
    assert np.isclose(curve.spjf[0], curve.sjf, rtol=0.02)
    # SRPT (size-aware preemptive) is the optimum
    assert curve.srpt < curve.sjf
    assert curve.srpt <= min(curve.spjf)


def test_bad_predictions_lose_to_blind():
    """
    The survey's central point: sufficiently noisy predictions make the size-based
    SPJF worse than a blind policy — a finite break-even noise level exists.
    """
    hp = _h2(1.5)
    curve = prediction_degradation_curve(0.7, hp, "H", sigmas=(0.1, 0.5, 1.0, 1.5, 2.0, 3.0))
    assert curve.breakeven_sigma is not None
    assert max(curve.spjf) > curve.blind_fb


if __name__ == "__main__":
    test_spjf_perfect_reduces_to_sjf()
    test_degradation_curve_monotone_and_bracketed()
    test_bad_predictions_lose_to_blind()
