"""v9 checks: Takahashi-Takami probabilities + tagged equations; quadrature correction."""

import numpy as np
import pytest

from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from works.negative_queues.revision_v7_20261107 import ph_queue
from works.negative_queues.revision_v7_20261107.positive_ph import gamma_ph
from works.negative_queues.revision_v9_20261108 import tt_tagged


@pytest.mark.parametrize("cv", [1.2, 0.8, 0.5])
@pytest.mark.parametrize("mode", ["rcs", "disaster"])
def test_tt_probabilities_with_tagged_equations_match_direct_solution(cv, mode):
    b = gamma_moments_by_mean_and_cv(2.1, cv)
    r = tt_tagged.calculate(b, 3, 1.0, 0.3, mode, levels=150, behind=40)
    y = [complex(*a) for a in r["h2"]["y"]]
    mu = [complex(*a) for a in r["h2"]["mu"]]
    d = ph_queue.calculate(ph_queue.hyperexponential(y, mu), n=3, lam=1.0, delta=0.3, mode=mode, action="remove",
                           levels=150, behind=40)
    np.testing.assert_allclose(r["w"] + r["v"], d["w"] + d["v"], rtol=1e-8)
    assert r["balance_residual"] < 1e-9
    assert max(r["little_residuals"]) < 1e-8


def test_rcs_sojourn_differs_from_constant_rate_library_value():
    b = gamma_moments_by_mean_and_cv(2.1, 1.2)
    r = tt_tagged.calculate(b, 3, 1.0, 0.3, "rcs", levels=150, behind=40)
    assert r["w"][0] == pytest.approx(r["tt_library"]["w"][0], rel=1e-7)
    assert r["v"][0] == pytest.approx(1.9220276572, rel=1e-8)
    assert r["tt_library"]["v"][0] > r["v"][0] * (1 + 1e-4)


def test_quadrature_correction_recovers_reference_at_zero_rate():
    # At delta=0 the corrected value equals the reference by construction; the
    # check is that the J-error of the retained-work model is negative and
    # shrinks with J (the premise of the correction), n=2, CV=0.8.
    lam = 0.5 * 2 / 2.1
    ref = ph_queue.calculate(gamma_ph(2.1, 0.8, 16), n=2, lam=lam, delta=0.0, action="restart", levels=120, behind=0)
    errors = []
    for j in (3, 4):
        ext = sum(c * ph_queue.calculate(ph_queue.fixed_work(0.8, j, k), n=2, lam=lam, delta=0.0, action="fixed",
                                         levels=120, behind=0)["w"][0] for c, k in zip((2, -9, 8), (8, 12, 16)))
        errors.append(ext - ref["w"][0])
    assert errors[0] < errors[1] < 0
    assert abs(errors[1]) < 0.5 * abs(errors[0])
