"""
Unit tests for the unified Erlang/H2 auto-dispatch
(most_queue.theory.batch.bulk_service_general, EPIC-037).
"""

import numpy as np
import pytest

from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc
from most_queue.theory.batch.bulk_service_general import fit_bulk_service_calc
from most_queue.theory.batch.bulk_service_h2 import BulkServiceH2Calc


def test_auto_picks_erlang_for_low_cv_and_matches_direct_construction():
    a, b, lam = 1, 4, 1.0
    mean, m2 = 2.0, 4.5  # var = 0.5 < mean^2 -> cv < 1

    calc = fit_bulk_service_calc(a, b, [mean, m2], family="auto", queue_truncation=250)
    assert isinstance(calc, BulkServiceErlangCalc)
    calc.set_sources(lam)
    v = calc.run().v[0]

    ref = BulkServiceErlangCalc(a=a, b=b, k=1, queue_truncation=250)
    ref.set_sources(lam)
    ref.set_servers_from_moments([mean, m2])
    v_ref = ref.run().v[0]

    assert np.isclose(v, v_ref, rtol=1e-9)


def test_auto_picks_h2_for_high_cv_and_matches_direct_construction():
    a, b, lam = 1, 4, 1.0
    mean, m2, m3 = 2.0, 12.0, 100.0  # var >> mean^2 -> cv > 1

    calc = fit_bulk_service_calc(a, b, [mean, m2, m3], family="auto", queue_truncation=250)
    assert isinstance(calc, BulkServiceH2Calc)
    calc.set_sources(lam)
    v = calc.run().v[0]

    ref = BulkServiceH2Calc(a=a, b=b, queue_truncation=250)
    ref.set_sources(lam)
    ref.set_servers_from_moments([mean, m2, m3])
    v_ref = ref.run().v[0]

    assert np.isclose(v, v_ref, rtol=1e-9)


def test_auto_resolves_boundary_cv_one_to_erlang():
    mean = 2.0
    m2 = 2 * mean * mean  # cv == 1 exactly (exponential moments)
    calc = fit_bulk_service_calc(1, 4, [mean, m2], family="auto", queue_truncation=250)
    assert isinstance(calc, BulkServiceErlangCalc)


def test_forced_erlang_rejects_high_cv():
    mean, m2 = 2.0, 12.0  # cv > 1
    with pytest.raises(ValueError):
        fit_bulk_service_calc(1, 4, [mean, m2], family="erlang")


def test_forced_h2_rejects_low_cv():
    mean, m2, m3 = 2.0, 4.5, 12.0  # cv < 1
    with pytest.raises(ValueError):
        fit_bulk_service_calc(1, 4, [mean, m2, m3], family="h2")


def test_forced_h2_requires_three_moments():
    mean, m2 = 2.0, 12.0  # cv > 1 but only 2 moments given
    with pytest.raises(ValueError):
        fit_bulk_service_calc(1, 4, [mean, m2], family="h2")


def test_invalid_family_rejected():
    with pytest.raises(ValueError):
        fit_bulk_service_calc(1, 4, [2.0, 4.5], family="bogus")  # type: ignore[arg-type]


if __name__ == "__main__":
    test_auto_picks_erlang_for_low_cv_and_matches_direct_construction()
    test_auto_picks_h2_for_high_cv_and_matches_direct_construction()
    test_auto_resolves_boundary_cv_one_to_erlang()
    test_forced_erlang_rejects_high_cv()
    test_forced_h2_rejects_low_cv()
    test_forced_h2_requires_three_moments()
    test_invalid_family_rejected()
    print("all bulk-service auto-dispatch tests passed")
