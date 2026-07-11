"""
Tests for the exact two-class M/PH/PH/k preemptive-resume solver (the §2.3 base
case of the RDR paper), where the low (target) class keeps a phase-type service.

Validated three independent ways:
  * both classes exponential  == MMkPriorityExact (exact CTMC),
  * high PH / low exponential == MPhNPrty (validated Takahashi-Takami solver),
  * both PH                   == an independent FCFS-resume work-based simulation
    (checked offline; the value is pinned here as a regression guard).
"""

import numpy as np
import pytest

from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.theory.priority.preemptive.m_ph_n_busy_approx import MPhNPrty, TakahashiTakamiParams
from most_queue.theory.priority.preemptive.mmk_prty_exact import MMkPriorityExact
from most_queue.theory.priority.preemptive.mph_ph_k_2class import MPhPhK2Class, PhaseType


@pytest.mark.parametrize(
    "n, lH, lL, muH, muL",
    [(1, 0.4, 0.3, 1.0, 1.0), (2, 0.4, 0.4, 1.0, 1.0), (2, 0.5, 0.3, 1.2, 0.8)],
)
def test_exponential_matches_exact_ctmc(n, lH, lL, muH, muL):
    """Both classes exponential must reproduce the exact CTMC (anchor)."""
    ph = MPhPhK2Class(n=n, truncation=80)
    ph.set_sources(l_high=lH, l_low=lL)
    ph.set_servers(PhaseType.exponential(muH), PhaseType.exponential(muL))
    r = ph.run()

    ex = MMkPriorityExact(n=n, truncation=150)
    ex.set_sources([lH, lL])
    ex.set_servers([muH, muL])
    re = ex.run()

    assert np.isclose(float(r.v[0][0]), float(re.v[0][0]), rtol=1e-3)
    assert np.isclose(float(r.v[1][0]), float(re.v[1][0]), rtol=1e-3)


@pytest.mark.parametrize("c2", [2.0, 8.0])
def test_high_ph_low_exp_matches_mphnprty(c2):
    """High PH / low exponential must match the validated MPhNPrty solver."""
    lH = lL = 0.4
    muL = 1.0
    b_high = gamma_moments_by_mean_and_cv(1.0, np.sqrt(c2))

    ph = MPhPhK2Class(n=2, truncation=70)
    ph.set_sources(l_high=lH, l_low=lL)
    ph.set_servers(PhaseType.from_moments(b_high), PhaseType.exponential(muL))
    r = ph.run()

    cp = TakahashiTakamiParams()
    cp.max_iter = 300
    mp = MPhNPrty(n=2, calc_params=cp)
    mp.set_sources(l_low=lL, l_high=lH)
    mp.set_servers(b_high=b_high, mu_low=muL)
    rm = mp.run()

    assert np.isclose(float(r.v[0][0]), float(np.asarray(rm.v[0][0]).real), rtol=2e-3)
    assert np.isclose(float(r.v[1][0]), float(np.asarray(rm.v[1][0]).real), rtol=2e-3)


def test_both_ph_regression():
    """
    Both classes PH (Coxian, C^2 = 8). The low-class mean is pinned to the value
    confirmed by an independent FCFS-resume work-based simulation (2.078).
    """
    b = gamma_moments_by_mean_and_cv(1.0, np.sqrt(8.0))
    ph = MPhPhK2Class(n=2, truncation=70)
    ph.set_sources(l_high=0.4, l_low=0.4)
    ph.set_servers(PhaseType.from_moments(b), PhaseType.from_moments(b))
    r = ph.run()
    assert np.isclose(float(r.v[0][0]), 1.162, rtol=0.02)
    assert np.isclose(float(r.v[1][0]), 2.078, rtol=0.02)


if __name__ == "__main__":
    test_exponential_matches_exact_ctmc(2, 0.4, 0.4, 1.0, 1.0)
    test_both_ph_regression()
