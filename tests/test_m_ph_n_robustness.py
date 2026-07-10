"""
Robustness tests for the M/PH/k two-class priority solver (MPhNPrty) across the
parameter range of the RDR paper's Fig. 7 (high class PH, varying C^2, k servers).

Covers two fixes:
  * the k=1 singular-matrix regression in the passage-time closed form
    (`_G_calc`), which now falls back to functional iteration;
  * the base Takahashi-Takami solver's divergence guards, which fail loudly
    instead of hanging forever or returning silent garbage.
"""

import numpy as np
import pytest

from most_queue.random.distributions import GammaDistribution
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.preemptive.m_ph_n_busy_approx import MPhNPrty, TakahashiTakamiParams


def _mphn(k, c2, rho=0.8):
    mean = float(k)  # fixed total capacity: each server has mean service = k
    lam_high = lam_low = rho * k / 2 / mean
    b_high = gamma_moments_by_mean_and_cv(mean, np.sqrt(c2))
    cp = TakahashiTakamiParams()
    cp.max_iter = 300
    tt = MPhNPrty(n=k, calc_params=cp)
    tt.set_sources(l_low=lam_low, l_high=lam_high)
    tt.set_servers(b_high=b_high, mu_low=1.0 / mean)
    return tt, b_high, lam_high, lam_low


@pytest.mark.parametrize("c2", [2.0, 4.0, 8.0])
def test_mph1_two_class_across_cv(c2):
    """
    M/PH/1 with two classes must compute for a range of C^2 (the k=1, C^2 in
    {4, ...} cases used to raise `LinAlgError: Singular matrix`) and agree with
    simulation.
    """
    tt, b_high, lam_high, lam_low = _mphn(k=1, c2=c2)
    res = tt.run()
    v_low = float(np.asarray(res.v[1][0]).real)
    assert np.isfinite(v_low) and v_low > 0

    qs = PriorityQueueSimulator(1, 2, "PR")
    qs.set_sources([{"type": "M", "params": lam_high}, {"type": "M", "params": lam_low}])
    gp = GammaDistribution.get_params([b_high[0], b_high[1]])
    qs.set_servers([{"type": "Gamma", "params": gp}, {"type": "M", "params": 1.0}])
    v_low_sim = qs.run(200_000).v[1][0]
    assert np.isclose(v_low, v_low_sim, rtol=0.15), f"M/PH/1 low class {v_low:.3f} vs sim {v_low_sim:.3f}"


def test_divergence_fails_loudly_not_silently():
    """
    An ill-conditioned large-chain regime (M/PH/4 fixed-capacity, C^2=8) must
    raise a clear error rather than hang forever or return silent garbage
    (previously the low-class mean came back as ~-7e40).
    """
    tt, _, _, _ = _mphn(k=4, c2=8.0)
    with pytest.raises(FloatingPointError):
        tt.run()


if __name__ == "__main__":
    test_mph1_two_class_across_cv(4.0)
    test_divergence_fails_loudly_not_silently()
