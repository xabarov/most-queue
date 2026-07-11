"""
Test RDR-A (Recursive Dimensionality Reduction, aggregated) for M/M/n queues
with an arbitrary number m of preemptive-resume priority classes.

The canonical RDR setting (paper's Figure 5) uses one common service rate across
classes; there RDR-A reproduces the per-class mean response time within a few
percent of the discrete-event simulation.
"""

import numpy as np
import pytest

from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.preemptive.rdr_a import RDRAPriorityCalc

NUM_OF_JOBS = 300_000
# RDR-A is an approximation; a few percent is the expected agreement.
RTOL = 0.12


def _sim_mean_sojourn(n, lambdas, mus, num_jobs=NUM_OF_JOBS):
    m = len(lambdas)
    qs = PriorityQueueSimulator(n, m, "PR")
    qs.set_sources([{"type": "M", "params": lambdas[i]} for i in range(m)])
    qs.set_servers([{"type": "M", "params": mus[i]} for i in range(m)])
    res = qs.run(num_jobs)
    return [res.v[i][0] for i in range(m)]


@pytest.mark.parametrize(
    "n, lambdas, mus",
    [
        (2, [0.3, 0.3, 0.3], [1.0, 1.0, 1.0]),  # M/M/2, 3 classes, equal mu, rho = 0.45
        (2, [0.5, 0.4, 0.3], [1.0, 1.0, 1.0]),  # M/M/2, 3 classes, equal mu, rho = 0.6
        (3, [0.6, 0.6, 0.6, 0.6], [1.0, 1.0, 1.0, 1.0]),  # M/M/3, 4 classes, equal mu (Fig. 5 style)
        (2, [0.3, 0.3, 0.3], [1.2, 1.0, 0.8]),  # M/M/2, 3 classes, heterogeneous mu
        (3, [0.5, 0.5, 0.5, 0.5], [1.3, 1.1, 1.0, 0.9]),  # M/M/3, 4 classes, heterogeneous mu
    ],
)
def test_rdr_a_vs_simulation(n, lambdas, mus):
    """RDR-A per-class mean response time vs simulation."""
    calc = RDRAPriorityCalc(n=n)
    calc.set_sources(lambdas)
    calc.set_servers(mus)
    res = calc.run()

    v_calc = [float(np.asarray(res.v[i][0]).real) for i in range(len(lambdas))]
    v_sim = _sim_mean_sojourn(n, lambdas, mus)

    for i, (c, s) in enumerate(zip(v_calc, v_sim)):
        assert np.isclose(c, s, rtol=RTOL), f"class {i}: RDR-A E[T]={c:.4f} vs sim {s:.4f}"


def test_rdr_a_reduces_to_mmn_for_single_class():
    """With a single class RDR-A is exactly M/M/n."""
    from most_queue.theory.fifo.mmnr import MMnrCalc

    calc = RDRAPriorityCalc(n=2)
    calc.set_sources([1.2])
    calc.set_servers([1.0])
    v_rdr = calc.run().v[0][0]

    mmn = MMnrCalc(n=2, r=300)
    mmn.set_sources(l=1.2)
    mmn.set_servers(mu=1.0)
    v_mmn = mmn.run().v[0]

    assert np.isclose(float(np.asarray(v_rdr).real), float(v_mmn), rtol=1e-6)


def test_rdr_a_ph_reduces_to_two_class_exact():
    """For m=2, RDR-A with PH service is exactly the two-class MPhPhK2Class solver."""
    from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
    from most_queue.theory.priority.preemptive.mph_ph_k_2class import MPhPhK2Class, PhaseType
    from most_queue.theory.priority.preemptive.rdr_a import RDRAPriorityPH

    b = gamma_moments_by_mean_and_cv(1.0, np.sqrt(8.0))
    rdr = RDRAPriorityPH(n=2, truncation=70)
    rdr.set_sources([0.4, 0.4])
    rdr.set_servers([b, b])
    res = rdr.run()

    ph = MPhPhK2Class(n=2, truncation=70)
    ph.set_sources(l_high=0.4, l_low=0.4)
    ph.set_servers(PhaseType.from_moments(b), PhaseType.from_moments(b))
    ref = ph.run()

    assert np.isclose(float(res.v[0][0]), float(ref.v[0][0]), rtol=1e-6)
    assert np.isclose(float(res.v[1][0]), float(ref.v[1][0]), rtol=1e-6)


def test_rdr_a_ph_m4_matches_fcfs_resume():
    """
    m=4, all classes PH (Coxian, C^2=8), M/PH/2, rho=0.4. Per-class means are
    pinned to the values confirmed against an independent FCFS-resume work-based
    simulation (0.0-1.4% agreement).
    """
    from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
    from most_queue.theory.priority.preemptive.rdr_a import RDRAPriorityPH

    b = gamma_moments_by_mean_and_cv(1.0, np.sqrt(8.0))
    rdr = RDRAPriorityPH(n=2, truncation=60)
    rdr.set_sources([0.2, 0.2, 0.2, 0.2])
    rdr.set_servers([b, b, b, b])
    res = rdr.run()
    expected = [1.038, 1.208, 1.592, 2.334]  # FCFS-resume work-sim reference
    for i, e in enumerate(expected):
        assert np.isclose(float(res.v[i][0]), e, rtol=0.03), f"class {i}: {res.v[i][0]:.3f} vs {e}"


@pytest.mark.parametrize("mu_high, mu_low", [(1.2, 1.0), (1.1, 1.0), (0.95, 1.0)])
def test_two_class_convergence_close_mu(mu_high, mu_low):
    """
    Regression: the two-class RDR solver used to stop after a single Takahashi-Takami
    iteration for some close (mu_high, mu_low) pairs, because its convergence test tracked
    only the scalar max(x), which could coincidentally match the arbitrary seed x[0]=0.4.
    The full-vector convergence test must now iterate to the correct low-class mean.
    """
    from most_queue.theory.priority.preemptive.mmn_2cls_pr_busy_approx import MMnPR2ClsBusyApprox

    l_high = l_low = 0.3
    calc = MMnPR2ClsBusyApprox(n=2)
    calc.set_sources(l_low=l_low, l_high=l_high)
    calc.set_servers(mu_low=mu_low, mu_high=mu_high)
    v_low = float(np.asarray(calc.run().v[1][0]).real)

    assert calc.n_iter_ >= 2, "fixed point must not exit after a single iteration"

    v_sim = _sim_mean_sojourn(2, [l_high, l_low], [mu_high, mu_low])[1]
    assert np.isclose(v_low, v_sim, rtol=RTOL), f"low-class E[T]={v_low:.4f} vs sim {v_sim:.4f}"


if __name__ == "__main__":
    test_rdr_a_vs_simulation(2, [0.3, 0.3, 0.3], [1.0, 1.0, 1.0])
    test_rdr_a_reduces_to_mmn_for_single_class()
    test_two_class_convergence_close_mu(1.2, 1.0)
