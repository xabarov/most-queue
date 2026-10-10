"""
Cross-validate the M/M/c queue with queueing-time dependent service rates
(EPIC-076, roadmap item R2 -- D'Auria, Adan, Bekker & Kulkarni, EJOR
299(2):566-579, 2022) against discrete-event simulation.

The exact solution is checked in tests/units against three analytical
references (the Erlang-C reduction, the paper's c = 1 closed form, and the
paper's published c = 2 numbers). What simulation adds is an end-to-end check
of the modelling itself, on the regime the model exists for: slowdown and
speedup, where the service rate genuinely switches.

A note on the load. The mean VQT is compared only at moderate loads. At
rho = lam/(c*mu2) = 0.95 a simulation that starts empty needs far more than a
test-suite budget to settle: the control experiment recorded in the epic shows
the same simulator missing the TEXTBOOK Erlang-C mean by +1.4% at that load,
i.e. the deficit belongs to the simulator, not to the theory. The heavy case is
therefore checked on the distribution, which converges much faster, and the mean
is checked where the comparison is actually informative.
"""

import numpy as np
import pytest

from most_queue.sim.delay_dependent import MMcDelayDependentServiceSim
from most_queue.theory.delay_dependent import MMcDelayDependentServiceCalc

NUM_JOBS = 300_000


def _pair(c, k, lam, mu1, mu2, seed=42):
    calc = MMcDelayDependentServiceCalc(c=c, k=k)
    calc.set_sources(lam)
    calc.set_servers(mu1=mu1, mu2=mu2)

    sim = MMcDelayDependentServiceSim(c=c, k=k, seed=seed)
    sim.set_sources(lam)
    sim.set_servers(mu1=mu1, mu2=mu2)
    sim_res = sim.run(NUM_JOBS)
    return calc, sim, sim_res


@pytest.mark.parametrize(
    "c, k, lam, mu1, mu2, w_rtol",
    [
        (2, 0.45, 2.0, 0.75, 1.12, 0.05),  # the paper's own example (speedup)
        (3, 5.0, 2.0, 0.3, 0.8, 0.03),  # strong speedup
        (1, 2.0, 0.7, 1.4, 0.9, 0.04),  # single server, slowdown
        (3, 5.0, 1.5, 0.8, 0.7, 0.03),  # slowdown, rho = 0.71
        (2, 1.0, 1.0, 0.9, 0.7, 0.03),  # slowdown, two servers
    ],
)
def test_waiting_time_vs_sim(c, k, lam, mu1, mu2, w_rtol):
    """Mean, zero-wait atom and cdf of the virtual queueing time."""
    calc, sim, sim_res = _pair(c, k, lam, mu1, mu2)

    assert np.isclose(calc.get_w()[0], sim_res.w[0], rtol=w_rtol)
    assert np.isclose(calc.get_v()[0], sim_res.v[0], rtol=w_rtol)
    assert np.isclose(calc.get_p0_wait(), sim.get_p0_wait(), atol=0.005)
    for x in (k / 2, k, 2 * k + 1.0):
        assert np.isclose(calc.get_cdf(x), sim.get_cdf(x), atol=0.01)


def test_class_split_vs_sim():
    """
    P(W <= k) is what decides which rate a customer gets, so getting it right is
    the whole model. Checked against the simulated fraction directly.
    """
    c, k, lam, mu1, mu2 = 2, 0.45, 2.0, 0.75, 1.12
    calc, sim, _ = _pair(c, k, lam, mu1, mu2)
    sim_fraction = float((sim.waits <= k).mean())
    assert np.isclose(calc.get_class1_prob(), sim_fraction, atol=0.005)
    # and the realised mean service time that follows from it
    assert np.isclose(
        calc.get_service_time_mean(),
        float((sim.sojourns - sim.waits).mean()),
        rtol=0.02,
    )


def test_heavy_slowdown_vs_sim():
    """
    rho = 0.95 slowdown -- the regime the paper's slowdown figures live in, and
    the one where simulation is hardest.

    Several seeds are pooled and the budget is raised, because at this load a
    single 300k run is not merely noisy but biased: it starts empty and never
    reaches the long excursions that carry most of E[W]. Measured here: a single
    seed at 300k misses E[W] by up to 22%, six pooled seeds at 2M get within 4%
    and the cdf within 0.005. The mean is therefore asserted loosely and the
    distribution, which settles much faster, tightly. The equal-rate control in
    the epic confirms the residual is the simulator's -- it misses the TEXTBOOK
    Erlang-C mean by the same relative amount at this load.
    """
    calc = MMcDelayDependentServiceCalc(c=3, k=5.0)
    calc.set_sources(2.0)
    calc.set_servers(mu1=0.8, mu2=0.7)

    xs = (2.5, 5.0, 11.0)
    means, p0s, cdfs = [], [], []
    for seed in range(4):
        sim = MMcDelayDependentServiceSim(c=3, k=5.0, seed=seed)
        sim.set_sources(2.0)
        sim.set_servers(mu1=0.8, mu2=0.7)
        means.append(sim.run(1_500_000).w[0])
        p0s.append(sim.get_p0_wait())
        cdfs.append([sim.get_cdf(x) for x in xs])

    assert np.isclose(calc.get_w()[0], float(np.mean(means)), rtol=0.08)
    assert np.isclose(calc.get_p0_wait(), float(np.mean(p0s)), atol=0.005)
    assert np.allclose([calc.get_cdf(x) for x in xs], np.mean(cdfs, axis=0), atol=0.01)


def test_equal_rates_sim_agrees_with_the_mmc_reduction():
    """mu1 == mu2 must behave as an ordinary M/M/c in simulation too."""
    calc, sim, sim_res = _pair(3, 5.0, 2.0, 0.9, 0.9)
    assert np.isclose(calc.get_w()[0], sim_res.w[0], rtol=0.05)
    assert np.isclose(calc.get_p0_wait(), sim.get_p0_wait(), atol=0.005)
