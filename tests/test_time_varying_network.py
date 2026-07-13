"""
Test the time-varying network PSA: constant rate reduces exactly to the
Jackson network; slow sinusoidal modulation matches the phase-bucketed
simulation.
"""

import math

import numpy as np

from most_queue.sim.networks.time_varying_network import TimeVaryingNetworkSim
from most_queue.theory.networks.jackson_network import JacksonNetworkCalc
from most_queue.theory.networks.time_varying_network import TimeVaryingNetworkCalc

# Tandem: source -> 1 -> 2 -> out
ROUTING = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
MU = [1.0, 1.4]
CHANNELS = [1, 1]

PERIOD = 2000.0  # slow modulation: period >> relaxation time
LAM_MEAN = 0.5
LAM_AMPL = 0.2


def lam_fn(t):
    return LAM_MEAN + LAM_AMPL * math.sin(2 * math.pi * t / PERIOD)


def test_constant_rate_reduces_to_jackson():
    """lambda(t) = const must reproduce the stationary Jackson answer."""
    calc = TimeVaryingNetworkCalc()
    calc.set_sources(lam_fn=lambda t: LAM_MEAN, R=ROUTING)
    calc.set_nodes(mu=MU, n=CHANNELS)
    res = calc.run(t_grid=[0.0, 10.0, 100.0])

    jackson = JacksonNetworkCalc()
    jackson.set_sources(arrival_rate=LAM_MEAN, R=ROUTING)
    jackson.set_nodes(mu=MU, n=CHANNELS)
    ref = jackson.run()

    assert np.allclose(res.v, [ref.v[0]] * 3, rtol=1e-12)
    assert np.allclose(res.mean_jobs_total, [sum(ref.mean_jobs)] * 3, rtol=1e-12)


def test_psa_vs_simulation_slow_modulation():
    """Slow sinusoidal load: PSA mean-jobs profile tracks the simulation."""
    n_buckets = 8
    sim = TimeVaryingNetworkSim(period=PERIOD, n_buckets=n_buckets, seed=42)
    sim.set_sources(lam_fn=lam_fn, lam_max=LAM_MEAN + LAM_AMPL, R=ROUTING)
    sim.set_nodes(mu=MU, n=CHANNELS)
    t_centers, sim_jobs = sim.run(horizon=300 * PERIOD)

    calc = TimeVaryingNetworkCalc()
    calc.set_sources(lam_fn=lam_fn, R=ROUTING)
    calc.set_nodes(mu=MU, n=CHANNELS)
    res = calc.run(t_grid=t_centers)

    assert np.allclose(
        res.mean_jobs_total, sim_jobs, rtol=0.1, atol=0.1
    ), f"PSA {res.mean_jobs_total} vs sim {sim_jobs}"
    # The profile must actually vary over the cycle (peak vs trough)
    assert max(res.mean_jobs_total) > 1.3 * min(res.mean_jobs_total)


def test_psa_rejects_instability():
    """PSA must raise when lambda(t) makes some node unstable."""
    calc = TimeVaryingNetworkCalc()
    calc.set_sources(lam_fn=lambda t: 1.5, R=ROUTING)  # > mu_1
    calc.set_nodes(mu=MU, n=CHANNELS)
    try:
        calc.run(t_grid=[0.0])
        raised = False
    except ValueError:
        raised = True
    assert raised


if __name__ == "__main__":
    test_constant_rate_reduces_to_jackson()
    test_psa_vs_simulation_slow_modulation()
    test_psa_rejects_instability()
