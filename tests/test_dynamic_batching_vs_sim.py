"""
The numerical embedded-chain analysis against independent event simulation
and the bounds of the source paper.

The chain route watches the system only at batch-completion epochs. A chain
built at embedded epochs can be internally consistent while describing the
wrong continuous-time system, so the event-by-event run is the reference that
settles it.

Warmup is excluded consistently from all statistics. No sign of transient bias
is inferred from a handful of seeds; replications quantify sampling uncertainty.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from most_queue.sim.dynamic_batching import DynamicBatchingSim
from most_queue.theory.batch.dynamic_batching import DynamicBatchingCalc

ALPHA, TAU0 = 0.0025, 0.02  # illustrative seconds, not a GPU calibration

PARAMS = yaml.safe_load((Path(__file__).parent / "default_params.yaml").read_text())["dynamic_batching"]


@pytest.mark.parametrize(
    "kind,params",
    [
        ("deterministic", {}),
        ("exponential", {}),
        ("erlang", {"shape": 3}),
        ("gamma", {"shape": 2.5}),
        ("h2", {"coev": 1.5}),
    ],
)
def test_all_service_laws_against_independent_event_simulation(kind, params):
    calc = DynamicBatchingCalc().set_sources(1).set_servers(0.1, 0.5, kind=kind, **params)
    expected = calc.run()
    rows = []
    for seed in range(PARAMS["replications"]):
        sim = DynamicBatchingSim(seed=seed).set_sources(1).set_servers(0.1, 0.5, kind=kind, **params)
        run = sim.run(PARAMS["jobs_per_replication"])
        rows.append([run.waiting_mean, run.latency_mean, run.batch_mean, run.idle_fraction])
    rows = np.array(rows)
    truth = np.array([expected.waiting_mean, expected.latency_mean, expected.batch_mean, expected.idle_probability])
    standard_error = rows.std(axis=0, ddof=1) / np.sqrt(len(rows))
    tolerance = PARAMS["standard_errors"] * standard_error + PARAMS["relative_floor"] * truth
    assert np.all(np.abs(rows.mean(axis=0) - truth) < tolerance), (rows.mean(axis=0), truth, standard_error)


@pytest.mark.slow
def test_heavy_load_with_independent_replications():
    calc = _exact(0.9)
    waits, latencies = _simulated(0.9, range(10), 200_000)
    for measured, expected in ((waits, calc.get_waiting_mean()), (latencies, calc.get_latency())):
        se = measured.std(ddof=1) / np.sqrt(len(measured))
        assert abs(measured.mean() - expected) < PARAMS["standard_errors"] * se


def _exact(load, alpha=ALPHA, tau0=TAU0):
    calc = DynamicBatchingCalc()
    calc.set_sources(load / alpha).set_deterministic_service(alpha, tau0)
    return calc


def _simulated(load, seeds, jobs, alpha=ALPHA, tau0=TAU0):
    """Mean waiting and latency over independent replications, with warm-up discarded."""
    waits, latencies = [], []
    for seed in seeds:
        sim = DynamicBatchingSim(seed=seed)
        sim.set_sources(load / alpha).set_deterministic_service(alpha, tau0)
        run = sim.run(total_served=jobs, warmup_fraction=0.2)
        waits.append(run.waiting_mean)
        latencies.append(run.latency_mean)
    return np.array(waits), np.array(latencies)


@pytest.mark.parametrize("load", [0.3, 0.6])
def test_simulation_confirms_the_exact_waiting_time(load):
    calc = _exact(load)
    waits, latencies = _simulated(load, range(10), 60_000)
    for measured, exact in ((waits, calc.get_waiting_mean()), (latencies, calc.get_latency())):
        error = measured.std(ddof=1) / np.sqrt(measured.size)
        assert abs(measured.mean() - exact) / error < 3.5, (load, measured.mean(), exact, error)


def test_simulation_confirms_the_mean_batch_size():
    calc = _exact(0.6)
    sim = DynamicBatchingSim(seed=5)
    sim.set_sources(0.6 / ALPHA).set_deterministic_service(ALPHA, TAU0)
    assert sim.run(total_served=80_000).batch_mean == pytest.approx(calc.get_batch_moments(1)[0], rel=0.05)


def test_simulation_confirms_the_idle_probability_at_light_load():
    """
    The idle probability is what the paper has to bound, so it is worth
    checking on its own rather than only through the latency it feeds.
    """
    calc = _exact(0.1)
    sim = DynamicBatchingSim(seed=7)
    sim.set_sources(0.1 / ALPHA).set_deterministic_service(ALPHA, TAU0)
    assert sim.run(total_served=80_000).idle_fraction == pytest.approx(calc.get_idle_probability(), abs=0.02)


# --------------------------------------------------------------------------
# what the published bound gives away
# --------------------------------------------------------------------------


def test_the_bound_becomes_close_when_idle_probability_is_small():
    """
    The paper's ``phi1`` is its Lemma 4 evaluated at zero idle probability, so
    its error becomes small as idle probability decreases. For these parameters
    it meets the stated tolerances; no universal load cutoff is asserted.
    """
    for load, tolerance in ((0.5, 1e-4), (0.7, 1e-6), (0.9, 1e-8)):
        report = _exact(load).bound_report()
        assert report["idle_probability"] < 1e-3, load
        # The residual slack is exactly the residual idle probability, so it
        # shrinks with the load rather than vanishing at any particular point.
        assert report["phi1"] == pytest.approx(report["latency"], rel=tolerance), load


def test_the_bound_is_loose_only_at_light_load_and_the_slack_is_the_idle_probability():
    """
    At a load of 0.1 the best published bound overstates the latency by about
    7%, and the reason is visible: the true idle probability is three times the
    lower bound the paper has to use for it.
    """
    report = _exact(0.1).bound_report()
    assert report["best_bound_ratio"] > 1.05
    assert report["idle_probability"] > 2.5 * report["idle_lower_bound"]
    # ... and feeding the exact idle probability into the same formula removes
    # all of the slack, which localises it precisely.
    assert report["lemma4_at_idle"] == pytest.approx(report["latency"], rel=1e-9)


def test_the_slack_closes_as_the_load_grows():
    previous = None
    for load in (0.1, 0.2, 0.3, 0.5):
        ratio = _exact(load).bound_report()["best_bound_ratio"]
        if previous is not None:
            assert ratio <= previous + 1e-9, load
        previous = ratio
