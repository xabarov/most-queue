"""Independent limiting laws, input validation and simulation accounting for L1."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import gamma, poisson

from most_queue.sim.dynamic_batching import DynamicBatchingSim, DynamicBatchingSimResults
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.batch.bulk_service import BulkServiceMM1Calc
from most_queue.theory.batch.dynamic_batching import DynamicBatchingCalc


@pytest.mark.parametrize("cls", [DynamicBatchingCalc, DynamicBatchingSim])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf])
def test_invalid_rates(cls, value):
    with pytest.raises(ValueError, match="arrival rate"):
        cls().set_sources(value)


@pytest.mark.parametrize("cls", [DynamicBatchingCalc, DynamicBatchingSim])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": np.nan},
        {"tau0": np.inf},
        {"kind": "unknown"},
        {"kind": "erlang", "shape": 1.5},
        {"kind": "erlang", "shape": True},
        {"kind": "gamma", "shape": np.nan},
        {"kind": "gamma", "shape": -1},
        {"kind": "h2", "coev": 0.9},
        {"kind": "h2", "coev": np.inf},
    ],
)
def test_invalid_service(cls, kwargs):
    params = {"alpha": 0.1, "tau0": 0.5}
    params.update(kwargs)
    with pytest.raises(ValueError):
        cls().set_servers(**params)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_batch": 2.5},
        {"max_batch": True},
        {"max_states": 0},
        {"max_nonzeros": 0},
        {"max_states": 4, "max_batch": 5},
        {"tolerance": np.nan},
        {"tolerance": np.inf},
        {"tolerance": 1},
    ],
)
def test_invalid_budgets(kwargs):
    with pytest.raises(ValueError):
        DynamicBatchingCalc(**kwargs)


@pytest.mark.parametrize(
    "kind,params,scv,pzero",
    [
        ("deterministic", {}, 0, np.exp(-0.5)),
        ("exponential", {}, 1, 1 / 1.5),
        ("erlang", {"shape": 3}, 1 / 3, (1 + 0.5 / 3) ** -3),
        ("gamma", {"shape": 2.5}, 0.4, (1 + 0.5 / 2.5) ** -2.5),
        ("gamma", {"shape": 0.5}, 2, (1 + 0.5 / 0.5) ** -0.5),
        # CV=sqrt(2), balanced branch means with p=(1+1/sqrt(3))/2.
        ("h2", {"coev": np.sqrt(2)}, 2, None),
    ],
)
def test_constant_mean_iid_corner_for_every_law(kind, params, scv, pzero):
    calc = DynamicBatchingCalc().set_sources(1).set_servers(0, 0.5, kind=kind, **params)
    if pzero is None:
        p = (1 + 1 / np.sqrt(3)) / 2
        pzero = p / (1 + 0.5 / (2 * p)) + (1 - p) / (1 + 0.5 / (2 * (1 - p)))
    expected_mean = 0.5 + pzero
    expected_second = expected_mean + (1 + scv) * 0.25
    result = calc.run()
    assert result.batch_mean == pytest.approx(expected_mean, rel=1e-10)
    assert result.batch_second_moment == pytest.approx(expected_second, rel=1e-10)
    assert result.waiting_mean == pytest.approx((1 + scv) * 0.25 / (2 * expected_mean), rel=1e-10)
    assert result.idle_probability == pytest.approx(pzero / expected_mean, rel=1e-10)
    assert result.service_mean == pytest.approx(0.5)
    assert result.converged


@pytest.mark.parametrize("shape", [0.5, 2.5, 5])
def test_gamma_arrival_count_against_direct_poisson_mixture_integration(shape):
    calc = DynamicBatchingCalc().set_sources(1.3).set_gamma_service(0.1, 0.5, shape)
    mean = 0.8
    components = calc._count_components(mean)
    for k in (0, 1, 3, 8):
        integral = quad(
            lambda t: poisson.pmf(k, 1.3 * t) * gamma.pdf(t, shape, scale=mean / shape), 0, np.inf, epsabs=1e-11
        )[0]
        assert sum(weight * dist.pmf(k) for weight, dist in components) == pytest.approx(integral, abs=1e-10)


def test_phase_family_reductions():
    def result(kind, **kwargs):
        return DynamicBatchingCalc().set_sources(1).set_servers(0.1, 0.5, kind=kind, **kwargs).get_latency()

    assert result("gamma", shape=3) == pytest.approx(result("erlang", shape=3), rel=1e-12)
    assert result("gamma", shape=1) == pytest.approx(result("exponential"), rel=1e-12)
    assert result("h2", coev=1) == pytest.approx(result("exponential"), rel=1e-12)


@pytest.mark.parametrize(
    "kind,params,alpha",
    [
        ("exponential", {}, 0.8),
        ("gamma", {"shape": 2}, 0.9),
        ("h2", {"coev": 2}, 0.5),
    ],
)
def test_stability_does_not_imply_a_finite_mean_wait(kind, params, alpha):
    calc = DynamicBatchingCalc(max_batch=10).set_sources(1).set_servers(alpha, 0.2, kind=kind, **params)
    assert calc.load < 1
    with pytest.raises(ValueError, match="infinite"):
        calc.run()


def test_automatic_precision_is_not_silently_relaxed():
    calc = DynamicBatchingCalc(max_states=64).set_sources(200).set_servers(0.0025, 0.02)
    with pytest.raises(ValueError, match="converge"):
        calc.run()
    pinned = DynamicBatchingCalc(max_batch=4).set_sources(200).set_servers(0.0025, 0.02).run()
    assert not pinned.converged
    assert pinned.refinement_error is None
    assert pinned.moment_residual > 0.1


def test_refinement_reduces_actual_error_not_only_probability_mass():
    def run(size):
        return DynamicBatchingCalc(max_batch=size).set_sources(9).set_servers(0.1, 0.2).run()

    reference = run(512)
    coarse, fine = run(64), run(128)
    errors = [abs(result.waiting_mean / reference.waiting_mean - 1) for result in (coarse, fine)]
    assert errors[1] < errors[0] / 100
    assert coarse.moment_residual > fine.moment_residual
    assert coarse.boundary_second_moment_fraction > fine.boundary_second_moment_fraction


def test_independent_finite_cap_ctmc_approaches_unbounded_model():
    reference = DynamicBatchingCalc().set_sources(1).set_exponential_service(0.1, 0.5).get_waiting_mean()
    errors = []
    for cap in (4, 8, 16, 32):
        calc = BulkServiceMM1Calc(a=1, b=cap, queue_truncation=100)
        calc.set_sources(1)
        calc.set_servers(lambda b: 1 / (0.1 * b + 0.5))
        errors.append(abs(calc.get_w(1)[0] - reference))
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 3e-9


def test_standard_contract_and_cache_isolation():
    calc = DynamicBatchingCalc().set_sources(1).set_servers(0.1, 0.5)
    result = calc.run()
    assert isinstance(calc, BaseQueue)
    assert isinstance(result, QueueResults)
    assert calc.is_sources_set and calc.is_servers_set
    assert calc.get_w() == [result.waiting_mean]
    assert calc.get_v() == [result.latency_mean]
    assert result.p is None and result.pi is None
    for accessor in (calc.get_w, calc.get_v):
        with pytest.raises(ValueError, match="first"):
            accessor(2)
    with pytest.raises(ValueError):
        calc.get_batch_moments(3)
    result.batch_distribution[:] = 0
    calc.get_batch_distribution()[:] = 0
    assert calc.get_batch_distribution().sum() == pytest.approx(1)
    before = calc.get_latency()
    calc.set_servers(0.1, 0.2)
    assert calc.get_latency() != before
    before = calc.get_latency()
    calc.set_sources(2)
    assert calc.get_latency() != before


def test_bounds_refuse_unstable_parameters_before_dividing_by_zero():
    calc = DynamicBatchingCalc().set_sources(1).set_servers(1, 0.2)
    for accessor in (calc.bound_phi0, calc.bound_phi1, calc.bound_report):
        with pytest.raises(ValueError, match="unstable"):
            accessor()


def test_zero_overhead_lemma_is_an_independent_closed_form():
    for rate in (0.1, 0.5, 0.9):
        result = DynamicBatchingCalc().set_sources(rate).set_servers(1, 0).run()
        assert result.waiting_mean == pytest.approx(rate / (2 * (1 - rate**2)), rel=1e-9)
        assert result.service_mean == pytest.approx(1 / (1 - rate**2), rel=1e-9)
        assert result.idle_probability == pytest.approx(1 - rate, abs=1e-10)


def test_warmup_excludes_all_metrics_on_a_hand_computed_event_trace():
    class Arrivals:
        """Absolute arrivals: 1, 1.5, 2, 6; first batch ends at 3."""

        def __init__(self):
            self.intervals = iter([1, 0.5, 0.5, 4])

        def exponential(self, _scale):
            return next(self.intervals)

    sim = DynamicBatchingSim().set_sources(1).set_servers(0, 2)
    sim.generator = Arrivals()
    result = sim.run(2, warmup_fraction=0.4)  # discard one job, then a batch of two
    assert result.warmup_served == 1
    assert result.served == 2 and result.batches == 1
    assert result.batch_mean == 2
    assert result.waits.tolist() == [1.5, 1]
    assert result.latencies.tolist() == [3.5, 3]
    assert result.observation_time == 2
    assert result.idle_fraction == 0
    assert result.p == [0, 0, 1]


def test_simulation_alias_and_statistics_are_reproducible():
    def sim():
        return DynamicBatchingSim(seed=17).set_sources(1).set_h2_service(0.1, 0.5, 1.5)

    a, b = sim().run(1000), sim().run(num_of_jobs=1000)
    np.testing.assert_array_equal(a.waits, b.waits)
    assert a.served >= 1000
    assert sum(a.p) == pytest.approx(1)
    assert a.utilization == pytest.approx(1 - a.idle_fraction)
    assert a.w[0] == a.waiting_mean
    assert a.v[0] == a.latency_mean
    assert a.service_mean + a.waiting_mean == pytest.approx(a.latency_mean)
    assert a.cdf(-1) == 0 and a.cdf(np.inf, "v") == 1
    for kwargs in ({"of": "unknown"}, {"t": np.nan}):
        with pytest.raises(ValueError):
            a.cdf(**({"t": 0} | kwargs))
    with pytest.raises(ValueError, match="no samples"):
        DynamicBatchingSimResults().cdf(0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_served": 0},
        {"total_served": 1.5},
        {"total_served": True},
        {"warmup_fraction": -1},
        {"warmup_fraction": 1},
        {"warmup_fraction": np.nan},
    ],
)
def test_simulator_run_validation(kwargs):
    with pytest.raises(ValueError):
        DynamicBatchingSim().set_sources(1).set_servers(0.1, 0.5).run(**kwargs)
