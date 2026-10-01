"""Deterministic identities and input checks for the small PH-MSJ chains."""

import numpy as np
import pytest

from most_queue.random.map_ph import PHDistribution, PHParams
from most_queue.random.utils.params import Cox2Params, ErlangParams, H2Params
from most_queue.theory.msj import MsjClass, MsjExactCalc, MsjPHCalc, MsjSaturatedCalc


def configured(k=2, rates=(0.4, 0.2), needs=(1, 2), services=None, **kwargs):
    """Build a calculator without hiding the library's public setters."""
    calc = MsjPHCalc(k, **kwargs)
    calc.set_sources(rates)
    calc.set_servers(needs, services or [PHDistribution.from_exp(1) for _ in rates])
    return calc


def test_exp_reduction_accounts_for_admitted_arrivals():
    calc = configured(truncation=7)
    result = calc.run()
    old = MsjExactCalc(2, [MsjClass(0.4, 1, 1), MsjClass(0.2, 2, 1)], truncation=7)
    old_result = old.run()
    # The historical calculator divides by the offered, not admitted, rate.
    assert result.boundary_mass == pytest.approx(old.boundary_mass)
    assert np.allclose(result.v_per_class, np.array(old_result.v_per_class) / (1 - old.boundary_mass))
    assert np.allclose(np.array(result.v_per_class) - result.w_per_class, 1)
    assert result.stationary_residual < 1e-12
    assert sum(result.p) == pytest.approx(1)
    assert result.utilization == pytest.approx(result.offered_load * (1 - result.boundary_mass))
    assert len(result.v) == len(result.w) == 1  # no fictional zero higher moments
    # This nonminimal Cox representation is exactly Exp(1): it also checks
    # invariance to the chosen PH representation, not a new service law.
    hidden_exp = PHDistribution.from_cox(Cox2Params(mu1=2, mu2=1, p1=0.5))
    alternate = configured(services=[hidden_exp, PHDistribution.from_exp(1)], truncation=7).run()
    assert alternate.v_per_class == pytest.approx(result.v_per_class)
    assert alternate.stability_threshold == pytest.approx(result.stability_threshold)


@pytest.mark.parametrize("k,needs", [(1, [1, 1]), (2, [1, 2]), (3, [1, 2]), (3, [2, 3])])
def test_saturated_exp_reduction(k, needs):
    rates = [0.1, 0.2]
    services = [PHDistribution.from_exp(0.8), PHDistribution.from_exp(1.3)]
    calc = configured(k, rates, needs, services)
    old = MsjSaturatedCalc(k, [MsjClass(rate, need, mu) for rate, need, mu in zip(rates, needs, [0.8, 1.3])])
    assert calc.saturated_throughput() == pytest.approx(old.run())


@pytest.mark.parametrize(
    "service",
    [
        PHDistribution.from_erlang(ErlangParams(r=2, mu=2)),
        PHDistribution.from_cox(Cox2Params(mu1=2, mu2=1.5, p1=0.75)),
        PHDistribution.from_h2(H2Params(p1=0.25, mu1=0.5, mu2=1.5)),
    ],
)
def test_single_job_at_a_time_is_mg1(service):
    b1, b2 = PHDistribution.calc_theory_moments(service, 2)
    rate = 0.35 / b1
    calc = configured(k=3, rates=[rate], needs=[3], services=[service], truncation=60)
    assert calc.saturated_throughput() == pytest.approx(1 / b1)
    result = calc.run()
    expected_w = rate * b2 / (2 * (1 - rate * b1))
    assert result.boundary_mass < 1e-9
    assert result.w[0] == pytest.approx(expected_w, rel=1e-7)
    assert result.v[0] == pytest.approx(expected_w + b1, rel=1e-7)


def test_resource_load_below_one_is_not_fcfs_stability():
    calc = configured(k=3, rates=[0.55, 0.55], needs=[2, 3])
    assert (0.55 * 2 + 0.55 * 3) / 3 < 1
    assert calc.saturated_throughput() == pytest.approx(1)
    with pytest.raises(ValueError, match="unstable"):
        calc.run()


def test_truncation_convergence_and_mean_conservation():
    coarse = configured(truncation=4).run()
    fine = configured(truncation=10).run()
    assert fine.boundary_mass < coarse.boundary_mass
    assert fine.v[0] > coarse.v[0]
    for result in (coarse, fine):
        mean_n = np.arange(len(result.p)) @ result.p
        assert mean_n == pytest.approx(result.throughput * result.v[0])


def test_configuration_and_invalidation():
    calc = MsjPHCalc(2)
    with pytest.raises(ValueError, match="Both servers"):
        calc.run()
    calc.set_sources([0.4])
    calc.set_servers([2], [PHDistribution.from_exp(1)])
    first = calc.get_w()[0]
    assert calc.get_v()[0] > first
    assert sum(calc.get_p()) == pytest.approx(1)
    calc.set_sources([0.2])
    assert calc.w is None and calc.result is None
    assert calc.get_w()[0] < first
    calc.set_sources([0.1, 0.1])
    with pytest.raises(ValueError, match="counts must match"):
        calc.run()


@pytest.mark.parametrize("rates", [[], [0], [-1], [np.nan], [np.inf], [[0.2]]])
def test_invalid_rates(rates):
    with pytest.raises(ValueError, match="rates"):
        MsjPHCalc(2).set_sources(rates)


@pytest.mark.parametrize(
    "params",
    [
        PHParams(np.array([-0.1, 1.1]), -np.eye(2)),
        PHParams(np.array([0.5]), np.array([[-1.0]])),
        PHParams(np.array([1.0]), np.array([[0.0]])),
        PHParams(np.array([1.0, 0]), np.array([[-1.0, -0.1], [0.0, -1.0]])),
        PHParams(np.array([1.0, 0]), np.array([[-1.0, 2.0], [0.0, -1.0]])),
        PHParams(np.array([1.0, 0]), np.array([[-1.0, 1.0], [1.0, -1.0]])),
        PHParams(np.array([1.0]), np.array([[np.nan]])),
        PHParams(np.array([1.0]), np.array([[-1.0 + 0j]])),
        PHParams(np.array([]), np.zeros((0, 0))),
    ],
)
def test_invalid_ph(params):
    with pytest.raises(ValueError, match="PH"):
        MsjPHCalc(2).set_servers([1], [params])


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_integer_parameters(value):
    for kwargs in ({"k": value}, {"k": 2, "truncation": value}, {"k": 2, "max_states": value}):
        with pytest.raises(ValueError, match="positive integer"):
            MsjPHCalc(**kwargs)


def test_state_limit_and_invalid_needs():
    with pytest.raises(ValueError, match="max_states"):
        configured(max_states=2).run()
    for needs in ([3], [], [0], [True]):
        with pytest.raises(ValueError):
            MsjPHCalc(2).set_servers(needs, [PHDistribution.from_exp(1)])


def test_saturated_fill_merges_permutations():
    """A tiny count-state chain must not enumerate 2**20 draw orderings."""
    calc = configured(k=20, rates=[0.1, 0.1], needs=[1, 1], max_states=100)
    assert calc.saturated_throughput() == pytest.approx(20)
