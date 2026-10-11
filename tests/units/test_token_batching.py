"""Independent enumerations, M/G/1 limits, and literal-paper diagnostics."""

import itertools

import numpy as np
import pytest
from scipy.stats import nbinom, poisson

from most_queue.sim.dynamic_batching import DynamicBatchingSim
from most_queue.sim.token_batching import TokenBatchingSim
from most_queue.structs import QueueResults
from most_queue.theory.batch.callback_batching import CallbackBatchingCalc
from most_queue.theory.batch.dynamic_batching import DynamicBatchingCalc
from most_queue.theory.batch.token_batching import (
    FixedTokenBatchingCalc,
    TokenBatchingCalc,
    TokenClippingCalc,
    optimize_token_limit,
    paper_fixed_formula,
)
from most_queue.theory.batch.token_service import TokenBatchService, TokenLengthDistribution

TOKENS = TokenLengthDistribution([1, 3, 9], [0.2, 0.6, 0.2])
COEFFICIENTS = (0.1, 0.2, 0.025, 0.04)


@pytest.mark.parametrize("elastic", [False, True])
@pytest.mark.parametrize("size", [1, 2, 3, 5])
def test_service_against_exhaustive_individual_requests(elastic, size):
    law = TokenBatchService(TOKENS, *COEFFICIENTS, elastic=elastic)
    sim = TokenBatchingSim("elastic" if elastic else "dynamic").set_servers(TOKENS, *COEFFICIENTS)
    durations, rewards, weights = [], [], []
    for indices in itertools.product(range(3), repeat=size):
        offsets = sim.completion_offsets(TOKENS.lengths[list(indices)])
        durations.append(offsets.max())
        rewards.append(offsets.mean())
        weights.append(np.prod(TOKENS.probabilities[list(indices)]))
    h, r, p = np.asarray(durations), np.asarray(rewards), np.asarray(weights)
    assert law.moments(size) == pytest.approx([p @ h, p @ h**2], rel=2e-13)
    assert law.job_mean(size) == pytest.approx(p @ r, rel=2e-13)
    count = law.counts(size, 1.3)[0][1]
    k = np.arange(15)
    direct = p @ poisson.pmf(k[None, :], 1.3 * h[:, None])
    assert count.pmf(k) == pytest.approx(direct, abs=2e-14)
    assert count.mean() == pytest.approx(1.3 * (p @ h), rel=2e-12)


def test_clipping_is_censoring_not_conditional_truncation():
    tokens = TokenLengthDistribution([9, 1, 3, 3], [0.2, 0.2, 0.3, 0.3], max_tokens=4)
    assert tokens.lengths.tolist() == [1, 3, 4]
    assert tokens.probabilities == pytest.approx([0.2, 0.6, 0.2])
    assert tokens.clipped_fraction == pytest.approx(0.2)
    assert tokens.utility == pytest.approx(0.8 + 0.2 * 4 / 9)
    assert tokens.maximum_pmf(2) == pytest.approx([0.04, 0.60, 0.36])
    with pytest.raises(ValueError):
        tokens.lengths[0] = 7


def test_count_quantile_does_not_collapse_at_rounded_endpoint():
    service = TokenBatchService(TOKENS, *COEFFICIENTS, elastic=True)
    for size in (23, 64, 128):
        count = service.counts(size, 3)[0][1]
        high = int(count.isf(1e-15))
        assert high > count.mean()
        assert count.sf(high) <= 1e-15
        assert count.cdf(high) + count.sf(high) == pytest.approx(1, abs=3e-14)


@pytest.mark.parametrize("factory,args", [("clipped_lognormal", (3, 0.7)), ("clipped_normal", (20, 4))])
def test_continuous_cdf_factories_keep_cap_atom(factory, args):
    tokens = getattr(TokenLengthDistribution, factory)(*args, 20)
    assert tokens.probabilities.sum() == pytest.approx(1)
    assert tokens.lengths[-1] == 20
    assert tokens.probabilities[-1] >= tokens.clipped_fraction
    assert np.isnan(tokens.utility)  # utility cannot be reconstructed from censored data


@pytest.mark.parametrize("elastic", [False, True])
def test_single_token_length_reduces_to_l1(elastic):
    tokens = TokenLengthDistribution([7], [1])
    c = TokenBatchingCalc(tolerance=1e-8).set_sources(1).set_servers(tokens, *COEFFICIENTS, elastic=elastic)
    a, tau = 0.1 + 0.025 * 6, 0.2 + 0.04 * 6
    reference = DynamicBatchingCalc(tolerance=1e-8).set_sources(1).set_deterministic_service(a, tau).run()
    result = c.run()
    assert isinstance(result, QueueResults)
    assert result.w == pytest.approx(reference.w, rel=2e-10)
    assert result.v == pytest.approx(reference.v, rel=2e-10)
    assert result.utilization == pytest.approx(reference.utilization, rel=2e-10)


@pytest.mark.parametrize("elastic", [False, True])
def test_fixed_one_is_mg1_not_md1(elastic):
    c = FixedTokenBatchingCalc(1).set_sources(1).set_servers(TOKENS, *COEFFICIENTS, elastic=elastic)
    result = c.run()
    durations = 0.1 + 0.2 + (0.025 + 0.04) * (TOKENS.lengths - 1)
    mean, second = TOKENS.probabilities @ durations, TOKENS.probabilities @ durations**2
    wait = second / (2 * (1 - mean))
    assert result.w == pytest.approx([wait], rel=1e-11)
    assert result.v == pytest.approx([wait + mean], rel=1e-11)
    assert result.utilization == pytest.approx(mean)
    assert result.converged
    assert result.p is None and result.pi is None
    result.queue_distribution[:] = 0
    assert c.run().queue_distribution.sum() == pytest.approx(1)


@pytest.mark.parametrize("size", [1, 2, 3, 8])
def test_paper_fixed_expression_omits_formation_and_includes_service(size):
    c = FixedTokenBatchingCalc(size).set_sources(1).set_servers(TOKENS, *COEFFICIENTS, mean_service_surrogate=True)
    result = c.run()
    h = c.service.moments(size)[0]
    raw = paper_fixed_formula(1, h, size, terms=1000)
    assert raw == pytest.approx(result.v[0] - (size - 1) / 2, rel=1e-11)
    if size == 1:
        assert raw - result.w[0] == pytest.approx(h)


def test_surrogate_underestimates_fixed_one_waiting_with_nonzero_variance():
    actual = FixedTokenBatchingCalc(1).set_sources(1).set_servers(TOKENS, *COEFFICIENTS).run()
    surrogate = (
        FixedTokenBatchingCalc(1).set_sources(1).set_servers(TOKENS, *COEFFICIENTS, mean_service_surrogate=True).run()
    )
    assert actual.w[0] > surrogate.w[0]


def test_clipping_pk_and_finite_candidate_optimizer():
    tokens = TokenLengthDistribution([1, 10, 100], [0.7, 0.25, 0.05])
    report = optimize_token_limit(tokens, [1, 5, 10, 20, 100], rate=1, a=0.1, theta=0.95)
    assert report["best"] == max((r for r in report["candidates"] if r["stable"]), key=lambda r: r["objective"])
    original = TokenClippingCalc().set_sources(1).set_servers(tokens, 0.1).run()
    clipped = (
        TokenClippingCalc()
        .set_sources(1)
        .set_servers(TokenLengthDistribution(tokens.lengths, tokens.probabilities, 10), 0.1)
        .run()
    )
    assert clipped.w[0] < original.w[0]
    with pytest.raises(ValueError, match="original"):
        optimize_token_limit(TokenLengthDistribution([1, 9], [0.5, 0.5], 3), [1, 3], 1, 0.1)


def test_generic_nonaffine_callback_and_exponential_reduction():
    h = lambda b: 0.1 * b + 0.2 + 0.03 * np.sqrt(b)
    c = CallbackBatchingCalc(tolerance=1e-8).set_sources(1)
    c.set_servers(
        lambda b: (h(b), h(b) ** 2),
        lambda b, rate: [(1, poisson(rate * h(b)))],
        mean_bound=(0.13, 0.2),
        second_bound=(0.13**2, 2 * 0.13 * 0.2, 0.2**2),
    )
    assert c.run().converged
    c.set_sources(0.8)
    assert c.run().converged
    c.set_servers(
        lambda b: (0.1 * b + 0.2, 2 * (0.1 * b + 0.2) ** 2),
        lambda b, rate: [(1, nbinom(1, 1 / (1 + rate * (0.1 * b + 0.2))))],
        mean_bound=(0.1, 0.2),
        second_bound=(0.02, 0.08, 0.08),
    )
    expected = DynamicBatchingCalc(tolerance=1e-8).set_sources(0.8).set_exponential_service(0.1, 0.2).run()
    assert c.run().w == pytest.approx(expected.w, rel=1e-11)
    sim = DynamicBatchingSim(seed=4).set_sources(1).set_service_callback(lambda b, rng: h(b), mean_bound=(0.13, 0.2))
    assert sim.run(100).served >= 100


def test_certificate_cannot_be_bypassed_with_fixed_truncation():
    c = CallbackBatchingCalc(max_batch=8).set_sources(1)
    c.set_servers(
        lambda b: (float(b), float(b * b)),
        lambda b, rate: [(1, poisson(rate * b))],
        mean_bound=(1, 0),
        second_bound=(1, 0, 0),
    )
    with pytest.raises(ValueError, match="certificate"):
        c.run()


def test_invalid_callback_consistency_and_budget():
    c = CallbackBatchingCalc().set_sources(1)
    c.set_servers(lambda b: (1, 1), lambda b, rate: [(1, poisson(2))], mean_bound=(0, 1), second_bound=(0, 0, 1))
    with pytest.raises(ValueError, match="inconsistent"):
        c.run()
    c.set_servers(lambda b: (2, 1), lambda b, rate: [(1, poisson(2))], mean_bound=(0, 2), second_bound=(0, 0, 4))
    with pytest.raises(ValueError, match="moments"):
        c.run()
    with pytest.raises(MemoryError):
        TokenBatchService(TOKENS, *COEFFICIENTS, max_fft_points=16).counts(1000, 1)
    with pytest.raises(MemoryError):
        FixedTokenBatchingCalc(2, max_entries=10).set_sources(1).set_servers(TOKENS, *COEFFICIENTS).run()


@pytest.mark.parametrize("elastic", [False, True])
def test_pathwise_envelope_bounds_random_service(elastic):
    c = TokenBatchingCalc(tolerance=1e-8).set_sources(2).set_servers(TOKENS, *COEFFICIENTS, elastic=elastic)
    report = c.paper_bound_report()
    assert report["phi1_certified"]
    assert report["waiting_mean"] <= report["paper_min_substitution"]


@pytest.mark.parametrize(
    "lengths,weights", [([], []), ([0], [1]), ([1.5], [1]), ([1], [0.5]), ([1, 2], [1.1, -0.1]), ([np.inf], [1])]
)
def test_invalid_token_pmfs(lengths, weights):
    with pytest.raises(ValueError):
        TokenLengthDistribution(lengths, weights)


@pytest.mark.parametrize("strategy,cap", [("bad", None), ("fixed", None), ("dynamic", True), ("elastic", 0)])
def test_invalid_sim_configuration(strategy, cap):
    with pytest.raises(ValueError):
        TokenBatchingSim(strategy, cap)


def test_simulator_early_release_trace_and_reproducibility():
    s = TokenBatchingSim("elastic", seed=3).set_sources(1).set_servers(TOKENS, *COEFFICIENTS)
    # pre=.5; 2 decode iterations with 2 active, then 6 with 1 active.
    assert s.completion_offsets([1, 3, 9]) == pytest.approx([0.5, 0.68, 1.07])
    a = s.run(100)
    b = TokenBatchingSim("elastic", seed=3).set_sources(1).set_servers(TOKENS, *COEFFICIENTS).run(100)
    assert a.waits == pytest.approx(b.waits)
    assert a.latencies == pytest.approx(b.latencies)
    assert sum(a.p) == pytest.approx(1)
    assert a.utilization == pytest.approx(1 - a.p[0])
    assert a.warmup_served >= 25
    assert a.cdf(0, "w") >= 0


@pytest.mark.parametrize(
    "constructor", [lambda: TokenBatchingCalc(max_batch=16), lambda: FixedTokenBatchingCalc(3, queue_truncation=16)]
)
def test_pinned_truncations_report_nonconvergence_and_reject_higher_moments(constructor):
    c = constructor().set_sources(1).set_servers(TOKENS, *COEFFICIENTS)
    assert not c.run().converged
    with pytest.raises(ValueError):
        c.get_w(2)
    with pytest.raises(ValueError):
        c.get_v(2)


@pytest.mark.parametrize("strategy", ["dynamic", "fixed", "elastic"])
def test_sim_domain_and_run_inputs(strategy):
    s = TokenBatchingSim(strategy, 2 if strategy == "fixed" else None, seed=3)
    with pytest.raises(ValueError):
        s.run(1)
    s.set_sources(1).set_servers(TOKENS, *COEFFICIENTS)
    for jobs in (0, True, 1.5):
        with pytest.raises(ValueError):
            s.run(jobs)
    for fraction in (-0.1, 1, np.nan):
        with pytest.raises(ValueError):
            s.run(1, fraction)


@pytest.mark.parametrize("cap", [1, 4, 8])
def test_sim_physical_cap_keeps_excess_requests(cap):
    result = TokenBatchingSim("dynamic", cap, seed=7).set_sources(2).set_servers(TOKENS, *COEFFICIENTS).run(1000)
    assert result.batch_mean <= cap
    assert result.served >= 1000
    assert result.served / result.observation_time == pytest.approx(2, rel=0.15)


@pytest.mark.parametrize(
    "method,args",
    [
        ("uniform", (True,)),
        ("uniform", (100001,)),
        ("clipped_lognormal", (3, 0, 10)),
        ("clipped_lognormal", (1000, 0.7, 10)),
        ("clipped_normal", (1, 0, 10)),
        ("from_cdf", (lambda x: np.full_like(x, 2.0), 10)),
    ],
)
def test_token_factory_input_guards(method, args):
    with pytest.raises((ValueError, MemoryError)):
        getattr(TokenLengthDistribution, method)(*args)


def test_fixed_and_single_request_stability_guards():
    with pytest.raises(ValueError, match="requires"):
        FixedTokenBatchingCalc(1).set_sources(10).set_servers(TOKENS, *COEFFICIENTS).run()
    with pytest.raises(ValueError, match="requires"):
        TokenClippingCalc().set_sources(10).set_servers(TOKENS, 0.1).run()
    with pytest.raises(ValueError, match="surrogate"):
        FixedTokenBatchingCalc(1).set_servers(TOKENS, *COEFFICIENTS, elastic=True, mean_service_surrogate=True)
    with pytest.raises(ValueError, match="no stable"):
        optimize_token_limit(TOKENS, [1, 2], 100, 0.1)


@pytest.mark.parametrize("failure", ["envelope", "job", "weight", "second_certificate"])
def test_generic_invalid_law_contracts(failure):
    c = CallbackBatchingCalc(max_batch=8).set_sources(1)
    if failure == "envelope":
        c.set_servers(lambda b: (2, 4), lambda b, r: [(1, poisson(2 * r))], mean_bound=(0, 1), second_bound=(0, 0, 1))
    elif failure == "job":
        c.set_servers(
            lambda b: (1, 1),
            lambda b, r: [(1, poisson(r))],
            mean_bound=(0, 1),
            second_bound=(0, 0, 1),
            job_mean=lambda b: 2,
        )
    elif failure == "weight":
        c.set_servers(lambda b: (1, 1), lambda b, r: [(0.5, poisson(r))], mean_bound=(0, 1), second_bound=(0, 0, 1))
    else:
        c.set_servers(
            lambda b: (0.1 * b, 0.01 * b * b),
            lambda b, r: [(1, poisson(0.1 * r * b))],
            mean_bound=(0.1, 0),
            second_bound=(1, 0, 0),
        )
    with pytest.raises(ValueError):
        c.run()
