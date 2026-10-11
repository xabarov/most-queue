"""Independent PK, isolated enumeration, renewal and token-loss arbiters."""

import numpy as np
import pytest

from examples.stage_aware_ttft_oracle import replay_loss
from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import StageAwareTtftCalc


def configured(cap=2, budget=2, queue=4, rate=0.3, law=((1, 1, 0.6), (3, 2, 0.4)), costs=(0.1, 0.02, 0.003), **kwargs):
    calc = StageAwareTtftCalc(cap, budget, queue, **kwargs)
    calc.set_sources(rate)
    calc.set_servers(law, *costs)
    return calc


@pytest.mark.parametrize("budget", [None, 1, 2, 4])
def test_isolated_against_token_sim(budget):
    calc = configured(cap=1, budget=budget)
    full, first = calc.isolated_times()
    for k, (n, m, _) in enumerate(calc.law):
        sim = PrefillDecodeSim(1, budget)
        sim.set_servers(n, m, 0.1, 0.02, 0.003)
        ref = sim.replay([0.0], [n], [m])
        assert full[k] == pytest.approx(ref.v[0])
        assert first[k] == pytest.approx(ref.ttft_mean)


def test_pk_joint_heterogeneity_independent_formula():
    result = configured(cap=1, budget=2, rate=0.5).run()
    # Enumerate full service for (1,1): .123+.126; (3,2): .146+.129+.132+.135.
    service = np.array([0.249, 0.542])
    first = np.array([0.249, 0.407])
    probs = np.array([0.6, 0.4])
    wait = 0.5 * (probs @ service**2) / (2 * (1 - 0.5 * (probs @ service)))
    assert result.w[0] == pytest.approx(wait)
    assert result.ttft_mean == pytest.approx(wait + probs @ first)
    assert result.v[0] == pytest.approx(wait + probs @ service)


@pytest.mark.parametrize("budget", [2, 4, None])
def test_zero_load_and_stationary_clock_reward(budget):
    calc = configured(budget=budget, rate=1e-5)
    result = calc.run()
    full, first = calc.isolated_times()
    assert result.ttft_mean == pytest.approx(np.array([0.6, 0.4]) @ first, rel=1e-4)
    assert result.v[0] == pytest.approx(np.array([0.6, 0.4]) @ full, rel=1e-4)
    assert result.w[0] >= 0
    assert result.stationary_residual < 1e-11
    assert result.blocking_probability < 1e-8


def test_zero_waiting_capacity_regenerative_loss():
    result = configured(queue=0, law=((1, 1, 1.0),), rate=2).run()
    # Exactly one request per busy cycle: service=.249, first=.249, idle=.5.
    assert result.accepted_rate == pytest.approx(1 / (0.5 + 0.249))
    assert result.blocking_probability == pytest.approx(0.249 / (0.5 + 0.249))
    assert result.ttft_mean == pytest.approx(0.249)
    assert result.w == [0]
    assert result.v[0] == pytest.approx(0.249)


@pytest.mark.parametrize("queue", [0, 2, 4])
def test_finite_loss_against_independent_arrival_event_oracle(queue):
    rate, count = 2.0, 30000
    rng = np.random.default_rng(93009)
    arrivals = np.cumsum(rng.exponential(1 / rate, count))
    types = rng.choice(2, count, p=[0.6, 0.4])
    marks = np.array([[1, 1], [3, 2]])[types]
    ref = replay_loss(arrivals, marks, 2, 2, queue, (0.1, 0.02, 0.003), warmup=3000)
    result = configured(queue=queue, rate=rate).run()
    assert result.ttft_mean == pytest.approx(ref["ttft"], rel=0.035)
    assert result.v[0] == pytest.approx(ref["v"], rel=0.035)
    assert result.w[0] == pytest.approx(ref["w"], abs=0.007)
    assert result.blocking_probability == pytest.approx(ref["loss"], abs=0.012)


def test_overload_contract_not_infinite_stability():
    with pytest.raises(ValueError, match="requires"):
        configured(cap=1, rate=100).run()
    result = configured(rate=100, queue=2).run()
    assert result.conservative_load > 1
    assert result.blocking_probability > 0.8
    assert "finite" in result.model


def test_explicit_complexity_guard():
    with pytest.raises(ValueError, match="state budget"):
        configured(max_states=2).run()
    with pytest.raises(ValueError, match="word budget"):
        configured(max_transition_words=2).run()


@pytest.mark.parametrize("kwargs", [{"batch_cap": True}, {"token_budget": 1}, {"queue_capacity": -0.5}])
def test_invalid_constructor(kwargs):
    with pytest.raises(ValueError):
        StageAwareTtftCalc(**kwargs)


@pytest.mark.parametrize("law", [[], [[1.5, 2, 1]], [[1, 2, 0.9]], [[0, 2, 1]], [[1, 2, float("nan")]]])
def test_invalid_law(law):
    with pytest.raises(ValueError):
        configured(law=law)
