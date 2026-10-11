"""Independent work enumeration, CTMC balance and token trace arbiters for L3."""

import numpy as np
import pytest
from scipy.linalg import solve

from most_queue.sim.prefill_decode import PrefillDecodeSim
from most_queue.theory.continuous_batching import OccupancyDependentQueueCalc, PrefillDecodeCalc


def configured(cap=4, budget=16, rate=0.3, n=8, m=4, costs=(0.1, 0.02, 0.003), states=256):
    calc = PrefillDecodeCalc(cap, budget, states)
    calc.set_sources(rate)
    calc.set_servers(n, m, *costs)
    return calc


@pytest.mark.parametrize("n,m,budget", [(8, 4, 16), (17, 3, 8), (5, 100, 4), (80, 2, 16)])
def test_chunks_independent_integer_search(n, m, budget):
    calc = configured(cap=4, budget=budget, n=n, m=m)
    for occupancy in [0, 0.1, 0.9, 1, 1.5, 2, 3, 4]:
        c = 1
        while (occupancy - 1) * (n + m) / (c + m) + n / c > budget:
            c += 1
        assert calc.prefill_chunks(occupancy) == c


@pytest.mark.parametrize("occupancy", [1, 2, 4])
@pytest.mark.parametrize("budget", [None, 4, 8, 16, 10**16])
def test_work_by_explicit_equal_chunk_and_decode_enumeration(occupancy, budget):
    calc = configured(budget=budget, n=12, m=3)
    profile = calc.stage_profile(occupancy)
    c = profile["chunks"]
    prefill = sum(0.02 * 12 / c + 0.003 * j * 12 / c for j in range(1, c + 1))
    decode = sum(0.02 + 0.003 * (12 + k) for k in range(1, 4))
    assert profile["prefill_work"] == pytest.approx(prefill)
    assert profile["decode_work"] == pytest.approx(decode)
    background = 0.1 + (occupancy - 1) * (prefill + decode) / (c + 3)
    assert profile["service"] == pytest.approx((c + 3) * background + prefill + decode)
    assert profile["prefill"] + 3 * profile["itl"] == pytest.approx(profile["service"])


def test_unlimited_and_single_request_chunk_count():
    assert configured(budget=None, n=4000).prefill_chunks(4) == 1
    for n in [1, 16, 17, 32, 33, 100]:
        assert configured(n=n).prefill_chunks(1) == int(np.ceil(n / 16))


def test_stationary_distribution_against_independent_dense_generator():
    cap, n, m, budget, rate = 4, 8, 4, 6, 0.5
    calc = configured(cap=cap, budget=budget, n=n, m=m, rate=rate)
    result = calc.run()
    # Assemble and solve a generator, not a second product formula.
    rates = []
    for i in range(1, cap + 1):
        c = 1
        while (i - 1) * (n + m) / (c + m) + n / c > budget:
            c += 1
        work = sum(0.02 * n / c + 0.003 * j * n / c for j in range(1, c + 1))
        work += sum(0.02 + 0.003 * (n + j) for j in range(1, m + 1))
        rates.append(i / ((c + m) * 0.1 + i * work))
    size = 100
    generator = np.zeros((size, size))
    for i in range(size):
        if i < size - 1:
            generator[i, i + 1] = rate
        if i:
            generator[i, i - 1] = rates[min(i, cap) - 1]
        generator[i, i] = -generator[i].sum()
    matrix = generator.T.copy()
    matrix[-1] = 1
    rhs = np.zeros(size)
    rhs[-1] = 1
    pi = solve(matrix, rhs)
    assert result.p[:size] == pytest.approx(pi, abs=2e-14)
    assert result.jobs_mean == pytest.approx(pi @ np.arange(size), abs=1e-12)
    assert result.w[0] == pytest.approx(pi @ np.maximum(np.arange(size) - cap, 0) / rate, abs=1e-12)
    assert result.stationary_residual < 1e-13


def test_existing_occupancy_core_and_wait_probability():
    calc = configured(rate=0.6)
    result = calc.run()
    old = OccupancyDependentQueueCalc(k=4, queue_truncation=255)
    old.set_sources(0.6)
    old.set_servers(lambda i: 1 / calc.stage_profile(i)["service"])
    assert result.p == pytest.approx(old.run(1).p, abs=1e-13)
    assert result.w[0] == pytest.approx(old.get_w(1)[0], rel=1e-10)
    assert result.waiting_probability == pytest.approx(old.get_p_wait())
    assert result.queue_nonempty_probability == pytest.approx(sum(result.p[5:]))
    assert result.waiting_probability - result.queue_nonempty_probability == pytest.approx(result.p[4])


@pytest.mark.parametrize("rho", [0.001, 0.5, 0.99, 1 - 1e-9])
def test_b1_exact_mm1_and_unrenormalized_prefix(rho):
    calc = configured(cap=1, budget=None, rate=rho, n=1, m=4, costs=(0.1, 0.1, 0.0), states=2)
    # Total deterministic work + five overheads = 1 second; exponential surrogate.
    result = calc.run()
    assert result.service_mean == pytest.approx(1)
    assert result.w[0] == pytest.approx(rho / (1 - rho), rel=1e-7)
    assert result.p == pytest.approx([1 - rho, (1 - rho) * rho])
    assert result.omitted_probability == pytest.approx(rho**2)
    assert sum(result.p) + result.omitted_probability == pytest.approx(1)
    assert result.batch_mean == pytest.approx(rho)
    assert result.v[0] == pytest.approx(1 / (1 - rho), rel=1e-7)


def test_near_saturation_and_large_cap_no_overflow():
    calc = configured(
        cap=256, budget=8192, rate=1, n=4096, m=4096, costs=(0.00668, 0.0000201, 0.0000000552), states=257
    )
    capacity = 256 / calc.stage_profile(256)["service"]
    calc.set_sources(capacity * (1 - 1e-8))
    result = calc.run()
    assert np.isfinite(result.jobs_mean)
    assert result.jobs_mean > 1e6
    assert result.stationary_residual < 1e-12
    assert sum(result.p) + result.omitted_probability == pytest.approx(1)


def test_literal_negative_prefill_not_silently_repaired():
    result = configured(budget=None, rate=1e-5, n=1, m=100, costs=(0.001, 0, 0.01)).run()
    assert result.prefill_mean < 0
    assert not result.metrics_valid
    assert any("negative" in text for text in result.diagnostics)
    assert result.prefill_mean + 100 * result.itl_mean == pytest.approx(result.service_mean)


def test_cache_copy_and_reconfiguration():
    calc = configured()
    initial = calc.run()
    initial.p[0] = -1
    assert calc.run().p[0] >= 0
    calc.set_sources(0.6)
    second = calc.run()
    calc.set_servers(8, 4, 0.2, 0.02, 0.003)
    assert calc.run().capacity != second.capacity
    assert calc.get_w() == calc.run().w
    assert calc.get_v() == calc.run().v


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch_cap=0),
        dict(batch_cap=True),
        dict(batch_cap=1.5),
        dict(batch_cap=4097),
        dict(token_budget=0),
        dict(token_budget=3, batch_cap=4),
        dict(token_budget=np.inf),
        dict(max_output_states=3, batch_cap=4),
    ],
)
def test_invalid_constructor(kwargs):
    with pytest.raises(ValueError):
        PrefillDecodeCalc(**kwargs)


@pytest.mark.parametrize(
    "values",
    [
        (0, 4, 0.1, 0.02, 0.003),
        (8, -1, 0.1, 0.02, 0.003),
        (8, 4, 0, 0.02, 0.003),
        (8, 4, 0.1, -1, 0.003),
        (8, 4, 0.1, 0.02, np.nan),
        (np.inf, 4, 0.1, 0.02, 0.003),
    ],
)
def test_invalid_service(values):
    with pytest.raises(ValueError):
        PrefillDecodeCalc().set_servers(*values)


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf])
def test_invalid_rate(rate):
    with pytest.raises(ValueError):
        configured(rate=rate)


def test_missing_unstable_and_higher_moments():
    with pytest.raises(ValueError):
        PrefillDecodeCalc().run()
    calc = configured()
    with pytest.raises(ValueError):
        calc.get_w(2)
    for i in [-1, 5, np.nan]:
        with pytest.raises(ValueError):
            calc.stage_profile(i)
    calc.set_sources(4 / calc.stage_profile(4)["service"])
    with pytest.raises(ValueError, match="stationary"):
        calc.run()


def test_isolated_integer_chunk_trace_and_work():
    sim = PrefillDecodeSim(batch_cap=1, token_budget=3)
    sim.set_servers(6, 2, 0.1, 0.02, 0.003)
    result = sim.replay([0], [6], [2], record_iterations=True)
    # Prefill chunks: 3 and 3 tokens; then KV lengths 7 and 8.
    durations = [0.1 + 0.02 * 3 + 0.003 * 3, 0.1 + 0.02 * 3 + 0.003 * 6, 0.1 + 0.02 + 0.003 * 7, 0.1 + 0.02 + 0.003 * 8]
    assert result.completion_times == pytest.approx([sum(durations)])
    assert result.prefill_end_times == pytest.approx([sum(durations[:2])])
    assert result.ttft_mean == pytest.approx(sum(durations[:3]))
    assert result.itl_mean == pytest.approx(durations[-1])
    assert [r["tokens"] for r in result.iteration_trace] == [3, 3, 1, 1]
    assert result.p is None  # no time window for a single arrival


@pytest.mark.parametrize("budget", [None, 2, 3, 7])
def test_individual_lengths_single_slot_vs_lindley_recursion(budget):
    rng = np.random.default_rng(1905)
    arrivals = np.cumsum(rng.exponential(0.8, 200))
    inputs = rng.integers(1, 9, 200)
    outputs = rng.integers(1, 5, 200)
    sim = PrefillDecodeSim(batch_cap=1, token_budget=budget)
    sim.set_servers(1, 1, 0.1, 0.02, 0.003)
    result = sim.replay(arrivals, inputs, outputs)
    finish = 0.0
    starts, finishes, first_tokens = [], [], []
    for arrival, n, m in zip(arrivals, inputs, outputs):
        start = max(arrival, finish)
        consumed, pref = 0, 0.0
        while consumed < n:
            chunk = min(n - consumed, budget if budget is not None else n)
            consumed += chunk
            pref += 0.1 + 0.02 * chunk + 0.003 * consumed
        decode = [0.1 + 0.02 + 0.003 * (n + k) for k in range(1, m + 1)]
        finish = start + pref + sum(decode)
        starts.append(start)
        finishes.append(finish)
        first_tokens.append(start + pref + decode[0])
    assert result.start_times == pytest.approx(starts, abs=1e-10)
    assert result.completion_times == pytest.approx(finishes, abs=1e-10)
    assert [row[0] for row in result.output_times] == pytest.approx(first_tokens, abs=1e-10)


def test_simulator_first_token_only_has_no_intertoken_gap():
    sim = PrefillDecodeSim(batch_cap=1)
    sim.set_servers(2, 1, 0.1, 0.02, 0.003)
    result = sim.replay([0, 10], [2, 2], [1, 1])
    assert result.itl_mean is None
    assert result.all_decode_mean == pytest.approx(0.1 + 0.02 + 0.003 * 3)


def test_input_time_and_cost_units_scale_together():
    first = configured(rate=0.3).run()
    second = configured(rate=0.003, costs=(10, 2, 0.3)).run()
    assert second.w[0] == pytest.approx(100 * first.w[0])
    assert second.ttft_mean == pytest.approx(100 * first.ttft_mean)
    assert second.p == pytest.approx(first.p)


def test_boundary_admission_and_decode_first_mixed_trace():
    sim = PrefillDecodeSim(batch_cap=2, token_budget=3)
    sim.set_servers(3, 2, 0.1, 0.02, 0.003)
    result = sim.replay([0, 0.05], [3, 2], [2, 1], record_iterations=True)
    # First arrival runs prefill alone until .169; second cannot join mid-iteration.
    assert result.start_times == pytest.approx([0, 0.169])
    assert result.iteration_trace[1]["actions"] == [(0, "decode", 1), (1, "prefill", 2)]
    assert result.iteration_trace[2]["actions"] == [(0, "decode", 1), (1, "decode", 1)]
    assert result.completion_times == pytest.approx([0.511, 0.511])
    assert result.w[0] == pytest.approx((0.169 - 0.05) / 2)
    assert result.p == pytest.approx([0, 1])  # observation ends at .05 before second arrival


def test_stalled_prefill_fcfs_and_cap_and_arrival_ties():
    sim = PrefillDecodeSim(batch_cap=3, token_budget=3)
    sim.set_servers(4, 2, 0.1, 0.02, 0.003)
    result = sim.replay([0, 0, 0, 0], [4] * 4, [2] * 4, record_iterations=True)
    assert result.iteration_trace[0]["actions"] == [(0, "prefill", 3)]
    assert result.iteration_trace[1]["actions"] == [(0, "prefill", 1), (1, "prefill", 2)]
    assert all(row["tokens"] <= 3 for row in result.iteration_trace)
    assert all(len(row["active"]) <= 3 for row in result.iteration_trace)
    assert result.start_times[3] == result.completion_times[0]
    assert all(len(row) == 2 for row in result.output_times)
    assert np.isfinite(result.v).all()


def test_reproducibility_and_individual_sampler():
    sim = PrefillDecodeSim(batch_cap=4, token_budget=8, seed=12)
    sim.set_sources(0.4)
    sim.set_servers(
        8, 4, 0.1, 0.02, 0.003, length_sampler=lambda rng: (int(rng.integers(2, 8)), int(rng.integers(1, 5)))
    )
    first, second = sim.run(1000), sim.run(1000)
    assert first.v == second.v
    assert first.output_times == second.output_times
    assert sum(first.p) == pytest.approx(1)
    assert len(set(map(len, first.output_times))) == 4


@pytest.mark.parametrize(
    "arrivals,inputs,outputs",
    [
        ([-1], [1], [1]),
        ([1, 0], [1, 1], [1, 1]),
        ([np.nan], [1], [1]),
        ([], [], []),
        ([0], [0], [1]),
        ([0], [1.5], [1]),
        ([0], [1], [np.inf]),
        ([0, 1], [1], [1, 1]),
    ],
)
def test_invalid_trace(arrivals, inputs, outputs):
    sim = PrefillDecodeSim(batch_cap=1)
    sim.set_servers(1, 1, 0.1, 0.01, 0.001)
    with pytest.raises(ValueError):
        sim.replay(arrivals, inputs, outputs)


def test_sim_budget_and_configuration_errors():
    with pytest.raises(ValueError):
        PrefillDecodeSim(batch_cap=3, token_budget=2)
    with pytest.raises(ValueError):
        PrefillDecodeSim().run(10)
    sim = PrefillDecodeSim(batch_cap=1, max_iterations=1)
    sim.set_servers(1, 2, 0.1, 0, 0)
    with pytest.raises(RuntimeError, match="max_iterations"):
        sim.replay([0], [1], [2])
    with pytest.raises(ValueError):
        sim.replay([0], [1], [2], warmup_fraction=1)
