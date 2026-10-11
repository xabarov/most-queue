# Mixed prefill and decode iteration costs

The three-parameter model of [Ramani and Tantawi](https://arxiv.org/html/2609.20957v1)
predicts mean first-token and inter-token latency from prompt length, output
length, arrival rate and accelerator costs. The occupancy calculation is exact
for its Markovian surrogate; the surrogate and TTFT/ITL are approximations to
actual token-level execution.

[Русская версия](prefill-decode.ru.md) · [Model catalog](../models.md)

For an explicit attained-token state reference, see the
[stage-aware mean TTFT calculator](stage-aware-ttft.md). It does not change the
literal L3 implementation: B=1 uses exact PK; B>1 offers finite loss or the
separate infinite calculator with analytic truncation bounds under a sufficient
stability condition (floating evaluation, not verified rounding).

## In plain words

Requests share an active pool of at most B slots. Each first processes its
prompt, then emits output tokens one per iteration; finishing requests leave
individually and free their slots for the FCFS queue. A long prefill can slow
tokens for existing decoders. Chunking limits the tokens processed in a single
iteration, but does not eliminate that interference.

![Mixed prefill and decode iterations](../figures/prefill_decode.png)

## Analytical model

`PrefillDecodeCalc` inherits `BaseQueue`. `set_sources(l)` takes a positive
finite Poisson rate. `set_servers(input_tokens, output_tokens, alpha, beta,
gamma)` takes positive mean lengths, positive baseline iteration time alpha,
and nonnegative per-token compute/KV-access times beta/gamma. All costs use
the same time unit and l its inverse.

`batch_cap` is an integer in 1..4096. `token_budget` is None (unlimited tokens)
or an integer at least `batch_cap`. A finite budget determines the effective
prefill count `c(i)` from Appendix A, not the runtime scheduler's actual chunks.
Fractional mean lengths are allowed. The arrival queue is infinite.

```python
from most_queue.theory.continuous_batching import PrefillDecodeCalc

calc = PrefillDecodeCalc(batch_cap=256, token_budget=8192)
calc.set_sources(l=2.0)  # requests/second
calc.set_servers(input_tokens=1024, output_tokens=256,
                 alpha=0.00668, beta=0.0000201, gamma=0.0000000552)
result = calc.run()
print(result.ttft_mean, result.itl_mean, result.w, result.v)
print(result.metrics_valid, result.diagnostics)
```

These costs are Table I's Llama-3.1-8B/H100 fit, **converted from ms to s**.
They are published estimates, not fresh calibration of your accelerator.

## Outputs and exactness

| Output | Meaning |
|---|---|
| `w=[mean]`, `v=[mean]` | Exact waiting/sojourn means of the surrogate |
| `ttft_mean`, `itl_mean`, `prefill_mean` | Literal paper mean-occupancy approximation, potentially nonphysical |
| `batch_mean`, `jobs_mean`, `queue_mean` | Time-stationary means in the surrogate |
| `waiting_probability` | PASTA probability of waiting, P(N >= B) |
| `queue_nonempty_probability` | P(N > B), the paper's `P_w` |
| `utilization` | Saturation load rho = lambda*tau(B)/B, **not** busy fraction |
| `busy_probability` | 1−p[0] |
| `p`, `omitted_probability`, `tail_ratio` | Unrenormalized PMF prefix, omitted mass, geometric tail ratio |
| `metrics_valid`, `diagnostics` | Nonnegative estimates and extrapolation warnings, not an accuracy certificate |

Stability requires rho < 1. Every mean includes the entire geometric tail.
`max_output_states` (integer B+1..1000000) limits only the displayed PMF: increasing it does not
change means. The last PMF entry is an individual state, not a folded tail.
`pi` is unset; `p` already describes time/PASTA occupancy. Only first latency
moments are supported; no analytic TTFT/ITL CDF or tail is exposed.

The final prefill formula substitutes **unconditional** mean batch occupancy
for a tagged request's occupancy. Below one, its background population is
negative. Such extrapolation is diagnosed; negative latencies are preserved
with `metrics_valid=False`, never silently clipped or relabelled as exact.

## Independent simulator

`most_queue.sim.prefill_decode.PrefillDecodeSim` follows individual integer
tokens, without calling the calculator or sampling its exponential completion
rates. At a boundary it admits FCFS up to B, schedules one token for each
decoder, then spends the remaining token budget on prefills in FCFS order.
New arrivals cannot join an iteration already running. Prefill completion does
not also decode in that iteration. Iteration time is alpha plus the actual
compute tokens and per-request KV reads/writes. This is a specified policy,
not a copy of vLLM or Sarathi-Serve.

```python
from most_queue.sim.prefill_decode import PrefillDecodeSim

sim = PrefillDecodeSim(batch_cap=4, token_budget=8, seed=7)
sim.set_sources(l=0.5)
sim.set_servers(8, 4, 0.1, 0.02, 0.003)  # synthetic seconds
observed = sim.run(num_of_jobs=20_000, warmup_fraction=0.2)
print(observed.ttft_mean, observed.itl_mean, observed.all_decode_mean)

# Deterministic trace with individual lengths and optional iteration records.
trace = sim.replay([0, 0.05], [8, 3], [4, 2], record_iterations=True)
print(trace.output_times, trace.iteration_trace[:2])
```

`length_sampler(rng)->(input_length, output_length)` is an optional keyword to
`set_servers`; sampled lengths must be integers in 1..1000000. Full timestamp
arrays include warmup jobs; delay statistics exclude their arrival-index
cohort. Time occupancy `p`, busy fraction and mean jobs/pool size cover the
interval from the first measured arrival to the last arrival, excluding drain.
There is no time PMF for a zero-length observation window.

DES returns empirical raw W/V moments 1–3. `itl_mean` averages **observed
gaps after the first output token**; it is None if every output has one token.
`first_decode_mean` measures prefill-to-first-token time. `all_decode_mean`
also includes those first decode gaps, so it matches the paper's m-step
accounting convention more closely. With varying lengths these averages are
token-weighted, not averages of per-request averages.

`run` processes a finite cohort then drains it; completing an overloaded trace
does not prove stationary stability. `max_iterations` is an explicit runtime
guard, not a queue capacity or a stability criterion.

## Limits and reproduction

For verified saturated drift and phase-aware infinite means see
[the stage-aware extension](stage-aware-ttft.md#phase-aware-infinite-queue-epic-096).
Exact phase drift does not verify all mean-bound arithmetic.

At B=1 with constant lengths the DES is M/D/1, whereas the surrogate is M/M/1:
its mean wait is twice the M/D/1 wait at equal service mean. Model differences
must not be absorbed into a loose test tolerance. Unequal final prefill chunks,
budget stalls, iteration-boundary admission and length variation can introduce
additional differences.

No KV-memory ceiling/eviction, prefix caching, speculative decoding, network
delay, tuner or autoscaling actuator is implemented in this family.
[Occupancy-dependent queues](continuous-batching.md) remain the generic
Markovian sibling; [variable-token batching](token-batching.md) is gated and
does not admit new requests between decode iterations.

See [method and validation report](../research/prefill-decode-validation-2026.md),
[experiment JSON](../research/prefill-decode-results-2026.json),
[experiment script](../../examples/prefill_decode_experiment.py) and
[EPIC-086](../epics/EPIC-086-prefill-decode-iteration-cost.md).
