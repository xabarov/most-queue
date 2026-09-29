# Bulk-service waiting-time moments (LLM/GPU dynamic batching) — literature review (2026)

Direct continuation of EPIC-012 (`docs/epics/EPIC-012-bulk-service.md`, `theory/batch/bulk_service.py`):
that epic already built an exact, batch-size-dependent-rate `M/M^[a,b]/1` CTMC calculator
(`BulkServiceMM1Calc`) — but only reports the **mean** waiting/sojourn time, via a coarse estimate
(`mean_batch_service = 1/mu(min(b, max(a,1)))`, a *fixed*-batch-size approximation of `E[S]` used
to back out `E[W] = E[V] - E[S]`). This is inaccurate whenever the batch size varies (any `a<b`).

## What this epic found (not just literature — verified numerically before scoping)

The full stationary distribution `pi(i,j)` (`i` = batch size in service, `j` = number waiting) was
already being solved exactly by the existing CTMC — computing moments of `N` (number in system)
from it is a trivial extension (already-known math). The actual gap is **waiting-time moments**,
which requires a genuinely different argument (PASTA + the waiting customer's own position in the
batch-formation process), not just more summation over the same `pi`.

**Derivation (validated against DES before committing to it):** by PASTA, an arriving customer
sees the stationary `(i,j)`. If `i>=1` (server busy), the customer's wait is the remaining current
batch's service time (`Exp(mu(i))`, memoryless) plus the time for `j // b` *full* batches of size
`b` ahead of them to clear (each `Exp(mu(b))`, independent) — a hypoexponential distribution, whose
raw moments are computable exactly via `conv_moments` (repeated convolution of the individual
exponentials' own moments), no new numerical machinery needed. This was verified against a from-scratch
Monte Carlo bulk-service simulation and matched to <0.5% on the first four raw moments.

**Caveat found and the reason this epic is scoped to `a=1`:** the derivation above silently
assumes a batch of exactly `min(b, remaining)` always forms as soon as the previous one clears.
That is only true when `a=1` — for `a>1`, if the remaining count after some full batches is `<a`,
the server goes idle and waits for **more arrivals** to refill the threshold (exactly the same
"idle-accumulation" sub-problem as the very first batch), which reintroduces a state-dependent,
recursive structure that breaks the clean hypoexponential decomposition. Numerically confirmed:
for `a=1` the derivation matches Monte Carlo to <0.5%; for `a=2` it was off by ~18% on the mean
(0.427 predicted vs 0.520 Monte Carlo) — a genuine, caught-before-shipping error, not a documentation
nuance. `a=1` (no minimum batch threshold — the server always processes whatever is queued as soon
as it's free, up to a maximum `b`) is also the practically dominant case in the GPU/LLM dynamic-batching
literature (see below): there is generally no reason to *delay* starting a smaller batch when the
GPU is otherwise idle.

## Literature

- Cheng B., Zeng Y., et al., **Queueing analysis of GPU-based inference servers with dynamic
  batching: A closed-form characterization**, Performance Evaluation, 2020,
  doi:10.1016/j.peva.2020.102183 (23 cites) — the direct GPU-batching motivation; a closed-form
  mean-delay characterization for exactly the `a=1` (no-minimum-threshold) dynamic-batching regime
  this epic targets.
- **Markovian bulk-arrival and bulk-service queues with general state-dependent control**, Queueing
  Systems, 2020, doi:10.1007/s11134-020-09660-0 (15 cites) — modern general-state-dependent-rate
  treatment in the flagship journal, confirms the CTMC approach (already used by
  `BulkServiceMM1Calc`) is the standard exact method for this model family.
  *Tail probabilities of the delay in a batch-service queueing model with batch-size dependent
  service time*, Computers & Operations Research, 2012, doi:10.1016/j.cor.2012.10.009 (39 cites) —
  the waiting-time-*distribution* (not just mean) question this epic addresses, for the classical
  (non-LLM) batch-size-dependent-service model.
- **A bulk service GI/M/1 queue with service rates depending on service batch size**, Journal of
  the Operations Research Society of Japan, 1996, doi:10.15807/jorsj.39.25 — classical foundation
  for batch-size-dependent rates.
- 2026 follow-ons confirming continued activity: *Optimizing Server Allocation for Multi-Tenant GPU
  Inference With Dynamic Batching: A Queueing Theoretic...*, IEEE Trans. Cloud Computing, 2026,
  doi:10.1109/tcc.2026.3696613; *Gating Beats Batching: Closed-form Cost, Latency, and Concurrency
  Bounds for LLM Calls...*, SSRN, 2026, doi:10.2139/ssrn.7508098.

## Scope decision

1. **Exact raw moments of `N`** (number in system) directly from the already-solved `pi` — trivial,
   zero risk, any `a`, `b`.
2. **Exact raw moments of `W`** (waiting time), restricted to **`a=1`** — hypoexponential
   decomposition via PASTA (see derivation above), reusing `conv_moments`. For `a>1`, `run()` keeps
   the existing mean-only (approximate) behavior unchanged, with the inaccuracy now documented
   rather than silent.
3. **Reserve, not in this epic:** exact `W` moments for `a>1` (needs an augmented
   absorbing-CTMC/first-passage argument tracking "customers still ahead of a tagged arrival" through
   idle-refill sub-phases — a real, solvable, but more involved extension); exact moments of `V`
   (sojourn time) even at `a=1` (the tagged customer's own eventual batch size depends on arrivals
   during their own wait, correlated with `W` itself, so `V ≠ W + S` by simple convolution — needs a
   joint argument, not attempted here).

Full derivation: `docs/roadmaps/bulk_service_waiting_moments_roadmap.md`.
