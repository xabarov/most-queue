# Bulk-service with general (Erlang-fitted) batch service time — literature review (2026)

Closes a reserve item flagged explicitly since EPIC-012: `BulkServiceMM1Calc`
(`theory/batch/bulk_service.py`, EPIC-012/032) is an exact CTMC for `M/M^[a,b]/1` — batch-size
threshold `(a,b)`, but the batch **service time itself is exponential** (possibly batch-size
dependent rate). This epic replaces the exponential assumption with a **general** distribution,
fitted to an Erlang(k, rate) phase-type representation from its raw moments — the same
moment-fitting convention already used throughout the library (`MaxDistribution`, the SLA layer,
`SplitJoinCalc`'s non-Pareto branch).

## Literature

- Neuts M.F., **Waiting Time Distribution in a Poisson Queue with a General Bulk Service Rule**,
  Management Science, 1975, doi:10.1287/mnsc.21.7.777 (64 cites) — the classical foundation:
  general `(a,b)` bulk-service rule combined with a general batch-service-time distribution,
  solved via an embedded Markov chain at departure epochs and generating-function inversion.
- Chaudhry M.L., Templeton J.G.C., *A First Course in Bulk Queues*, Wiley, 1983 — the standard
  textbook treatment of the embedded-chain/PGF method this literature builds on.
- Recent activity confirming the sub-area stays alive: *Analysis of finite-buffer
  state-dependent bulk queues*, OR Spectrum, 2012, doi:10.1007/s00291-012-0282-7; *A Novel
  Computational Procedure for the Waiting-Time Distribution (In the Queue) for Bulk-Service
  Finite...*, Mathematics, 2023, doi:10.3390/math11051142; *Performance analysis of a versatile
  bulk-service queue with group-arrival, batch-size-dependent service...*, Quality Technology &
  Quantitative Management, 2024, doi:10.1080/16843703.2024.2391673.

## Scope decision: phase-type (Erlang) augmentation, not the classical embedded chain

The classical route (Neuts 1975, Chaudhry-Templeton) works at *departure epochs* via an embedded
Markov chain and PGF inversion — converting its embedded-chain stationary distribution into
continuous-time quantities (e.g. `P(N=n)` at an arbitrary instant, not just at departures) needs a
supplementary-variable argument that is more involved to get right from scratch than the effort
budget for this round of the session warrants (after two demanding "first-principles derivation"
epics — EDF and deadline admission control — a lower-risk approach was explicitly requested).

Instead, this epic reuses a technique already proven repeatedly in this repository: **fit the
general batch-service distribution to a phase-type family from its raw moments, and augment the
existing `(batch-size-in-service, waiting)` CTMC with a phase dimension** — the same "exact given
the assumed family" standard already applied throughout (`MaxDistribution`'s H2/Gamma/Erlang
fits, the SLA layer's H2/Gamma tail fit, `SplitJoinCalc`'s non-Pareto branch). Scoped specifically
to **Erlang(k, rate)** (not the more general H2, which needs a slightly more involved two-branch
phase structure) — Erlang naturally represents low-CV (≤1) batch-service times, a realistic regime
for GPU/LLM batch processing (fairly predictable per-batch duration), and its single-chain phase
structure augments the existing CTMC state space cleanly: `(batch size i, Erlang phase p, waiting
j)`. `k=1` (Erlang collapses to Exponential) must reduce **exactly** to the already-validated
`BulkServiceMM1Calc` — verified to full float64 precision before writing any further code.

**A units/convention bug caught before committing to the derivation:** the phase-transition rate
was first set to `k*rate` (intending the *aggregate* rate across `k` phases to match a "rate"
input meaning `1/mean`) — this is wrong: `k` phases each at rate `k*rate` gives a total mean of
`1/rate`, not `k/rate`. Caught immediately by the `k=1` regression check disagreeing with
`k=3` against an independent DES by a large (not noise-level) margin; fixed by using the per-phase
rate directly (`rate`, giving total Erlang mean `k/rate`) — validated afterward against DES to
within Monte Carlo tolerance for `k=3`.

## Scope

1. **Core deliverable:** `BulkServiceErlangCalc` — `M/Erlang(k,rate)^[a,b]/1`, mean number in
   system / mean sojourn & wait (Little's law), same accuracy level EPIC-012 originally shipped
   (mean-only; exact raw moments for the general-service case are a further reserve item, mirroring
   how EPIC-032 added moments to EPIC-012's exponential model as a *separate* follow-up epic).
2. **Reserve, not in this epic:** H2-fitted (CV≥1) batch service; exact raw moments (not just
   mean) for the Erlang-fitted case; batch-size-dependent general-service parameters (already
   supported for the exponential-rate case via a callable, not attempted here for the
   phase-type case); the classical embedded-chain/PGF route for a fully general (not
   phase-type-restricted) service distribution.

Full derivation: `docs/roadmaps/bulk_service_general_erlang_roadmap.md`.
