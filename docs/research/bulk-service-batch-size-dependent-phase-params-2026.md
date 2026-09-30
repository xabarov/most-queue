# Bulk-service batch-size-dependent Erlang/H2 parameters — literature review (2026)

Closes a reserve item flagged since EPIC-035/036: `BulkServiceMM1Calc` (EPIC-012/032) already
supports a batch-size-dependent service rate (`mu(batch_size)` callable, motivated by LLM/GPU
dynamic batching -- a larger batch takes longer per unit but serves more requests at once), but
`BulkServiceErlangCalc`/`BulkServiceH2Calc` (EPIC-035/036) only accepted scalar (batch-size-
independent) parameters.

## No new literature search needed

Purely a mechanical port of `BulkServiceMM1Calc`'s existing callable-parameter convention onto the
phase-type calculators -- no new queueing theory. The only genuinely new piece is the bookkeeping
subtlety described below.

## Scope decision: rate/parameters vary by batch size, phase COUNT stays fixed

Making the phase count itself (`k` for Erlang, or the branch structure for H2) depend on batch
size would require a variable-size state space per batch size -- the same class of complexity
that made EPIC-041 (per-server phase counts) harder than EPIC-039. Kept out of scope here
deliberately: `k` (Erlang) stays a single constructor-level constant across all batch sizes; only
the per-phase `rate` becomes a callable. For H2, the two-branch structure is unconditionally fixed
(always exactly 2 phases); only `p1`/`mu1`/`mu2` become callables.

## The H2 subtlety: which batch's parameters apply at a "batch starts" transition

For Erlang, only the CURRENT batch's own rate matters at every transition (the phase sequentially
advances within a single batch's own service) -- straightforward substitution `rate` ->
`rate_fn(i)` where `i` is the batch currently in service.

For H2, a batch-completion event does two things at once: the COMPLETING batch (size `i`, some
phase) exits at its own rate; and the NEW batch (size `take = min(b, j)`, possibly a DIFFERENT
size than `i`) immediately starts and must choose ITS OWN branch, weighted by `p1(take)`, not
`p1(i)`. Getting this backwards (using the completing batch's `p1` to weight the new batch's
branch split) would silently apply the wrong batch's H2 parameters whenever `p1` varies with
size -- caught and fixed before shipping by explicit code review of exactly which batch size each
`p1_fn(...)` call site refers to, cross-checked against an independent DES with a genuinely
size-varying `p1(i)` (not just a sanity check with constant parameters, which would not have
exposed this class of bug).

## `mean_batch_service` generalization

Both calculators' `get_w()` uses `E[W] = E[V] - mean_batch_service` (mean-only scope, same as
before). With a scalar rate, `mean_batch_service` was a single number (`k/rate` or
`p1/mu1+p2/mu2`). With a batch-size-dependent rate, it generalizes to a busy-time-weighted average
of the per-size mean service time, weighted by `P(batch size = i | server busy)` from the solved
stationary distribution -- the natural, minimal generalization consistent with the existing
mean-only approximation level (not a new PASTA/tagged-customer derivation).

## Scope

**Core deliverable:** `set_servers()` on both `BulkServiceErlangCalc` and `BulkServiceH2Calc`
accepts either a scalar (backward-compatible, unchanged behavior) or a callable
`f(batch_size) -> value` for each server parameter, validated against an independent DES with
genuinely size-varying parameters (not just the degenerate constant case).

**Reserve:** batch-size-dependent phase COUNT (a materially harder, EPIC-041-class extension);
exact moments (still mean-only, same as EPIC-035/036 before this epic).

No roadmap file -- small enough in scope to document fully here and in the module docstrings.
