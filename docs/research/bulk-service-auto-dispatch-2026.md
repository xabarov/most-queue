# Unified Erlang/H2 auto-dispatch for bulk-service batch-service time (2026)

Direct continuation of EPIC-035 (`BulkServiceErlangCalc`, CV≤1) and EPIC-036 (`BulkServiceH2Calc`,
CV≥1): both epics closed a reserve item flagged since EPIC-012 ("general, not exponential,
batch-service time") by fitting to a phase-type family from raw moments, but left the caller to
pick the right family (Erlang vs H2) by hand based on the CV of their own distribution. This epic
closes that gap with a small factory that picks automatically, mirroring the convention already
established in `most_queue.theory.utils.sla.fit_from_moments`'s `family="auto"` (there: CV≥1 → H2,
CV<1 → Gamma).

## No new literature search needed

The scope here is purely software-engineering: no new queueing theory, no new derivation. Both
underlying calculators (`BulkServiceErlangCalc`, `BulkServiceH2Calc`) are already validated
(reduction to `BulkServiceMM1Calc` + independent DES, see the EPIC-035/036 research docs). The only
question is which family to pick given a CV, which is the exact same decision `fit_from_moments`
already makes for the SLA layer -- reused directly rather than re-derived.

## Design difference from `sla.fit_from_moments`

`sla.fit_from_moments` picks Gamma (not Erlang) for CV<1 because Gamma's fractional shape parameter
gives a strictly better fit than Erlang's integer phase count for an arbitrary CV<1 distribution --
but `BulkServiceGammaCalc` doesn't exist (the CTMC phase-augmentation technique used here requires
an integer number of *exponential* phases, which Gamma with non-integer shape does not have; Erlang
is the phase-type-compatible member of the Gamma family). So this dispatcher picks between the two
calculators that actually exist in this library: `BulkServiceErlangCalc` for CV≤1,
`BulkServiceH2Calc` for CV≥1 -- the boundary CV=1 (exact Exponential) is representable exactly by
both (`Erlang(k=1)` or `H2(p1=1)`); resolved to Erlang by convention (matches `fit_erlang`'s own
`round(1/cv²)` -> `r=1` at cv=1).

## Scope

**Core deliverable:** `fit_bulk_service_calc(a, b, moments, family="auto", queue_truncation=300)`
in `most_queue/theory/batch/bulk_service_general.py` -- returns a ready-to-configure
`BulkServiceErlangCalc` or `BulkServiceH2Calc` instance (caller still calls `set_sources()` then
`run()`), selecting the family automatically from the CV of the given raw moments, with explicit
`family="erlang"`/`"h2"` overrides that validate CV-feasibility rather than silently producing a
wrong fit (same "raise, don't silently degrade" discipline as `sla.fit_from_moments`).

**Reserve, not in this epic:** everything already flagged as reserve by EPIC-035/036 individually
(exact moments beyond the mean, batch-size-dependent phase-type parameters) remains reserve --
this epic only unifies the *selection*, not the underlying accuracy/scope of either calculator.

Full derivation: `docs/roadmaps/bulk_service_auto_dispatch_roadmap.md`.
