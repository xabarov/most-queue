# Bulk-service with H2-fitted batch service time (CV≥1) — literature review (2026)

Direct continuation of EPIC-035 (`docs/research/bulk-service-general-erlang-2026.md`): that epic
fitted the general batch-service distribution to Erlang(k, rate) (CV≤1). This epic covers the
complementary regime — high-variance (`CV≥1`) batch-service time — via the H2 (hyperexponential,
2-phase mixture) family, the same family used throughout the rest of the library (`MaxDistribution`,
the SLA layer, `fit_h2`) whenever `CV≥1`.

## Literature

Same classical foundation as EPIC-035 (Neuts 1975, Chaudhry & Templeton's *A First Course in Bulk
Queues*) — no new literature search was needed; the scope here is purely mechanical: apply the
identical "fit to a phase-type family from moments, augment the existing CTMC with a phase
dimension" technique already validated for Erlang, to the H2 family instead.

## Model difference from Erlang (EPIC-035)

Erlang(k, rate) is a **chain** of `k` sequential phases (a batch's service progresses through
phase 0, 1, ..., k-1 before completing). H2 is a **branch**: at the moment a batch *starts*
service, one of two phases is chosen probabilistically (phase 1 w.p. `p1`, rate `mu1`; phase 2
w.p. `p2=1-p1`, rate `mu2`) and the batch stays in that single phase for its entire service — no
mid-service phase transitions. This is actually a *simpler* CTMC augmentation than Erlang's
sequential-chain case: state `(i, phase∈{0,1}, j)`, and every "a batch starts" transition (from
idle-threshold or from a completing batch immediately starting the next one) **splits** into two
weighted sub-transitions (rate `×p1` into phase 0, rate `×p2` into phase 1) instead of a single
deterministic "always start in phase 0" transition.

`p1→1` (H2 degenerates to a single exponential `Exp(mu1)`) must reduce **exactly** to the
already-validated `BulkServiceMM1Calc` — verified to full float64 precision
(`1.0293216473192262` vs `1.0293216473192266`) before writing the general-case tests. The general
(genuinely two-phase) case was also validated directly against an independent DES on the first
attempt — no bug this time (unlike EPIC-035's `k²` phase-rate mixup), likely because H2's
"choose-once-at-start" branching is structurally simpler to get right than Erlang's
sequential-phase-advance rate convention.

## Scope

1. **Core deliverable:** `BulkServiceH2Calc` — `M/H2(p1,mu1,mu2)^[a,b]/1`, mean number in
   system / mean sojourn & wait (Little's law) — same mean-only scope as EPIC-035's Erlang
   calculator (and EPIC-012's original exponential calculator before EPIC-032 added exact moments
   as a separate follow-up).
2. **Reserve, not in this epic:** exact raw moments (as with Erlang, this needs a PASTA/tagged-
   customer argument that tracks which phase an arrival finds the batch in — not attempted here);
   batch-size-dependent H2 parameters; a unified `set_servers_from_moments` dispatcher that
   auto-selects Erlang vs H2 by CV (mirroring `theory.utils.sla.fit_from_moments`'s `family="auto"`
   convention) across *both* `BulkServiceErlangCalc` and `BulkServiceH2Calc` -- currently two
   separate classes, chosen manually by the caller based on their distribution's CV.

Full derivation: `docs/roadmaps/bulk_service_general_h2_roadmap.md`.
