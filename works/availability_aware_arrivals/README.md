# Availability-aware arrival history artifacts

EPIC-065 removes EPIC-062's completed-prefix arrival staleness: the
arrival-mark donor history no longer stalls at the first job still
unresolved at cutoff. Same four origins, split, held-out cohorts, policies
and replications as EPIC-062; only the donor-pool construction changes.
**792 schedules** (4 origins x (1 observed + 4 variants x 8 replications) x 6
policies).

[Method](../../docs/availability_aware_arrivals.md),
[epic](../../docs/epics/EPIC-065-availability-aware-arrival-history.md),
[results](../../docs/research/availability-aware-arrivals-results-2026-10.md).

## Provenance and transformations

Same two sources as EPIC-062: SDSC SP2 (SDSC / Victor Hazlewood, conversion
Dror Feitelson, DIR-LAB mirror; NPACI JOBLOG research/educational/non-profit,
**not MIT**) and Acme/Kalos (Shanghai AI Laboratory / InternLM AcmeTrace,
**CC-BY-4.0, not MIT**). No new source, URL, revision or download path.

Transformations: availability-aware donor construction (submit<cutoff over
the already-parsed completed+cancelled / completed+cancelled+failed+timeout+
node_failed population, via `most_queue.sim.utils.workload_trace.
availability_prefix`), reused completed-only service fit, empirical marked
tuples, coarse service ECDFs, empty-start replay and conditional summaries.
Outputs contain aggregates and hashes, not raw rows or user IDs. Raw sources
remain in `.cache/real_trace/` outside git.

## Files and reproduction

- `protocol.json`: fixed configuration, source metadata, every cohort/history,
  and the stale-vs-availability prefix-lag contrast, saved before any replay.
- Four `sdsc-*.json` / `kalos-*.json`: marked/service/workload hashes,
  diagnostics, all replay rows, conditional intervals, queue losses,
  contrasts and decisions.
- `manifest.json`: environment, source, and implementation/output hashes.
- `audit.py`: independent prefix-lag/donor-pool-size re-derivation from the
  raw cached sources, reusing EPIC-059's independent raw parser; does not
  import this epic's `availability_prefix`/`prepare` or re-run the scheduler
  (that reuses the already-tested, unchanged dispatch engine).

```bash
.venv/bin/python -m examples.availability_aware_arrivals_experiment --download \
    --output-dir works/availability_aware_arrivals
.venv/bin/python -m examples.availability_aware_arrivals_experiment --workers 6 \
    --output-dir /tmp/most-queue-availability-repeat
.venv/bin/python -m works.availability_aware_arrivals.audit
```

Eight-worker primary and six-worker repeat produced **six byte-identical
JSONs**. Independent audit confirmed prefix_jobs/unresolved_at_cutoff/
prefix_lag for all four origins. Removing the staleness artifact is not a
uniform improvement: Q (queue-prediction log-error) gets markedly worse on
SDSC and markedly better on Kalos .70; discipline choice by mean T is
unchanged on every origin. See the results report for the full table and a
hedged (not proven) explanation via arrival-rate (mean gap) shift.
