# Queue-aware service model selection artifacts

EPIC-061 compares early queue-summary selection against CRPS selection and a
fixed coarse baseline. Two validation and two test origins per source; family
choices freeze before test scoring/replay. These traces were explored previously:
temporal separation is not a new blind evaluation.

[Protocol/API](../../docs/queue_aware_selection.md),
[epic](../../docs/epics/EPIC-061-queue-aware-model-selection.md),
[results](../../docs/research/queue-aware-selection-results-2026-10.md).

## Provenance and transformations

SDSC SP2: SDSC / Victor Hazlewood, conversion by Dror Feitelson, DIR-LAB mirror;
NPACI JOBLOG research/educational/non-profit conditions, **not MIT**.
Kalos: Shanghai AI Laboratory / InternLM AcmeTrace, **CC-BY-4.0, not MIT**.
Exact URLs, revisions, SHA-256 and eligibility counts are in manifest.json;
source/license audits are retained in [SDSC](../../docs/real_trace_calibration.md)
and [Kalos](../../docs/modern_gpu_trace.md).

Transformations: eligible completed histories, conditional service ECDFs, fixed
cohorts with partial observed terminal environment, unchanged lifecycle replay,
validation-only selection and late refits. Outputs contain aggregates and hashes,
not raw rows or user IDs. Raw data remain in `.cache/real_trace/` outside git.

## Files and reproduction

- Four `*-validation-*.json`: uncapped controls and all candidates, CRPS, coverage,
  drift, service/workload hashes, summaries and decisions.
- `selection.json`: both sources' frozen queue/CRPS choices, validation hashes,
  leave-one-origin/seed-out sensitivity. Written before any test scoring/replay.
- Four `*-test-*.json`: same fixed candidates, frozen selections, paired test
  contrasts and full success accounting; SDSC includes a separate cap scenario.
- `manifest.json`: source/config/environment, implementation/output hashes.
- `audit.py`: independent raw/CDF/objective checks and observed event metrics,
  reusing the dispatcher, not an independent scheduler implementation.

```bash
.venv/bin/python -m examples.queue_aware_selection_experiment --download --output-dir works/queue_aware_selection
.venv/bin/python -m examples.queue_aware_selection_experiment --workers 6 --output-dir /tmp/most-queue-selection-repeat
.venv/bin/python -m works.queue_aware_selection.audit
```

Completed: **1500 schedules**, including 60 observed controls. Eight-worker primary
and six-worker repeat produced **10 byte-identical JSONs**. Independent audit:
200 service and 250 scenario-specific workload tapes, 24 score/coverage blocks,
all 60 observed schedules, 1500 rows, 1980 intervals, 360 paired per-policy
intervals, 60 policy decisions, two selections and 12 selected-method contrasts.
Test loss of MC means is distinct from mean seed-level loss. Paired CIs are
conditional on the frozen selections, not selection uncertainty. Zero regret
with tied controls is not scheduler validation; caps must be read with success.
