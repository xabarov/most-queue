# Joint marked-arrival experiment artifacts

EPIC-062 generates interarrival gaps jointly with K/context/request and uses a
common recent coarse S|K mechanism. Six generated variants and observed controls,
four origins, eight seeds, six unchanged policies: **1176 schedules**.
Every scenario is completed-only and empty-start; historical terminal/carry-in
environment is not copied onto synthetic timestamps. These periods were explored
previously, so temporal evaluation is not a new blind holdout.

[Protocol/API](../../docs/joint_marked_arrivals.md),
[epic](../../docs/epics/EPIC-062-joint-marked-arrivals.md),
[results](../../docs/research/joint-marked-arrivals-results-2026-10.md).

## Provenance and transformations

SDSC SP2: SDSC / Victor Hazlewood, conversion by Dror Feitelson, DIR-LAB mirror;
NPACI JOBLOG research/educational/non-profit conditions, **not MIT**.
Kalos: Shanghai AI Laboratory / InternLM AcmeTrace, **CC-BY-4.0, not MIT**.
Exact URLs, revisions, SHA-256 and eligibility counts are in manifest.json;
source/license audits are retained in [SDSC](../../docs/real_trace_calibration.md)
and [Kalos](../../docs/modern_gpu_trace.md).

Transformations: completed-history eligibility, contiguous fully resolved arrival
prefixes, empirical marked tuples, coarse service ECDFs, empty-start replay and
conditional summaries. Outputs contain aggregates and hashes, not raw rows or
user IDs. Raw sources remain in `.cache/real_trace/` outside git. Prefix staleness
is explicit; completed labels/type remain retrospective. Context/request do not
affect S or policies. Synthetic job positions are not matched historical IDs.

## Files and reproduction

- `protocol.json`: fixed configuration, source metadata, every cohort/history
  and prefix audit, saved before any replay.
- Four `sdsc-*.json` / `kalos-*.json`: marked/service/workload hashes, diagnostics,
  all replay rows, conditional intervals, queue losses, contrasts and decisions.
- `manifest.json`: environment, source, 20 implementation and five output hashes.
- `audit.py`: independent raw/prefix/generator/metric reconstruction; observed
  schedules reuse the dispatcher, not an independent scheduler implementation.

```bash
.venv/bin/python -m examples.joint_marked_arrivals_experiment --download --output-dir works/joint_marked_arrivals
.venv/bin/python -m examples.joint_marked_arrivals_experiment --workers 6 --output-dir /tmp/most-queue-joint-repeat
.venv/bin/python -m works.joint_marked_arrivals.audit
```

Eight-worker primary and six-worker repeat produced **six byte-identical JSONs**.
Independent audit: 196 workload tapes, 32 matched block/shuffle inventory pairs,
all 24 observed schedules, 1176 rows, 1008 model intervals, 24 queue losses,
32 paired contrasts and 48 decisions. Loss of MC means differs from mean
seed-level loss. Neither conditional CIs nor zero regret with tied controls
validate population effects or production scheduler choice. No universal joint
generator advantage was found; fixed-arrival coarse remains a mandatory control.
