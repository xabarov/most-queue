# GPU resource envelope artifacts

EPIC-060 is a prespecified resource sensitivity study, not an estimate of quotas
or production capacity. Three origins, two allocation assumptions and four node
inventories produce 24 cells: 20 feasible, four explicitly infeasible. Target IDs
and forecasts are fixed per origin; a service tape is shared across resource
cells and policies within each variant/seed/origin.

See [protocol/API](../../docs/gpu_resource_envelope.md),
[epic](../../docs/epics/EPIC-060-gpu-resource-envelope.md),
[results](../../docs/research/gpu-resource-envelope-results-2026-10.md).

## Provenance

Data: Shanghai AI Laboratory / InternLM, AcmeTrace Kalos 2023; Qinghao Hu et al.,
[NSDI 2024](https://www.usenix.org/conference/nsdi24/presentation/hu).
[Pinned source](https://github.com/InternLM/AcmeTrace/tree/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb),
[CC-BY-4.0](https://github.com/InternLM/AcmeTrace/blob/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb/LICENSE.txt), not MIT.
SHA-256 `7c5a7845da1d66448fa668342e4ecb622fd9a005118080660ec3ad2788d4c9bf`.
Raw records remain in `.cache/real_trace/acme-kalos.csv` outside git; transformed
outputs contain aggregates and hashes, not raw jobs or user IDs.

Transformations: accepted observed execution intervals, completed-history ECDF,
fixed selected cohort and terminal competitors, hypothetical GPU/node resource
demands, replay with unchanged nonpreemptive dispatchers. Recorded node_num
does not identify actual placement or exclusivity.

## Files

- `envelope.json`: full recorded-occupation compatibility and complete fixed-cohort
  feasibility matrix, written before any scheduler replay.
- `origin-0.json`, `origin-1.json`, `origin-2.json`: .35/.50/.70 origins, all rows,
  workload hashes, same-cell model errors, paired nominal contrasts and policy
  regret with explicit reference tie counts. No rows for infeasible cells.
- `manifest.json`: source/config/environment and implementation/artifact hashes.
- `audit.py`: independent raw parsing, CDF/projection/metrics/summary checks;
  reuses the EPIC-059 independent raw parser and scheduling engine, not new runner
  or model code. This is not an independent dispatcher implementation.

```bash
.venv/bin/python -m examples.gpu_resource_envelope_experiment --download --output-dir works/gpu_resource_envelope
.venv/bin/python -m examples.gpu_resource_envelope_experiment --workers 6 --output-dir /tmp/most-queue-resource-repeat
.venv/bin/python -m works.gpu_resource_envelope.audit
```

Completed: **1080 schedules**, 432/324/324 by origin. The full repeat with six
workers (primary: eight) produced five byte-identical JSON files. The independent
audit verified 180 cell-specific workload tapes, replayed all 120 observed
schedules, and checked 1080 rows, 1200 model intervals, 720 contrasts, 40 choices
and four infeasible cells. All 20 cells tie across six policies on observed p99 T;
zero decision regret does not validate scheduler selection.

Reserved GPU-equivalent seconds and requested GPU-seconds are distinct under
exclusive-node assumptions; neither is hardware utilization. Window utilization
excludes drain, while work ledgers include drain and historical carry residuals.
