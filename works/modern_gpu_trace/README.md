# Acme Kalos calibration artifacts

EPIC-058 contains 1224 schedules: four fixed origins, three lifecycle scenarios,
one observed-duration reference and two empirical service generators with eight
seeds, six unchanged dispatchers. This is a nominal GPU-request pool abstraction,
not a reconstruction of Kubernetes, quotas or physical GPU utilization.

See [protocol and API](../../docs/modern_gpu_trace.md),
[epic](../../docs/epics/EPIC-058-modern-gpu-trace-validation.md),
[results](../../docs/research/modern-gpu-trace-results-2026-10.md).

## Provenance and attribution

Data: Shanghai AI Laboratory, AcmeTrace / Kalos; Qinghao Hu et al.,
[Characterization of Large Language Model Development in the Datacenter](https://www.usenix.org/conference/nsdi24/presentation/hu),
NSDI 2024. [Official source](https://github.com/InternLM/AcmeTrace/tree/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb),
[CC-BY-4.0 license](https://github.com/InternLM/AcmeTrace/blob/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb/LICENSE.txt).
The source data is not relicensed under Most-Queue's MIT license.

Pinned file: `data/job_trace/trace_kalos.csv`, SHA-256
`7c5a7845da1d66448fa668342e4ecb622fd9a005118080660ec3ad2788d4c9bf`.
Raw records remain in `.cache/real_trace/acme-kalos.csv`, outside git.
Transformations: select positive observed GPU execution intervals, recompute
service from end-start, retain terminal labels, construct conditional empirical
replays. Published JSON contains derived aggregates and hashes, not raw rows.

## Files

- `manifest.json`: protocol, input/code/output hashes, source audit and aggregates.
- `origin-0.json` through `origin-3.json`: .35/.45/.50/.55 raw-time cuts, fitted
  distribution diagnostics, fixed-cohort hashes, per-seed rows, conditional
  intervals, paired scenario changes and policy decisions.
- `audit.py`: independent source parsing, empirical inverse-CDF reconstruction,
  workload/metric checks and ordinary-scheduler compatibility. It imports neither
  the experiment, source adapter nor fitting helpers. It reuses the schedulers;
  it is not an independent implementation of their dispatch algorithms.

Fits contain generic `ServiceCalibration.parameters()` diagnostics, including
unused PH/lognormal parameters; this experiment generates empirical service only.
`mean_t`/`p99_t` concern the 1000 successful targets, not all arrivals. Ledger
covers environment/warmup/drain; carry-running work starts at the replay boundary.

## Reproduce

```bash
.venv/bin/python -m examples.modern_gpu_trace_experiment --download \
  --output-dir works/modern_gpu_trace
.venv/bin/python -m examples.modern_gpu_trace_experiment \
  --output-dir /tmp/most-queue-modern-repeat
.venv/bin/python -m works.modern_gpu_trace.audit \
  --cache .cache/real_trace/acme-kalos.csv
```

The full repeated experiment produced identical bytes for all four origin JSONs
and manifest. Audit: 204 workload tapes, 1224 rows, 1296 MC intervals, 144
deterministic differences, 48 decisions, 72 observed control replays and 408
empty-state rows checked against `MsjGeneralSim`. Capacity, work and time-integral
checks passed. Zero regret here is uninformative: all six observed policies tie
because target waiting is zero in the nominal-pool model.
