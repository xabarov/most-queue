# EPIC-055 real trace calibration artifacts

`sdsc-sp2.json` contains the prescribed 594 scheduler runs, source/config audit,
historical group fits, common-trace fingerprints, Monte Carlo summaries and
decision-fidelity diagnostics. Units are seconds and nodes. Block IDs are zero
based. The 18 observed rows are separate from 576 generated-model rows.

Reproduce from the repository root:

```bash
.venv/bin/python -m examples.real_trace_calibration_experiment --download \
  --output works/real_trace_calibration/sdsc-sp2.json
```

After caching, omit `--download`. No raw trace is committed. Data provenance,
the pinned mirror URL and SHA-256 are present in both code and result JSON.
The original source is the cleaned SDSC SP2 log from the
[Parallel Workloads Archive](https://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/index.html),
credited to SDSC / Victor Hazlewood, with SWF conversion by Dror Feitelson.
The cached file preserves its original notice. JOBLOG data is Copyright 2000
The Regents of the University of California, All Rights Reserved; its usage
conditions are separate from this project's MIT license. Consult the linked
source for educational, research and non-profit reuse conditions.

Read the [protocol](../../docs/real_trace_calibration.md) and
[results and limitations](../../docs/research/real-trace-calibration-results-2026-10.md).
Do not interpret successful-job filtering as full-cluster replay or the conditional
Monte Carlo intervals as population confidence intervals. An empirical model here
means iid resampling of historical observations, not the observed test sequence.
