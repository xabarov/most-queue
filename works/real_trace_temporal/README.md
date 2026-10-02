# EPIC-056 temporal and dependent service artifacts

Four prespecified historical origins, seven model variants, six unchanged MSJ
disciplines and eight seeds. `manifest.json` lists data/code/output hashes,
configuration, aggregate errors and decision fidelity. `origin-0.json` through
`origin-3.json` retain every run, audit, fit summary, diagnostic and paired contrast.

```bash
.venv/bin/python -m examples.real_trace_temporal_experiment \
  --output-dir works/real_trace_temporal
```

Add `--download` only if the verified EPIC-055 source cache is missing, after
reviewing [source usage conditions](https://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/index.html).
Credit: SDSC / Victor Hazlewood; conversion Dror Feitelson; pinned DIR-LAB mirror.
The JOBLOG data is Copyright 2000 The Regents of the University of California,
All Rights Reserved, with research/educational/non-profit reuse conditions;
it is not distributed under this repository's MIT license. Raw data stays in
the ignored cache with its original notice.

Units are seconds and nodes. Results concern selected completed jobs, not the
entire production cluster. Rank blocks preserve the same donor-rank marginal as
rank-iid, not necessarily as uniform-iid. Circular seams and gaps are artificial.
All intervals are conditional Monte Carlo intervals, not population guarantees.

[Protocol](../../docs/real_trace_temporal.md),
[results and limitations](../../docs/research/real-trace-temporal-results-2026-10.md).
