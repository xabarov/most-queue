# Initial state and terminal-outcome replay artifacts

EPIC-057: four origins, four fixed scenarios, observed durations and two empirical
models, eight seeds, six unchanged policies; 1632 schedules. Origin JSON files
retain all runs, scenario audits, model errors, paired scenario changes and
policy choices. The manifest pins data, implementation and artifact hashes.

```bash
.venv/bin/python -m examples.real_trace_lifecycle_experiment \
  --output-dir works/real_trace_lifecycle
```

Reuse the EPIC-055 verified cache; `--download` is opt-in. Credit: SDSC / Victor
Hazlewood; conversion Dror Feitelson; pinned DIR-LAB mirror. The JOBLOG data is
Copyright 2000 The Regents of the University of California, All Rights Reserved,
subject to [educational/research/non-profit reuse conditions](https://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/index.html).
It is not licensed under the repository's MIT license; raw data and its notice
remain in the ignored cache. Only aggregate results and hashes are distributed.

Units: seconds and nodes. Status=5 runtime means observed occupied service until
termination, not full latent completion demand. Missing cancellations are not
zero-work jobs. Carry-in is retrospective and partial. Requested-time limits are
hypothetical enforcement; timeout latency is not successful-completion latency.
Intervals condition on the supplied historical environment.

[Protocol/API](../../docs/real_trace_lifecycle.md),
[results](../../docs/research/real-trace-lifecycle-results-2026-10.md).
