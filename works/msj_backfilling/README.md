# MSJ backfilling pilot

`results.json` contains the raw 144 per-seed runs and paired confidence intervals
from `examples/msj_backfilling_experiment.py` (EPIC-047), generated on 2026-10-02.

Reproduce from the repository root:

```bash
.venv/bin/python -m examples.msj_backfilling_experiment \
  --jobs 12000 --replications 6 --output works/msj_backfilling/results.json
```

Interpretation and limitations: [research report](../../docs/research/msj-ph-backfilling-results-2026-10.md).
These are synthetic finite-cohort results, not measured cluster performance or
proof of stationary tail guarantees. The JSON protocol records workload parameters;
the example specifies distributions, information assumptions and overrun behavior.
