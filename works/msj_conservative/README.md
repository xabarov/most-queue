# Controlled-load conservative backfilling experiment

Implementation: [EPIC-048](../../docs/epics/EPIC-048-msj-conservative-controlled-load.md).
Protocol, interpretation and limitations: [report](../../docs/research/msj-conservative-results-2026-10.md).

From the repository root, with the project environment:

```bash
.venv/bin/python -m examples.msj_conservative_experiment --jobs 6000 --replications 6 --geometry two_class --output works/msj_conservative/results.json
.venv/bin/python -m examples.msj_conservative_experiment --jobs 6000 --replications 6 --geometry three_class --output works/msj_conservative/three_class_results.json
```

Each JSON contains its protocol, 360 per-seed runs, 60 policy summaries and
30 paired coupling contrasts. Seeds: 48000–48005; 600 warm-up plus 6000 measured
arrivals per run, all drained. Recorded with Python 3.12.3, NumPy 2.2.5,
SciPy 1.13.0. No external trace, trained predictor or fitted rate is used.
The two geometries use K={1,3} and K={1,3,4}, respectively; `p99_wide` refers
to the largest resource-demand class, not their pooled wide-job population.

Rates target the known ensemble resource load, including the finite-permutation
correction. Realized trace loads fluctuate. Intervals are Student 95% intervals
of independent run statistics or paired differences, not stationary SLOs or
population quantile confidence bounds. No multiplicity adjustment is applied.
The three-class diagnostic was added after observing matching oracle time metrics
in the original two-class grid; it is exploratory, not a preregistered confirmation.
