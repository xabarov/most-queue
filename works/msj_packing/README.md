# EPIC-052 reproducible packing comparison

[Method and API](../../docs/msj_packing.md),
[registered protocol](../../docs/epics/EPIC-052-msj-packing-baselines.md),
[results and limits](../../docs/research/msj-packing-results-2026-10.md).

Run from the repository root with the development environment:

```bash
for regime in one_or_all powers_of_two general; do
  for shape in erlang2 lognormal_hetero; do
    .venv/bin/python -m examples.msj_packing_experiment \
      --regime "$regime" --shape "$shape" --jobs 4000 --replications 8 \
      --output "works/msj_packing/$regime-$shape.json"
  done
done
```

Six strict JSON files, 896 scheduler runs total, 48 independently seeded
history/test bundles within their respective regime/shape. Reusing seed numbers
across regimes does not make those different workloads independent or suitable
for direct policy-paired contrasts. The meaningful pairing is within a fixed
regime, shape, load and seed. Eight repetitions give exploratory Student 95%
intervals, not multiplicity-corrected decisions or population p99 confidence bounds.

Each file includes the full protocol, history/test/forecast/arrival hashes,
historical completion fraction, initial class quantiles and held-out class
coverage; raw seed-level metrics and paired FCFS/MSF summaries. `schedule_hash`
hashes first-start/final-completion arrays. Raw service segments are available
from the simulator but not embedded in these compact artifacts. All latency
statistics exclude the 400 warm-up jobs; operational counters include all 4400
jobs and draining. Float hashes target the same numerical environment.

For a smoke test, use `--jobs 100 --replications 2` with a separate output path.
Do not overwrite the full-run artifacts with smoke output. To audit the first
two full-sized seeds, call `experiment(regime, shape, jobs=4000, replications=2)`
and compare `history_runs` and `replications` against the matching prefixes of
these files. This audit reproduced **224 schedules exactly**, including hashes
and all recorded metrics. No EPIC-047–051 artifact was changed.

`erlang2` describes the independent residual noise U, not the marginal S:
both families include the declared lognormal size feature. ServerFilling is
only present for its eligible power-of-two regimes, and MSFQ only for one-or-all.
Missing policies are not failed or selectively removed runs. No oracle estimates,
rounded resource needs, test-fitted thresholds or silently dropped jobs occur.
