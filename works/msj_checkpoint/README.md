# EPIC-053 reproducible checkpoint-cost comparison

[Method and API](../../docs/msj_checkpoint.md),
[registered protocol](../../docs/epics/EPIC-053-msj-checkpoint-cost.md),
[results and limits](../../docs/research/msj-checkpoint-results-2026-10.md).

Run from the repository root with the development environment:

```bash
for regime in one_or_all powers_of_two; do
  for shape in erlang2 lognormal_hetero; do
    .venv/bin/python -m examples.msj_checkpoint_experiment \
      --regime "$regime" --shape "$shape" --jobs 4000 --replications 8 \
      --output "works/msj_checkpoint/$regime-$shape.json"
  done
done
```

Four strict JSON files, **832 scheduler runs**: two resource regimes, two
service families, two useful loads, thirteen policy/cost modes and eight seeds
(53000–53007). Each run has 400 warm-up and 4000 measured jobs, all drained.
The independently censored history contains 4000 jobs per bundle. Four separate
random streams generate history, future jobs, arrivals and censoring. Policies
within a regime/shape/load/seed receive identical arrivals, K, S and forecasts.
Reused seed numbers across workloads do not justify cross-workload pairing.

`erlang2` describes the noise U, not the marginal service S: both service
families multiply class-mean-scaled U by the declared lognormal size feature.
Useful arrival rate is fixed by theoretical E[K*S], never reduced to offset
checkpoint cost. The constants c,r are factors of theoretical E[S], not realized
job durations. The explicit overhead gate is part of the experimental model.

Artifacts include protocol, history/test/arrival/forecast hashes, historical
completion fraction and initial class quantiles, raw seed metrics and three
paired summary families (`zero_contrasts`, `first_fit_contrasts`, `msf_contrasts`).
Intervals are Student 95% across eight seed-level statistics, without a
multiplicity correction. Run-level p99 intervals are not population p99 bounds.
`schedule_hash` covers first starts/final completions; the three interval logs
are available from the simulator, not embedded in these compact files.

Latency excludes warm-up; counters and overhead resource-time include warm-up
and drain. Utilization uses the first measured to last input arrival window;
productive utilization excludes overhead. Throughput counts all completions in
that window divided by its duration, including warm-up jobs finishing there.
`cohort_resource_ratio` instead divides all
useful plus overhead server-time by k times the **first-to-last input arrival**
span. It is an accounting ratio, not stationary utilization or an independently
observed sustainable offered load. Backlog at last arrival and drain time are
reported separately. Packing has no promises; its promise-violation rate is null.

Use `--jobs 100 --replications 2` with a separate output path for a smoke test.
Do not overwrite full-run artifacts with smoke output. For the registered audit,
call `experiment(regime, shape, jobs=4000, replications=2)` from
`examples.msj_checkpoint_experiment`, comparing `history_runs` and `replications`
with the matching saved prefixes. The full-sized audit and validation outcomes
are recorded in the [results report](../../docs/research/msj-checkpoint-results-2026-10.md).
Float hashes target the same numerical environment.
