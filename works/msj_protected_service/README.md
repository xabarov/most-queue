# EPIC-054 reproducible useful-service protection experiment

[Method and API](../../docs/msj_protected_service.md),
[prespecified protocol](../../docs/epics/EPIC-054-msj-protected-service.md),
[results and limits](../../docs/research/msj-protected-service-results-2026-10.md).

Run from the repository root with the development environment:

```bash
for regime in one_or_all powers_of_two; do
  for shape in erlang2 lognormal_hetero; do
    for cost in 0.05 0.2 1; do
      .venv/bin/python -m examples.msj_protected_service_experiment \
        --regime "$regime" --shape "$shape" --cost "$cost" \
        --jobs 4000 --replications 8 --tuning-jobs 2000 --tuning-replications 4 \
        --output "works/msj_protected_service/$regime-$shape-$cost.json"
    done
  done
done
```

Twelve strict JSON files, each with two useful loads .35/.55. Each file contains
32 actual tuning runs and 128 actual test runs, plus 16 derived `tuned` records:
**384 tuning + 1536 test = 1920 scheduler executions**, 1728 test records in total.
`derived=true` is explicit; the selected result is not resimulated or counted as
another independent observation.

The tuning traces are fully completed, independent offline work: seeds
54000–54003, 200 warm-up + 2000 measured jobs. They select h∈{0,1,4,16} for
q=h(c+r) by the mean work-weighted T, separately by regime/shape/load/cost.
Exact ties select smaller h. Held-out seeds 54100–54107 have 400 warm-up + 4000
measured jobs; no held-out label participates in selection. Separate, independently
censored 4000-job KM histories supply EASY forecasts. Costs are symmetric and
scaled by theoretical E[S], not the realized job service or a sample mean.

Service laws match EPIC-053: both include the declared lognormal size feature.
`erlang2` labels the multiplicative noise, not the marginal service distribution.
Arrival rate is fixed by theoretical useful load, not adjusted for overhead.
All policies within a point share the exact arrivals, classes, services and
forecasts. Reused seed numbers across resource/service regimes do not make those
different scenarios a valid policy pair.

Artifacts retain `tuning_history`, `tuning_runs`, frozen `selection` with every
candidate score, held-out `history_runs`, `replications` and three paired summary
families. Baselines are q=0 **with the same overhead**, FirstFit and MSF; the
separately labelled zero-cost SF remains a reference. Intervals are Student 95%
across eight seed statistics, conditional on the fixed training selection,
without multiplicity adjustment. Run p99 intervals are not population p99 bounds.

Latency excludes arrival-index warm-up. All-job counters and overhead resource
times include warm-up/drain; time averages and completion throughput use the
first measured to last input arrival window and exclude drain. The cohort
resource ratio uses all work over the full input-arrival span and is not
stationary utilization. Protection is useful service, not extra overhead.
`protected_preemptions` counts rejected job/event attempts, not a counterfactual
number of saved interruptions. See the method document for timers and ties.

For a smoke test use `--jobs 100 --replications 2 --tuning-jobs 100
--tuning-replications 2` and a different output path. For the registered full-size
audit call `experiment(regime, shape, cost, size=StudySize(replications=2))` from
`examples.msj_protected_service_experiment`. All four tuning seeds, scores and
choices must match; compare held-out history/runs with the matching saved
prefixes. Each audit call executes 32 tuning + 32 test schedules, 768 over all
files. Audit outcomes are recorded in the results report; float hashes target
the same numerical environment. Raw phase logs remain available from the API.
