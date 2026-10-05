# MSJ capacity calendar artifacts

EPIC-064 replays three prescribed capacity scenarios (fixed aggregate pool,
daily aggregate pool, isolated-VC pool) over one prescribed seven-day Helios
Venus window, under all six `LIFECYCLE_POLICIES`. This reuses EPIC-063's
verified archive and parsing; no new source or download path.

[Method](../../docs/msj_capacity_calendar.md),
[results and limitations](../../docs/research/msj-capacity-calendar-results-2026-10.md),
[epic](../../docs/epics/EPIC-064-msj-capacity-calendar.md).

Source: SenseTime / S-Lab-System-Group,
[HeliosData revision 159f0ca](https://github.com/S-Lab-System-Group/HeliosData/tree/159f0caeec16600b9b6017862952a36aae01c43f),
[CC-BY-4.0](https://github.com/S-Lab-System-Group/HeliosData/blob/159f0caeec16600b9b6017862952a36aae01c43f/LICENSE.txt),
not the project's MIT license. Please credit Hu et al.,
[SC 2021](https://doi.org/10.1145/3458817.3476223).
Source is provided as-is; these transformations and conclusions are ours, not
endorsed by the source provider. Raw data remain in ignored cache; only
aggregates, hashes and audit code are stored here.

Oracle estimates (`estimate == service`) stand in for Helios's missing
requested-runtime field. Daily VC GPU counts are a modeled capacity-calendar
hypothesis per EPIC-063, not a confirmed production quota; the window used
here happens to have a constant total pool, so `fixed_aggregate` and
`daily_aggregate` coincide (the shrink/growth mechanics are instead locked
down by `tests/units/test_msj_capacity_calendar.py` on synthetic data).

- `capacity-calendar-helios.json`: per-scenario job counts, needs sets,
  calendar breakpoints and per-policy status counts/utilization/backfill/
  reservation diagnostics.
- `manifest.json`: pinned archive/source hashes, implementation hashes,
  attribution, scheduler_runs.
- `verify.py`: independent pandas re-derivation of each scenario's job set
  and calendar capacities, with no import of the primary experiment script.
- `verification.json`: independent check receipt.

```bash
.venv/bin/python -m examples.msj_capacity_calendar_experiment --download \
    --output-dir works/msj_capacity_calendar
.venv/bin/python -m examples.msj_capacity_calendar_experiment \
    --output-dir /tmp/most-queue-msj-capacity-calendar-repeat
.venv/bin/python -m works.msj_capacity_calendar.verify
```

Scope: **18 schedules** (3 scenarios x 6 policies) over 3794 (990 for the
isolated VC) in-window Helios Venus GPU jobs. Zero infeasible jobs and zero
reservation/calendar mismatches occurred on this particular window; see the
results report for why that is an honest limitation of the prescribed window,
not evidence the mechanics are untested.
