# Feature-conditional service artifacts

EPIC-059: 450 lifecycle schedules, three prespecified service candidates per
source, selected by an earlier validation CRPS before late test scoring/replay.
See [protocol/API](../../docs/feature_service.md),
[epic](../../docs/epics/EPIC-059-feature-conditional-service.md),
[results](../../docs/research/feature-service-results-2026-10.md).

## Provenance and attribution

SDSC SP2: SDSC / Victor Hazlewood, conversion Dror Feitelson,
[Parallel Workloads Archive](https://www.cs.huji.ac.il/labs/parallel/workload/l_sdsc_sp2/index.html),
DIR-LAB mirror commit `cd433e3d62a01705f255bd48cb55b393862800e7`.
NPACI JOBLOG research/educational/non-profit conditions apply; not MIT.
SHA-256 `64bc6c621c97fe10e74cfc6a8c568ba544fddd99c4f810473027dca99efc2ed1`.

Acme/Kalos: Shanghai AI Laboratory / InternLM, Qinghao Hu et al.,
[NSDI 2024](https://www.usenix.org/conference/nsdi24/presentation/hu),
[pinned source](https://github.com/InternLM/AcmeTrace/tree/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb),
[CC-BY-4.0](https://github.com/InternLM/AcmeTrace/blob/f9fdf591b4876c2875a9e3d28adb1bda8120dfcb/LICENSE.txt).
SHA-256 `7c5a7845da1d66448fa668342e4ecb622fd9a005118080660ec3ad2788d4c9bf`.

Raw records remain in `.cache`, outside git. Transformations: successful-history
selection, conditional ECDFs, partial observed carry-in, generated selected service,
replay metrics. Saved JSONs are our aggregates/hashes, not republished raw data.

## Files and commands

- `selection.json`: early validation scores, training/cohort hashes, frozen choice.
- `sdsc.json`, `kalos.json`: late refits, target scores, per-seed rows, diagnostics,
  conditional MC intervals, paired errors and policy choices with reference ties.
- `manifest.json`: source/config/environment, code hashes and three output hashes.
- `audit.py`: independent source/CDF/score/tape/summary reconstruction and event-based
  checks of all 18 observed controls. Reuses dispatch engines, not runner/fit helpers.

```bash
.venv/bin/python -m examples.feature_service_experiment --download --output-dir works/feature_service
.venv/bin/python -m examples.feature_service_experiment --output-dir /tmp/most-queue-feature-repeat
.venv/bin/python -m works.feature_service.audit
```

`selected` is a pointer to a validation-chosen candidate, not a fourth test-tuned
fit. Kalos type is retrospective; no claim of online availability. A short
terminal latency under runtime cap is not necessarily successful completion.

Full repeat: all four JSON files byte-identical. Independent audit passed:
12 score blocks, 50 generated/observed service tapes, 75 workload tapes,
450 rows, 594 MC intervals, 108 paired intervals, 18 decisions and 18 observed
control schedules with event-based resource integration.
