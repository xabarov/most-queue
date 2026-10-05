# Resource observability audit artifacts

EPIC-063 compares six source schemas and audits all four published Helios
clusters. This is a data audit, **not scheduler replay** (zero scheduler runs).
Helios is a candidate for explicitly modeled daily-capacity scenarios, not an
identified production quota/placement model.

[Method](../../docs/resource_observability.md),
[results and source matrix](../../docs/research/resource-observability-results-2026-10.md),
[epic](../../docs/epics/EPIC-063-resource-observability-audit.md).

Source: SenseTime / S-Lab-System-Group,
[HeliosData revision 159f0ca](https://github.com/S-Lab-System-Group/HeliosData/tree/159f0caeec16600b9b6017862952a36aae01c43f),
[CC-BY-4.0](https://github.com/S-Lab-System-Group/HeliosData/blob/159f0caeec16600b9b6017862952a36aae01c43f/LICENSE.txt),
not the project's MIT license. Please credit Hu et al.,
[SC 2021](https://doi.org/10.1145/3458817.3476223).
Source is provided as-is; these transformations and conclusions are ours, not
endorsed by the source provider. Raw data remain in ignored cache; only
aggregates, hashes and audit code are stored here.

Transformations: count every raw row/state, compare exported durations with
timestamp differences, join GPU requests to same-date VC counts, integrate
closed terminal requested GPU occupancy and hypothetical daily excess. No
timezone, missing capacity, hidden service or scheduler policy is inferred.
Daily counts are not asserted to be hard quotas. Nonterminal rows stay in
accounting but do not become terminal occupation merely because end is filled.

- `helios-audit.json`: counts, raw member hashes, date/VC compatibility, work
  and occupancy diagnostics, completed-only descriptive S/W.
- `manifest.json`: pinned source, attribution, implementation/output hashes.
- `verify.py`: independent pandas parsing and vectorized event integration,
  with no import of the primary audit implementation.
- `verification.json`: independent check receipt and duplicate/unknown-VC diagnostics.

```bash
.venv/bin/python -m examples.resource_observability_audit --download --output-dir works/resource_observability
.venv/bin/python -m examples.resource_observability_audit --output-dir /tmp/most-queue-resource-observability-repeat
.venv/bin/python -m works.resource_observability.verify
.venv/bin/python -m works.resource_observability.verify --output-dir /tmp/most-queue-resource-observability-repeat
```

Full scope: **3,362,981 rows**, including **1,580,464 GPU jobs**, eight source
payloads and 724 daily configuration rows. Primary and independent checks do
not prove source completeness or availability of every mark at submission.
