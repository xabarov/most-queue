# Roadmap: unified Erlang/H2 auto-dispatch for bulk-service batch-service time

> Context: `docs/research/bulk-service-auto-dispatch-2026.md`.
> Base: EPIC-035 (`most_queue/theory/batch/bulk_service_erlang.py`), EPIC-036
> (`most_queue/theory/batch/bulk_service_h2.py`).

## 1. API

```python
Family = Literal["auto", "erlang", "h2"]

def fit_bulk_service_calc(
    a: int,
    b: int,
    moments: list[float],
    family: Family = "auto",
    queue_truncation: int = 300,
) -> BulkServiceErlangCalc | BulkServiceH2Calc:
    ...
```

- `family="auto"`: `cv <= 1` → `BulkServiceErlangCalc` (k=1 fitted via `set_servers_from_moments`,
  using `fit_erlang`'s own rounding to pick `k`); `cv > 1` → `BulkServiceH2Calc`.
- `family="erlang"`: force Erlang -- raises `ValueError` if `cv > 1` (mirrors `fit_erlang`
  producing an invalid `r=0` order otherwise).
- `family="h2"`: force H2 -- raises `ValueError` if `cv < 1` (H2 cannot represent `cv < 1`) or if
  fewer than 3 raw moments are given (H2 fitting needs the third moment).

Returned object is NOT yet fully configured -- caller must still call `set_sources(l)` before
`run()` (matches every other calculator's `set_sources`/`set_servers`/`run()` lifecycle; only
`set_servers` is pre-filled here since that's the part the dispatcher's job is to pick).

## 2. CV computation

Reuse the exact `_coeff_of_variation` formula already used by `theory.utils.sla`
(`sqrt(m2 - m1^2) / m1`, clamped to 0 for negative variance from numerical noise) -- duplicated
locally rather than imported, since `theory.utils.sla` is a different layer (SLA/deadline) and
importing across unrelated theory submodules for a two-line helper is not worth the coupling.

## 3. Tests

| File | What we check |
|---|---|
| `tests/units/test_bulk_service_general.py` (new) | `family="auto"` picks Erlang for a CV≤1 moment set and matches a direct `BulkServiceErlangCalc` construction; picks H2 for a CV≥1 moment set and matches a direct `BulkServiceH2Calc` construction; `family="erlang"` rejects `cv>1`; `family="h2"` rejects `cv<1` and rejects `<3` moments; invalid `family` string rejected; boundary `cv=1` resolves to Erlang. |

## 4. Documentation

- `docs/models/batch.md`+`.ru.md` -- short note appended after the Erlang/H2 subsections pointing
  to the dispatcher as the recommended entry point when the caller doesn't want to compute CV by
  hand.

## 5. Effort

| Stage | Complexity | Days |
|---|---|---|
| Implementation (factory function) | trivial | 0.25 |
| Tests | low | 0.25 |
| Docs | low | 0.25 |
| **Total** | | **0.75** |

---

**Next step:** `most_queue/theory/batch/bulk_service_general.py`, function `fit_bulk_service_calc`
per §1.
