"""
Unified Erlang/H2 auto-dispatch for general (non-exponential) batch-service
time in bulk-service queues -- closes the reserve flagged in both EPIC-035
(BulkServiceErlangCalc, CV<=1) and EPIC-036 (BulkServiceH2Calc, CV>=1):
picking the right phase-type family by hand from raw moments. Mirrors
theory.utils.sla.fit_from_moments's family="auto" convention (there:
CV>=1 -> H2, CV<1 -> Gamma; here: CV>1 -> H2, CV<=1 -> Erlang, since Erlang
-- not Gamma -- is the phase-type-compatible member of that family for the
CTMC phase-augmentation technique both calculators use).

Purely a dispatch layer: no new queueing theory, both underlying CTMCs are
already validated (see docs/research/bulk-service-general-erlang-2026.md,
docs/research/bulk-service-general-h2-2026.md).
"""

from __future__ import annotations

import math
from typing import Literal

from most_queue.theory.batch.bulk_service_erlang import BulkServiceErlangCalc
from most_queue.theory.batch.bulk_service_h2 import BulkServiceH2Calc

Family = Literal["auto", "erlang", "h2"]


def _coeff_of_variation(moments: list[float]) -> float:
    """cv = std(S) / E[S] from the first two raw moments."""
    m1, m2 = moments[0], moments[1]
    if m1 <= 0:
        raise ValueError(f"mean (moments[0]) must be positive, got {m1}")
    var = m2 - m1 * m1
    if var < 0:
        var = 0.0
    return math.sqrt(var) / m1


def fit_bulk_service_calc(
    a: int,
    b: int,
    moments: list[float],
    family: Family = "auto",
    queue_truncation: int = 300,
) -> BulkServiceErlangCalc | BulkServiceH2Calc:
    """
    Pick Erlang (CV<=1, EPIC-035) or H2 (CV>=1, EPIC-036) for the batch-
    service time from its raw moments, and return a calculator with
    set_servers() already applied -- caller still calls set_sources()
    before run().

    - family="auto": cv <= 1 -> Erlang, cv > 1 -> H2.
    - family="erlang": force Erlang -- raises ValueError if cv > 1
      (fit_erlang would otherwise return an invalid r=0 order).
    - family="h2": force H2 -- raises ValueError if cv < 1 (an H2 mixture
      cannot represent cv < 1) or if fewer than 3 raw moments are given.
    """
    if len(moments) < 2:
        raise ValueError("at least two raw moments (mean, second moment) are required")

    cv = _coeff_of_variation(moments)
    resolved = ("erlang" if cv <= 1.0 else "h2") if family == "auto" else family

    if resolved == "erlang":
        if cv > 1.0:
            raise ValueError(f"Erlang cannot represent cv > 1 (got cv={cv:.4f}); use family='h2' or 'auto'")
        calc = BulkServiceErlangCalc(a=a, b=b, k=1, queue_truncation=queue_truncation)
        calc.set_servers_from_moments(moments[:2])
        return calc

    if resolved == "h2":
        if cv < 1.0:
            raise ValueError(f"H2 cannot represent cv < 1 (got cv={cv:.4f}); use family='erlang' or 'auto'")
        if len(moments) < 3:
            raise ValueError("family='h2' requires at least three raw moments")
        calc = BulkServiceH2Calc(a=a, b=b, queue_truncation=queue_truncation)
        calc.set_servers_from_moments(moments[:3])
        return calc

    raise ValueError(f"unknown family: {family!r}; expected 'auto', 'erlang' or 'h2'")
