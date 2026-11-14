"""Offline pre-creation budget check; this does not launch or monitor pods."""

import argparse
import json
import math
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

_MONEY = {
    "policy": ("campaign_budget_usd", "max_compute_usd_per_hour", "max_total_usd_per_hour"),
    "plan": ("compute_usd_per_hour", "storage_usd_per_hour"),
    "ledger": ("spent_usd",),
}
_INTEGERS = {
    "policy": (
        "max_campaign_seconds",
        "cleanup_reserve_seconds",
        "max_allocations",
        "max_concurrent_pods",
        "gpu_count",
        "max_container_disk_gb",
        "max_persistent_disk_gb",
    ),
    "plan": ("gpu_count", "work_seconds", "container_disk_gb", "persistent_disk_gb"),
    "ledger": ("allocated_seconds", "allocations"),
}


def _validate(policy, plan, ledger):
    """Validate complete inputs before any comparisons or arithmetic."""
    for name, data in (("policy", policy), ("plan", plan), ("ledger", ledger)):
        if not isinstance(data, dict):
            raise ValueError(f"{name} must be an object")
        for field in _MONEY[name]:
            value = data.get(field)
            if type(value) not in (int, float) or value < 0:
                raise ValueError(f"{name}.{field} must be a finite nonnegative number")
            if not Decimal(str(value)).is_finite():
                raise ValueError(f"{name}.{field} must be a finite nonnegative number")
        for field in _INTEGERS[name]:
            value = data.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name}.{field} must be a nonnegative integer")
    for field in ("authorized", "allow_network_volume"):
        if not isinstance(policy.get(field), bool):
            raise ValueError(f"policy.{field} must be a boolean")
    for data, field in ((policy, "allowed_cloud"), (plan, "gpu_id"), (plan, "cloud")):
        if not isinstance(data.get(field), str) or not data[field]:
            raise ValueError(f"{field} must be a nonempty string")
    for data, field in ((policy, "allowed_gpu_ids"), (plan, "network_volume_ids"), (ledger, "active_pod_ids")):
        value = data.get(field)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"{field} must be a list of nonempty strings")
    if plan["work_seconds"] == 0:
        raise ValueError("plan.work_seconds must be positive")


def assess_plan(policy, plan, ledger):
    """Return a conservative local decision, with cleanup time charged additionally."""
    reasons = []
    try:
        _validate(policy, plan, ledger)
        with localcontext() as context:
            context.prec = 1000
            compute = Decimal(str(plan["compute_usd_per_hour"]))
            total_rate = compute + Decimal(str(plan["storage_usd_per_hour"]))
            seconds = plan["work_seconds"] + policy["cleanup_reserve_seconds"]
            projected = Decimal(str(ledger["spent_usd"])) + total_rate * seconds / Decimal(3600)
            checks = (
                (policy["authorized"] is True, "Explicit authorization is required"),
                (plan["gpu_count"] == policy["gpu_count"] == 1, "Exactly one GPU is required"),
                (plan["gpu_id"] in policy["allowed_gpu_ids"], "GPU is not allowed"),
                (plan["cloud"] == policy["allowed_cloud"], "Cloud does not match policy"),
                (not plan["network_volume_ids"], "Network volumes are forbidden"),
                (plan["container_disk_gb"] <= policy["max_container_disk_gb"], "Container disk exceeds limit"),
                (plan["persistent_disk_gb"] <= policy["max_persistent_disk_gb"], "Persistent disk exceeds limit"),
                (len(ledger["active_pod_ids"]) < policy["max_concurrent_pods"], "Concurrent pod limit reached"),
                (ledger["allocations"] < policy["max_allocations"], "Allocation limit reached"),
                (compute <= Decimal(str(policy["max_compute_usd_per_hour"])), "Compute rate exceeds limit"),
                (total_rate <= Decimal(str(policy["max_total_usd_per_hour"])), "Total hourly rate exceeds limit"),
                (ledger["allocated_seconds"] + seconds <= policy["max_campaign_seconds"], "Campaign time exceeded"),
                (projected <= Decimal(str(policy["campaign_budget_usd"])), "Campaign budget exceeded"),
            )
            reasons.extend(reason for passed, reason in checks if not passed)
            projected_float = float(projected)
            if not math.isfinite(projected_float):
                raise ValueError("Projected cost cannot be represented as a finite JSON number")
    except (ValueError, TypeError, InvalidOperation, OverflowError) as error:
        return {"allowed": False, "reasons": [str(error)], "projected_total_usd": None}
    return {"allowed": not reasons, "reasons": reasons, "projected_total_usd": projected_float}


class _Parser(argparse.ArgumentParser):
    """Keep invalid command lines machine readable."""

    def error(self, message):
        raise ValueError(message)


def main(argv=None):
    """Read three local JSON files and emit a decision without creating resources."""
    parser = _Parser(description=__doc__)
    for name in ("policy", "plan", "ledger"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    try:
        args = parser.parse_args(argv)
        inputs = [json.loads(getattr(args, name).read_text(encoding="utf-8")) for name in ("policy", "plan", "ledger")]
        result = assess_plan(*inputs)
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        result = {"allowed": False, "reasons": [str(error)], "projected_total_usd": None}
    print(json.dumps(result, allow_nan=False))
    return 0 if result["allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
