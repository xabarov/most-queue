"""Offline budget decisions and JSON CLI regression checks."""

import json

import pytest

from tools.runpod_budget import assess_plan, main


@pytest.fixture(name="inputs")
def budget_inputs():
    """An authorized first attempt with conservative accounting."""
    return (
        {
            "authorized": True,
            "campaign_budget_usd": 3,
            "max_compute_usd_per_hour": 0.75,
            "max_total_usd_per_hour": 0.77,
            "max_campaign_seconds": 7200,
            "cleanup_reserve_seconds": 900,
            "max_allocations": 2,
            "max_concurrent_pods": 1,
            "allowed_gpu_ids": ["NVIDIA A40", "NVIDIA RTX A6000"],
            "allowed_cloud": "SECURE",
            "gpu_count": 1,
            "max_container_disk_gb": 80,
            "max_persistent_disk_gb": 0,
            "allow_network_volume": False,
        },
        {
            "gpu_id": "NVIDIA A40",
            "cloud": "SECURE",
            "gpu_count": 1,
            "compute_usd_per_hour": 0.75,
            "storage_usd_per_hour": 0.02,
            "work_seconds": 2700,
            "container_disk_gb": 80,
            "persistent_disk_gb": 0,
            "network_volume_ids": [],
        },
        {"spent_usd": 0, "allocated_seconds": 0, "allocations": 0, "active_pod_ids": []},
    )


def test_allowed_and_decimal_rate_boundary(inputs):
    """0.75 + 0.02 equals the 0.77 limit exactly in decimal accounting."""
    result = assess_plan(*inputs)
    assert result == {"allowed": True, "reasons": [], "projected_total_usd": 0.77}


@pytest.mark.parametrize("spent,allowed", [(2.23, True), (2.23000001, False)])
def test_money_boundary_includes_cleanup(inputs, spent, allowed):
    """Cleanup consumes a further 900 seconds of the budget."""
    inputs[2]["spent_usd"] = spent
    result = assess_plan(*inputs)
    assert result["allowed"] is allowed
    assert result["projected_total_usd"] == pytest.approx(spent + 0.77)


@pytest.mark.parametrize("allocated,allowed", [(3600, True), (3601, False)])
def test_time_boundary_includes_previous_attempts_and_cleanup(inputs, allocated, allowed):
    """All earlier attempt time remains charged."""
    inputs[2]["allocated_seconds"] = allocated
    assert assess_plan(*inputs)["allowed"] is allowed


@pytest.mark.parametrize("allocations,allowed", [(1, True), (2, False)])
def test_second_allowed_third_rejected(inputs, allocations, allowed):
    """Allocation count covers all creations, including failed attempts."""
    inputs[2]["allocations"] = allocations
    assert assess_plan(*inputs)["allowed"] is allowed


@pytest.mark.parametrize(
    "section,field,value",
    [
        (0, "authorized", False),
        (0, "gpu_count", 2),
        (1, "gpu_count", 2),
        (1, "gpu_id", "nvidia a40"),
        (1, "cloud", "COMMUNITY"),
        (1, "container_disk_gb", 81),
        (1, "persistent_disk_gb", 1),
        (1, "network_volume_ids", ["volume"]),
        (2, "active_pod_ids", ["pod"]),
        (1, "compute_usd_per_hour", 0.75000001),
        (1, "storage_usd_per_hour", 0.02000001),
    ],
)
def test_constraints(inputs, section, field, value):
    """A violation of any contract condition denies the plan."""
    inputs[section][field] = value
    assert not assess_plan(*inputs)["allowed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, -1, "0.75", None, [], {}])
@pytest.mark.parametrize("section,field", [(0, "campaign_budget_usd"), (1, "compute_usd_per_hour"), (2, "spent_usd")])
def test_invalid_money(inputs, section, field, value):
    """Malformed numbers fail closed without raising."""
    inputs[section][field] = value
    assert assess_plan(*inputs) == {
        "allowed": False,
        "reasons": [f"{('policy', 'plan', 'ledger')[section]}.{field} must be a finite nonnegative number"],
        "projected_total_usd": None,
    }


@pytest.mark.parametrize("value", [True, -1, 1.5, "1", None, float("nan"), float("inf")])
@pytest.mark.parametrize("section,field", [(0, "max_allocations"), (1, "work_seconds"), (2, "allocated_seconds")])
def test_invalid_integer(inputs, section, field, value):
    """Durations and counters are actual nonnegative integers."""
    inputs[section][field] = value
    assert not assess_plan(*inputs)["allowed"]


@pytest.mark.parametrize(
    "section,field,value",
    [
        (0, "authorized", 1),
        (0, "allow_network_volume", 0),
        (0, "allowed_gpu_ids", "NVIDIA A40"),
        (1, "network_volume_ids", {}),
        (2, "active_pod_ids", [None]),
        (1, "work_seconds", 0),
        (1, "cloud", None),
        (1, "gpu_id", []),
        (0, "allowed_cloud", ""),
    ],
)
def test_invalid_structures(inputs, section, field, value):
    """Lists, strings, and boolean flags must have their declared types."""
    inputs[section][field] = value
    assert not assess_plan(*inputs)["allowed"]


@pytest.mark.parametrize("section", range(3))
def test_missing_and_nonobject_inputs(inputs, section):
    """Absent required fields and top-level arrays are denied."""
    changed = list(inputs)
    changed[section] = []
    assert not assess_plan(*changed)["allowed"]
    inputs[section].clear()
    assert not assess_plan(*inputs)["allowed"]


def test_cli(inputs, tmp_path, capsys):
    """The CLI emits JSON and exits successfully only for allowed plans."""
    args = []
    for name, data in zip(("policy", "plan", "ledger"), inputs):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        args.extend([f"--{name}", str(path)])
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["allowed"]
    (tmp_path / "policy.json").write_text("{", encoding="utf-8")
    assert main(args) == 1
    assert not json.loads(capsys.readouterr().out)["allowed"]
    assert main([]) == 1
    assert not json.loads(capsys.readouterr().out)["allowed"]
