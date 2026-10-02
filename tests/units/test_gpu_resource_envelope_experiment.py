"""Offline projection, pairing, accounting and unchanged-dispatcher regressions."""

import csv
import io
import json
from dataclasses import replace

import numpy as np
import pytest

from examples import gpu_resource_envelope_experiment as module
from examples import modern_gpu_trace_experiment as gpu
from most_queue.sim.utils.acme_trace import ACME_FIELDS, parse_acme_kalos
from most_queue.sim.utils.resource_envelope import ResourceRequest
from tests.units.test_acme_trace import record


def csv_lines(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=ACME_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().splitlines()


def setup():
    rows = []
    for i in range(100):
        need = (1, 2, 8, 16)[i % 4]
        rows.append(
            record(str(i * 2), i * 10, i * 10 + 1, i * 10 + 4 + i % 2, need=need, node_num=str((need + 7) // 8))
        )
        if i % 5 == 0:
            rows.append(record(str(i * 2 + 1), i * 10 + 1, i * 10 + 2, i * 10 + 6, state="FAILED"))
    lines = csv_lines(rows)
    source = parse_acme_kalos(lines, 32)
    requests = module.read_requests(lines, source)
    config = module.EnvelopeConfig(0.5, jobs=20, warmup=5, replications=2, minimum=2)
    return source, requests, module.prepare(source, config), config


def base_case(source, prepared, config):
    forecasts = {k: 7 for k in range(1, source.capacity + 1)}
    cohort = prepared["cohort"]
    return gpu.build_case(
        source, cohort, [j.runtime for j in cohort], forecasts, "carry_terminal", warmup=config.warmup
    )


@pytest.mark.parametrize("policy", module.POLICIES)
def test_nominal_gpu_projection_matches_unchanged_lifecycle_replay(policy):
    source, requests, prepared, config = setup()
    base = base_case(source, prepared, config)
    cell = module.envelope(prepared, requests, "gpu_pool", 4)
    case = module.project_case(base, prepared, requests, cell)
    assert case == base
    expected = gpu.replay_case(base, 32, policy)
    actual = module.run_case(case, prepared, cell, policy)
    for metric in ("mean_t", "p99_t", "mean_w", "gpu_weighted_mean_t", "idle_with_queue"):
        assert actual[metric] == expected[metric]
    assert actual["requested_gpu_time"] == actual["reserved_gpu_equivalent_time"] == expected["total_resource_time"]
    assert actual["requested_gpu_utilization"] == pytest.approx(expected["utilization"])


def test_exclusive_projection_preserves_all_non_resource_fields_and_target_gpu_weights():
    source, requests, prepared, config = setup()
    base = base_case(source, prepared, config)
    case = module.project_case(base, prepared, requests, module.envelope(prepared, requests, "exclusive_nodes", 4))
    assert case.target_indices == base.target_indices and case.target_needs == base.target_needs
    assert case.observation_window == base.observation_window
    for before, after in zip(base.arrivals, case.arrivals):
        assert replace(after, cls=before.cls) == before
    for before, after in zip(base.running, case.running):
        assert before.age == after.age and replace(after.job, cls=before.job.cls) == before.job


def test_one_node_exclusivity_has_independent_analytic_wait_and_slack():
    lines = csv_lines([record("a", 0, 0, 3), record("b", 1, 1, 3), record("c", 2, 2, 3)])
    source = parse_acme_kalos(lines)
    requests = module.read_requests(lines, source)
    cohort = source.jobs
    prepared = {"running": (), "waiting": (), "arrivals": cohort}
    base = gpu.build_case(source, cohort, [3, 2, 1], {1: 3}, "carry_terminal", warmup=0)
    metrics = {}
    for allocation in module.ALLOCATIONS:
        cell = module.envelope(prepared, requests, allocation, 1)
        case = module.project_case(base, prepared, requests, cell)
        metrics[allocation] = module.run_case(case, prepared, cell, "fcfs")
    assert metrics["gpu_pool"]["mean_w"] == 0
    assert metrics["exclusive_nodes"]["mean_w"] == pytest.approx(5 / 3)  # starts 0,3,5
    assert metrics["exclusive_nodes"]["mean_t"] == pytest.approx(11 / 3)
    assert metrics["exclusive_nodes"]["reserved_gpu_equivalent_time"] == 48
    assert metrics["exclusive_nodes"]["requested_gpu_time"] == 6
    assert metrics["exclusive_nodes"]["reserved_utilization"] == 1
    assert metrics["exclusive_nodes"]["requested_gpu_utilization"] == 0.125


def test_envelope_failure_is_never_scheduled_or_reported_as_zero_latency():
    source, requests, prepared, config = setup()
    cell = module.envelope(prepared, requests, "exclusive_nodes", 1)
    assert not cell["feasible"] and cell["oversized"]["arrivals"] > 0
    with pytest.raises(ValueError, match="infeasible"):
        module.project_case(base_case(source, prepared, config), prepared, requests, cell)


def test_small_requests_can_make_exclusive_carry_infeasible_without_oversized_jobs():
    source, requests, prepared, _ = setup()
    jobs = tuple(j for j in source.jobs if j.need == 1)[:3]
    changed = dict(prepared, running=jobs)
    gpu_cell = module.envelope(changed, requests, "gpu_pool", 2)
    node_cell = module.envelope(changed, requests, "exclusive_nodes", 2)
    assert gpu_cell["feasible"] and not node_cell["feasible"]
    assert node_cell["reasons"] == ["initial_running_exceeds_capacity"]


def test_model_fit_excludes_future_and_unsuccessful_outcomes():
    source, _, prepared, config = setup()
    cutoff = prepared["split"]["cutoff"]
    assert all(j.ended_at < cutoff and j.outcome == "completed" for j in prepared["history"])
    modified = replace(
        source,
        terminal_jobs=tuple(
            replace(j, ended_at=j.ended_at + 10000) if j.outcome != "completed" else j for j in source.terminal_jobs
        ),
    )
    assert module.prepare(modified, config)["history"] == prepared["history"]


def test_repeat_pairing_metrics_intervals_and_infeasible_matrix():
    source, requests, prepared, config = setup()
    result = module.run_origin(source, requests, prepared, config, nodes=(4, 2, 1))
    assert result == module.run_origin(source, requests, prepared, config, nodes=(4, 2, 1))
    json.dumps(result, allow_nan=False)
    assert sum(c["feasible"] for c in result["cells"]) == 4
    assert result["scheduler_runs"] == 4 * 3 * 6
    for seed in (None, 60000, 60001):
        assert len({x["services_sha256"] for x in result["tapes"] if x["seed"] == seed}) == 1
    for row in result["rows"]:
        assert row["nodes"] != 1
        assert row["mean_t"] - row["mean_w"] == pytest.approx(row["target_mean_s"])
        assert row["requested_gpu_time"] <= row["reserved_gpu_equivalent_time"]
        assert row["requested_gpu_utilization"] <= row["reserved_utilization"] + 1e-12
        assert sum(x["count"] for x in row["groups"]) == row["target_count"] == 20
    for summary in result["summaries"]:
        rows = [
            r
            for r in result["rows"]
            if r["variant"] == "recent_coarse" and all(r[k] == summary[k] for k in ("allocation", "nodes", "policy"))
        ]
        for metric in module.METRICS:
            assert summary["metrics"][metric]["mean"] == np.mean([r[metric] for r in rows])
    for contrast in result["contrasts"]:
        a = [
            r for r in result["rows"] if all(r[k] == contrast[k] for k in ("allocation", "nodes", "variant", "policy"))
        ]
        b = [
            r
            for r in result["rows"]
            if r["allocation"] == "gpu_pool"
            and r["nodes"] == 4
            and all(r[k] == contrast[k] for k in ("variant", "policy"))
        ]
        for metric, value in contrast["difference_from_nominal_gpu"].items():
            assert value["mean"] == np.mean([x[metric] - y[metric] for x, y in zip(a, b)])


def test_parallel_workers_preserve_all_values_and_row_order():
    source, requests, prepared, config = setup()
    expected = module.run_origin(source, requests, prepared, config, nodes=(4,))
    assert module.run_origin(source, requests, prepared, config, nodes=(4,), workers=2) == expected
    with pytest.raises(ValueError, match="workers"):
        list(module.ordered_replays([], True))


@pytest.mark.parametrize("change", [{"gpu_num": "2"}, {"node_num": "0"}, {"node_num": "1.5"}, {"node_num": "nan"}])
def test_mismatching_or_malformed_resource_metadata_is_rejected(change):
    original = record()
    source = parse_acme_kalos(csv_lines([original]))
    with pytest.raises(ValueError):
        module.read_requests(csv_lines([{**original, **change}]), source)


def test_missing_and_duplicate_metadata_is_rejected():
    rows = [record(), record("b", 1, 2, 3)]
    source = parse_acme_kalos(csv_lines(rows))
    for value in (rows[:1], rows + rows[:1]):
        with pytest.raises(ValueError):
            module.read_requests(csv_lines(value), source)


@pytest.mark.parametrize("nodes", [(4, 4), (2, 4), (2, 1)])
def test_invalid_node_grid(nodes):
    source, requests, prepared, config = setup()
    with pytest.raises(ValueError):
        module.run_origin(source, requests, prepared, config, nodes=nodes)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fraction": 1},
        {"fraction": True},
        {"jobs": 0},
        {"warmup": -1},
        {"history_limit": 1},
        {"minimum": 1},
        {"replications": 1},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        module.EnvelopeConfig(**{"fraction": 0.5, **kwargs})
