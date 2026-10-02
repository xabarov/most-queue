"""Offline temporal guards, exact controls, Lindley, and baseline regression."""

import json
import sys
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from examples import joint_marked_arrivals_experiment as module
from tests.units.test_modern_gpu_trace_experiment import setup as gpu_setup
from tests.units.test_real_trace_lifecycle_experiment import source_fixture


def setup(name="sdsc"):
    source = source_fixture() if name == "sdsc" else gpu_setup()[0]
    config = module.JointConfig((0.4, 0.8), jobs=20, warmup=5, history_limit=60, block_length=3, replications=2)
    return source, config, module.prepare(source, name, config)


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_history_is_known_cohorts_disjoint_and_prefix_contiguous(name):
    source, config, blocks = setup(name)
    ids = set()
    for b in blocks:
        assert all(j.completed_at < b["split"]["cutoff"] for j in b["prefix"] + b["history"])
        assert len(b["cohort"]) == 25
        assert not ids.intersection(j.job_id for j in b["cohort"])
        ids.update(j.job_id for j in b["cohort"])
        assert len(b["recent_prefix"]) <= config.history_limit + 1
        complete = module.feature.completed_source(source, name)
        assert list(b["prefix"]) == list(complete.jobs[: len(b["prefix"])])


def test_unfinished_job_truncates_whole_arrival_suffix_but_not_service_history():
    source, config, blocks = setup()
    target = blocks[0]["prefix"][-15]
    changed = replace(
        source, jobs=tuple(replace(j, runtime=1e6) if j.job_id == target.job_id else j for j in source.jobs)
    )
    b = module.prepare(changed, "sdsc", config)[0]
    assert b["prefix"][-1].submit < target.submit
    assert any(j.submit > target.submit for j in b["history"])
    assert b["prefix_audit"]["omitted_suffix"] == blocks[0]["prefix_audit"]["omitted_suffix"] + 15
    assert b["prefix_audit"]["completed_suffix_omitted"] == blocks[0]["prefix_audit"]["completed_suffix_omitted"] + 14


def test_invalid_coverage_is_rejected_not_resized():
    source, config, _ = setup()
    with pytest.raises(ValueError, match="full held-out"):
        module.prepare(source, "sdsc", replace(config, jobs=1000))
    with pytest.raises(ValueError, match="overlap"):
        module.prepare(source, "sdsc", replace(config, fractions=(0.4, 0.41)))
    with pytest.raises(ValueError, match="prefix"):
        module.prepare(source, "sdsc", replace(config, fractions=(0.005,)))


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_workload_pairing_and_exact_phase_work_inventories(name):
    _, config, blocks = setup(name)
    b = blocks[0]
    tapes = {(v, s): (r, svc) for v, s, r, svc in module.workloads(b, name, config)}
    assert len(tapes) == 13
    for seed in range(config.first_seed, config.first_seed + config.replications):
        block, bs = tapes["recent_joint_block20", seed]
        shuffled, ss = tapes["recent_joint_shuffle20", seed]
        iid, _ = tapes["recent_joint_iid", seed]
        independent, _ = tapes["recent_gap_independent", seed]
        for begin, end in ((0, config.warmup), (config.warmup, len(block))):
            assert Counter(zip(block[begin:end], bs[begin:end])) == Counter(zip(shuffled[begin:end], ss[begin:end]))
            marks = lambda records: Counter((r.need, r.context, r.requested_time) for r in records[begin:end])
            assert marks(iid) == marks(independent)
        a, z = module.arrival_times(block), module.arrival_times(shuffled)
        assert a[config.warmup] == z[config.warmup] and a[-1] == z[-1]
        np.testing.assert_array_equal(module.arrival_times(iid), module.arrival_times(independent))
        assert tapes["fixed_coarse", seed][0] == tapes["observed", None][0]


def test_future_target_service_cannot_change_generated_tapes_or_forecast():
    source, config, blocks = setup()
    target_ids = {j.job_id for j in blocks[0]["cohort"]}
    changed = replace(
        source, jobs=tuple(replace(j, runtime=j.runtime * 3) if j.job_id in target_ids else j for j in source.jobs)
    )
    b = module.prepare(changed, "sdsc", config)[0]
    assert b["history"] == blocks[0]["history"] and b["prefix"] == blocks[0]["prefix"]
    assert module.forecasts_for(b, config) == module.forecasts_for(blocks[0], config)
    left = list(module.workloads(blocks[0], "sdsc", config))[1:]
    right = list(module.workloads(b, "sdsc", config))[1:]
    for (v, s, r, x), (v2, s2, r2, y) in zip(left, right):
        assert (v, s, r) == (v2, s2, r2)
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_replay_determinism_metrics_and_previous_fixed_baseline(name):
    _, config, blocks = setup(name)
    b = blocks[0]
    result = module.run_block(b, name, config)
    assert result == module.run_block(b, name, config)
    assert result["scheduler_runs"] == 78 and len(result["contrasts"]) == 8
    json.dumps(result, allow_nan=False)
    for row in result["rows"]:
        assert row["target_count"] == 20 and sum(g["count"] for g in row["groups"]) == 20
        assert row["mean_t"] - row["mean_w"] == pytest.approx(row["target_mean_s"])
        tape = next(t for t in result["tapes"] if (t["variant"], t["seed"]) == (row["variant"], row["seed"]))
        assert row["resource_time"] == pytest.approx(tape["all_resource_work"])
    forecasts = module.forecasts_for(b, config)
    for v, s, records, services in module.workloads(b, name, config):
        if v not in ("observed", "fixed_coarse"):
            continue
        needs, trace = module.trace_for(records, services, forecasts)
        for policy in module.POLICIES:
            old = module.feature.swf.replay(b["capacity"], needs, trace, policy, config.warmup)
            row = next(r for r in result["rows"] if (r["variant"], r["seed"], r["policy"]) == (v, s, policy))
            for k in ("mean_t", "p99_t", "mean_w", "utilization", "idle_with_queue", "groups"):
                assert old[k] == row[k]
            assert old["node_weighted_mean_t"] == row["weighted_mean_t"]


def test_capacity_one_matches_lindley_with_simultaneous_arrivals():
    records = tuple(module.ArrivalMark(gap, 1) for gap in (0, 0, 1, 0, 3))
    services = np.array([3.0, 1.0, 2.0, 4.0, 1.0])
    needs, trace = module.trace_for(records, services, {1: 2})
    waits = []
    end = 0
    for job in trace:
        begin = max(end, job.arrival)
        waits.append(begin - job.arrival)
        end = begin + job.service
    row = module.replay(({"policy": "fcfs"}, 1, needs, trace, 0))
    assert row["mean_w"] == np.mean(waits)
    assert row["mean_t"] == np.mean(np.array(waits) + services)
    assert row["resource_time"] == sum(services)


def test_batch_horizon_has_null_time_averages_but_defined_latency():
    records = (module.ArrivalMark(0, 1),) * 4
    services = np.arange(1, 5)
    needs, trace = module.trace_for(records, services, {1: 2})
    row = module.replay(({"policy": "fcfs"}, 1, needs, trace, 1))
    assert row["utilization"] is None and row["idle_with_queue"] is None and row["mean_t"] > 0
    d = module.diagnostics(records, services, 1, records)
    assert d["arrival_span"] == 0 and d["transition_rate"] is None and d["gap_scv"] is None
    assert d["lag1_gap"] is None and d["gap_need_correlation"] is None
    assert d["zero_gap_fraction"] == 1


def test_four_random_streams_are_reproducible_and_not_aliased():
    a = module.uniform_streams("sdsc", 100, 63000, 10)
    b = module.uniform_streams("sdsc", 100, 63000, 10)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
        assert np.all((x > 0) & (x < 1))
    for x in a[1:]:
        assert not np.array_equal(x, a[0])
    assert not np.array_equal(a[0], module.uniform_streams("kalos", 100, 63000, 10)[0])


def test_parallel_execution_preserves_exact_artifact():
    _, config, blocks = setup()
    assert module.run_block(blocks[0], "sdsc", config, workers=2) == module.run_block(blocks[0], "sdsc", config)


def test_cli_records_protocol_before_any_replay(tmp_path, monkeypatch):
    sdsc, config, _ = setup()
    kalos = setup("kalos")[0]
    monkeypatch.setattr(module, "CONFIGS", {"sdsc": config, "kalos": config})
    monkeypatch.setattr(module.feature.swf, "verified_source", lambda *args: b"synthetic")
    monkeypatch.setattr(module.feature.gpu, "verified_source", lambda *args: b"synthetic")
    monkeypatch.setattr(module.feature, "parse_swf_lifecycle", lambda *args: sdsc)
    monkeypatch.setattr(module.feature, "parse_acme_kalos", lambda *args: kalos)
    original = module.run_block

    def checked(*args, **kwargs):
        p = json.loads((tmp_path / "protocol.json").read_text())
        assert set(p["blocks"]) == {"sdsc", "kalos"}
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_block", checked)
    monkeypatch.setattr(sys, "argv", ["runner", "--output-dir", str(tmp_path), "--workers", "1"])
    module.main()
    assert len(list(tmp_path.glob("*.json"))) == 6
    assert json.loads((tmp_path / "manifest.json").read_text())["scheduler_runs"] == 312


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fractions": ()},
        {"fractions": (0.4, 0.4)},
        {"fractions": (np.nan,)},
        {"fractions": (1,)},
        {"jobs": 0},
        {"warmup": True},
        {"history_limit": 2},
        {"minimum": 1},
        {"replications": 1},
    ],
)
def test_bad_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        module.JointConfig(**kwargs)
