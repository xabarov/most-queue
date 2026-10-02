"""Synthetic-only replay, leakage, cache and independent metric checks."""

import hashlib
import io
import json
from dataclasses import replace

import numpy as np
import pytest

from examples import modern_gpu_trace_experiment as module
from examples.real_trace_temporal_experiment import TemporalConfig, prepare_origins
from most_queue.random.trace_resampling import ConditionalEmpirical
from most_queue.sim.msj_general import MsjGeneralSim, MsjTraceJob
from most_queue.sim.msj_lifecycle import MsjCarryIn, MsjLifecycleJob, MsjLifecycleSim
from tests.units.test_acme_trace import parse, record


def setup():
    rows = []
    for i in range(300):
        rows.append(record(str(2 * i), i * 10, i * 10 + 1, i * 10 + 4 + i % 5, need=1 + i % 2))
        if i % 7 == 0:
            rows.append(
                record(str(2 * i + 1), i * 10 + 1, i * 10 + 3, i * 10 + 6, state="FAILED" if i % 2 else "CANCELLED")
            )
    source = parse(rows, 4)
    config = TemporalConfig(fractions=(0.4,), jobs=20, warmup=5, replications=2, first_seed=58000)
    prepared = prepare_origins(source, config)[0]
    model = ConditionalEmpirical.fit([j.need for j in prepared[0]], [j.runtime for j in prepared[0]])
    forecasts = {k: model.distribution(k)[0].parameters()["p90"] for k in range(1, 5)}
    return source, config, prepared, forecasts, [j.runtime for j in prepared[2]]


def test_targets_forecasts_and_environment_stay_paired():
    source, config, prepared, forecasts, services = setup()
    base = module.build_case(source, prepared[2], services, forecasts, "carry_terminal", warmup=config.warmup)
    changed = module.build_case(
        source, prepared[2], np.array(services) * 1000, forecasts, "carry_terminal", warmup=config.warmup
    )
    assert base.running == changed.running and base.waiting == changed.waiting
    assert [j.estimate for j in base.arrivals] == [j.estimate for j in changed.arrivals]
    assert [j for j in base.arrivals if j.outcome != "completed"] == [
        j for j in changed.arrivals if j.outcome != "completed"
    ]
    assert all(j.runtime_limit is None for j in base.arrivals)
    for case in (base, changed):
        offset = len(case.running) + len(case.waiting)
        assert [case.arrivals[i - offset].arrival for i in case.target_indices] == [
            j.submit - prepared[2][0].submit for j in prepared[2][5:]
        ]
        assert len(case.target_indices) == 20
    assert base.trace_sha256 != changed.trace_sha256


def test_determinism_means_intervals_pairing_and_resource_conservation():
    source, config, prepared, _, _ = setup()
    result = module.run_origin(source, prepared, config)
    assert result == module.run_origin(source, prepared, config)
    json.dumps(result, allow_nan=False)
    assert result["scheduler_runs"] == 3 * 5 * 6
    for row in result["rows"]:
        assert row["mean_t"] - row["mean_w"] == pytest.approx(row["target_mean_s"])
        assert row["total_resource_time"] == sum(row["resource_time_by_outcome"].values())
        assert row["target_count"] == 20
    for summary in result["summaries"]:
        rows = [r for r in result["rows"] if all(r[k] == summary[k] for k in ("scenario", "model", "policy"))]
        for metric in module.METRICS:
            assert summary["metrics"][metric]["mean"] == np.mean([r[metric] for r in rows])
    for change in result["scenario_changes"]:
        a = [r for r in result["rows"] if all(r[k] == change[k] for k in ("scenario", "model", "policy"))]
        b = [
            r
            for r in result["rows"]
            if r["scenario"] == change["baseline"] and all(r[k] == change[k] for k in ("model", "policy"))
        ]
        assert change["differences"]["mean_t"]["mean"] == np.mean([x["mean_t"] - y["mean_t"] for x, y in zip(a, b)])
    assert len(module.aggregate([result])) == 6


@pytest.mark.parametrize("policy", module.POLICIES)
def test_empty_completed_matches_unchanged_scheduler(policy):
    source, config, prepared, forecasts, services = setup()
    case = module.build_case(source, prepared[2], services, forecasts, "empty_completed", warmup=config.warmup)
    reference = MsjGeneralSim(4, policy)
    reference.set_servers(range(1, 5))
    result = reference.run_trace(
        tuple(MsjTraceJob(j.arrival, j.cls, j.service, j.estimate) for j in case.arrivals), warmup_jobs=5
    )
    row = module.replay_case(case, 4, policy)
    assert row["mean_t"] == result.v[0] and row["p99_t"] == result.v_quantiles[0.99]
    assert row["mean_w"] == result.w[0] and row["utilization"] == result.utilization


def test_lifecycle_new_labels_capacity_one_match_independent_lindley():
    sim = MsjLifecycleSim(1)
    sim.set_servers([1])
    jobs = tuple(
        MsjLifecycleJob(i, 0, i + 1, 3, label)
        for i, label in enumerate(("completed", "cancelled", "failed", "node_failed", "timed_out"))
    )
    result = sim.run_lifecycle(jobs)
    now = 0
    for idx, job in enumerate(jobs):
        start = max(job.arrival, now)
        now = start + job.service
        assert result.start_times[idx] == start and result.release_times[idx] == now
        assert result.outcomes[idx] == job.outcome
        assert result.resource_time_by_outcome[job.outcome] == job.service


def test_failure_carry_preserves_residual_and_caps_do_not_create_success():
    sim = MsjLifecycleSim(2)
    sim.set_servers([1])
    result = sim.run_lifecycle(
        [MsjLifecycleJob(0, 0, 5, 3, "failed", 2), MsjLifecycleJob(1, 0, 1, 1, "node_failed", 1)],
        initial_running=[MsjCarryIn(MsjLifecycleJob(0, 0, 9, 4, "failed"), 6)],
    )
    assert result.outcomes == ["failed", "timed_out", "node_failed"]
    assert result.release_times == [3, 2, 3]
    assert result.resource_time_by_outcome == {
        "completed": 0,
        "cancelled": 0,
        "timed_out": 2,
        "failed": 3,
        "node_failed": 1,
    }


def test_training_does_not_include_future_or_unsuccessful_service():
    source, config, prepared, _, _ = setup()
    cutoff = prepared[3]["cutoff"]
    assert all(j.outcome == "completed" and j.ended_at < cutoff for j in prepared[0])
    modified = replace(
        source,
        terminal_jobs=tuple(
            replace(j, ended_at=j.ended_at + 100000) if j.outcome != "completed" else j for j in source.terminal_jobs
        ),
    )
    assert prepare_origins(modified, config) == prepare_origins(source, config)


@pytest.mark.parametrize("services", [[1], [float("nan")] * 25, [0] * 25])
def test_bad_service_vectors(services):
    source, config, prepared, forecasts, _ = setup()
    with pytest.raises(ValueError):
        module.build_case(source, prepared[2], services, forecasts, "empty_completed", warmup=config.warmup)


def test_invalid_cohorts_and_scenarios():
    source, _, prepared, forecasts, services = setup()
    for cohort, scenario, warmup in (
        (prepared[2], "bad", 5),
        (prepared[2], "empty_completed", True),
        (prepared[2], "empty_completed", 25),
        (prepared[2][::-1], "empty_completed", 5),
        (prepared[2][:-1] + (prepared[2][0],), "empty_completed", 5),
    ):
        with pytest.raises(ValueError):
            module.build_case(source, cohort, services, forecasts, scenario, warmup=warmup)


def test_pinned_cache_offline_opt_in_and_corruption(monkeypatch, tmp_path):
    payload = b"synthetic,not,real,data\n"
    monkeypatch.setattr(module, "SOURCE_SHA256", hashlib.sha256(payload).hexdigest())
    path = tmp_path / "trace.csv"
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(module, "urlopen", download)
    with pytest.raises(ValueError, match="not cached"):
        module.verified_source(path)
    assert not calls
    assert module.verified_source(path, True) == payload
    assert module.verified_source(path) == payload and len(calls) == 1
    monkeypatch.setattr(module, "SOURCE_SHA256", "bad")
    with pytest.raises(ValueError, match="SHA-256"):
        module.verified_source(path, True)
    assert path.read_bytes() == payload and len(calls) == 1
    missing = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="SHA-256"):
        module.verified_source(missing, True)
    assert not missing.exists()


def test_download_size_cap_before_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "urlopen", lambda *a, **k: io.BytesIO(b"x" * 12_000_001))
    path = tmp_path / "trace.csv"
    with pytest.raises(ValueError, match="size limit"):
        module.verified_source(path, True)
    assert not path.exists()


def test_source_diagnostics_use_execution_work_not_precomputed_gpu_time():
    source = parse([record(), record("b", submit=1, start=3, end=6, state="FAILED")])
    audit = module.source_diagnostics(source)
    assert audit["accepted_peak_simultaneous_gpu_request"] == 2
    assert audit["accepted_request_seconds_by_outcome"]["completed"] == 3
    assert audit["accepted_request_seconds_by_outcome"]["failed"] == 3


def test_source_diagnostics_reject_zero_span():
    with pytest.raises(ValueError, match="positive arrival span"):
        module.source_diagnostics(parse([record()]))
