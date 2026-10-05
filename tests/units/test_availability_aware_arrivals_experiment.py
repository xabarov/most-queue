"""Offline checks that availability-aware donors never stall like the old prefix."""

import json
import sys
from dataclasses import replace

import numpy as np
import pytest

from examples import availability_aware_arrivals_experiment as module
from examples import joint_marked_arrivals_experiment as joint
from tests.units.test_modern_gpu_trace_experiment import setup as gpu_setup
from tests.units.test_real_trace_lifecycle_experiment import source_fixture


def setup(name="sdsc"):
    source = source_fixture() if name == "sdsc" else gpu_setup()[0]
    config = joint.JointConfig((0.4, 0.8), jobs=20, warmup=5, history_limit=60, block_length=3, replications=2)
    return source, config, module.prepare(source, name, config)


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_availability_prefix_is_never_shorter_than_the_stale_prefix(name):
    _, _, blocks = setup(name)
    for b in blocks:
        assert len(b["avail_prefix"]) >= len(b["prefix"])
        assert b["avail_prefix_audit"]["prefix_lag"] <= b["prefix_audit"]["prefix_lag"]


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_availability_prefix_reaches_the_cutoff_when_nothing_unresolved(name):
    """Without a stalling job, both constructions pick the same last submit."""
    source, config, blocks = setup(name)
    for old, new in zip(blocks, module.prepare(source, name, config)):
        # The fixture has no artificially-extended runtime, so unless a job's
        # own completed_at happens to straddle the cutoff, lag should match.
        if old["prefix_audit"]["omitted_suffix"] == 0:
            assert new["avail_prefix_audit"]["prefix_lag"] == pytest.approx(old["prefix_audit"]["prefix_lag"])


def test_unresolved_job_shrinks_the_lag_gap_instead_of_stalling_the_prefix():
    source, config, blocks = setup()
    baseline_omitted = blocks[0]["prefix_audit"]["omitted_suffix"]
    target = blocks[0]["prefix"][-15]
    changed = replace(
        source, jobs=tuple(replace(j, runtime=1e6) if j.job_id == target.job_id else j for j in source.jobs)
    )
    old = joint.prepare(changed, "sdsc", config)[0]
    new = module.prepare(changed, "sdsc", config)[0]
    # The stale construction stalls 15 jobs early (as in the joint-arrivals
    # suite); the availability-aware one is unaffected by the same job.
    assert old["prefix_audit"]["omitted_suffix"] == baseline_omitted + 15
    assert new["avail_prefix_audit"]["unresolved_at_cutoff"] >= 1
    assert new["avail_prefix_audit"]["prefix_lag"] < old["prefix_audit"]["prefix_lag"]
    assert len(new["avail_prefix"]) > len(old["prefix"])


def test_availability_pool_never_uses_runtime_or_wait_fields():
    """ArrivalMark donors only carry submit-derived gap/need/context/request."""
    _, config, blocks = setup()
    pool = joint.fit_arrivals(blocks[0]["avail_recent_prefix"], "sdsc")
    for donor in pool.donors:
        assert not hasattr(donor, "runtime") and not hasattr(donor, "wait")


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_workload_pairing_includes_all_four_variants(name):
    _, config, blocks = setup(name)
    b = blocks[0]
    tapes = {(v, s): (r, svc) for v, s, r, svc in module.workloads(b, name, config)}
    assert {v for v, _ in tapes} == {"observed", *module.VARIANTS}
    assert len(tapes) == 1 + len(module.VARIANTS) * config.replications
    for seed in range(config.first_seed, config.first_seed + config.replications):
        assert tapes["fixed_coarse", seed][0] == tapes["observed", None][0]
        stale, _ = tapes["recent_joint_iid_stale", seed]
        avail, _ = tapes["recent_availability_iid", seed]
        # Same uniforms drive both donor pools; pools differ, so records may
        # legitimately differ, but both must be well-formed ArrivalMark tuples.
        assert len(stale) == len(avail) == len(tapes["observed", None][0])


def test_future_target_service_cannot_change_generated_tapes_or_forecast():
    source, config, blocks = setup()
    target_ids = {j.job_id for j in blocks[0]["cohort"]}
    changed = replace(
        source, jobs=tuple(replace(j, runtime=j.runtime * 3) if j.job_id in target_ids else j for j in source.jobs)
    )
    b = module.prepare(changed, "sdsc", config)[0]
    assert b["avail_prefix"] == blocks[0]["avail_prefix"]
    assert module.forecasts_for(b, config) == module.forecasts_for(blocks[0], config)
    left = list(module.workloads(blocks[0], "sdsc", config))[1:]
    right = list(module.workloads(b, "sdsc", config))[1:]
    for (v, s, r, x), (v2, s2, r2, y) in zip(left, right):
        assert (v, s, r) == (v2, s2, r2)
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_replay_determinism_and_block_audit_lag_reduction(name):
    _, config, blocks = setup(name)
    b = blocks[0]
    result = module.run_block(b, name, config)
    assert result == module.run_block(b, name, config)
    expected_tapes = 1 + len(module.VARIANTS) * config.replications  # +1 for the single "observed" tape
    assert result["scheduler_runs"] == expected_tapes * len(module.POLICIES)
    assert len(result["contrasts"]) == len(module.CONTRASTS)
    json.dumps(result, allow_nan=False)
    audit = module.block_audit(b, name)
    assert audit["prefix_lag_reduction"] == pytest.approx(
        audit["prefix_audit_stale"]["prefix_lag"] - audit["prefix_audit_availability"]["prefix_lag"]
    )
    assert audit["prefix_lag_reduction"] >= 0
    for row in result["rows"]:
        assert row["target_count"] == 20 and sum(g["count"] for g in row["groups"]) == 20
        assert row["mean_t"] - row["mean_w"] == pytest.approx(row["target_mean_s"])


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
        assert "prefix_lag_reduction" in p["blocks"]["sdsc"][0]
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_block", checked)
    monkeypatch.setattr(sys, "argv", ["runner", "--output-dir", str(tmp_path), "--workers", "1"])
    module.main()
    assert len(list(tmp_path.glob("*.json"))) == 6
    expected_tapes = 1 + len(module.VARIANTS) * config.replications
    expected_runs = expected_tapes * len(module.POLICIES) * 4  # 4 origin blocks (2 sources x 2 fractions)
    assert json.loads((tmp_path / "manifest.json").read_text())["scheduler_runs"] == expected_runs
