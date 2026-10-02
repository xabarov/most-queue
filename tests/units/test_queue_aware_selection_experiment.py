"""Synthetic temporal, selection, pairing and regression checks; no downloads."""

import copy
import json
import sys
from dataclasses import replace

import numpy as np
import pytest

from examples import queue_aware_selection_experiment as module
from tests.units.test_modern_gpu_trace_experiment import setup as gpu_setup
from tests.units.test_real_trace_lifecycle_experiment import source_fixture


def setup(name="sdsc"):
    source = source_fixture() if name == "sdsc" else gpu_setup()[0]
    config = module.QueueSelectionConfig((0.2, 0.4), (0.65, 0.8), validation_jobs=20, jobs=20, warmup=5, replications=2)
    return source, config, module.prepare(source, name, config)


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_temporal_contract_completed_history_and_disjoint_cohorts(name):
    _, config, blocks = setup(name)
    ids = set()
    for b in blocks:
        assert all(j.completed_at < b["split"]["cutoff"] for j in b["history"])
        assert len(b["cohort"]) == 25
        assert not ids.intersection(j.job_id for j in b["cohort"])
        ids.update(j.job_id for j in b["cohort"])
        if b["phase"] == "validation":
            assert b["environment_max_completion"] < blocks[2]["split"]["cutoff"]
    assert config.validation_seed + config.replications <= config.test_seed


def test_unfinished_validation_companion_rejects_configuration():
    source, config, blocks = setup()
    boundary = blocks[0]["cohort"][0].submit
    companion = next(j for j in source.jobs if j.status == 5 and boundary <= j.submit <= blocks[0]["cohort"][-1].submit)
    changed = replace(
        source, jobs=tuple(replace(j, runtime=1e6) if j.job_id == companion.job_id else j for j in source.jobs)
    )
    with pytest.raises(ValueError, match="environment must finish"):
        module.prepare(changed, "sdsc", config)


def test_unfinished_validation_target_rejects_configuration():
    source, config, blocks = setup()
    target = blocks[0]["cohort"][0]
    changed = replace(
        source, jobs=tuple(replace(j, runtime=1e6) if j.job_id == target.job_id else j for j in source.jobs)
    )
    with pytest.raises(ValueError, match="environment must finish"):
        module.prepare(changed, "sdsc", config)


def test_short_or_overlapping_blocks_rejected_without_shrinking():
    source, config, _ = setup()
    with pytest.raises(ValueError, match="full cohorts"):
        module.prepare(source, "sdsc", replace(config, jobs=2000))
    with pytest.raises(ValueError, match="overlap"):
        module.prepare(source, "sdsc", replace(config, validation_fractions=(0.2, 0.21)))


@pytest.mark.parametrize("name", ["sdsc", "kalos"])
def test_full_small_protocol_determinism_metrics_selection_and_freeze(name):
    source, config, blocks = setup(name)
    validation = [module.run_block(source, name, config, b) for b in blocks[:2]]
    selection = module.select(validation, name)
    assert validation[0] == module.run_block(source, name, config, blocks[0])
    assert all(set(r["scenario"] for r in v["rows"]) == {module.main_scenario(name)} for v in validation)
    reference, predictions = module.metric_arrays(validation, name)
    assert selection["queue"] == module.select_queue_model(
        predictions, reference, candidate_order=module.feature.CANDIDATES[name]
    )
    assert len(selection["leave_one_origin_out"]) == len(selection["leave_one_seed_out"]) == 2
    assert selection["crps"]["selected"] == min(
        module.feature.CANDIDATES[name], key=lambda v: np.mean([r["scores"][v]["crps"] for r in validation])
    )
    frozen = copy.deepcopy(selection)
    result = module.run_block(source, name, config, blocks[2])
    assert result["scheduler_runs"] == 42 * (2 if name == "sdsc" else 1)
    assert {key: result[key] for key in ("summaries", "contrasts", "decisions")} == module.feature.summarize(
        result["rows"], name
    )
    for row in result["rows"]:
        assert row["target_count"] == 20
        assert row["mean_release_t"] - row["mean_w"] == pytest.approx(row["target_consumed_service_mean"])
        assert sum(g["count"] for g in row["groups"]) == 20
        assert sum(row["resource_time_by_outcome"].values()) == row["total_resource_time"]
        assert row["success_rate"] + row["timeout_rate"] == 1
    evaluation = module.evaluate_selection(result, selection)
    for row in evaluation["selection_evaluation"]:
        for method, interval in row["paired_queue_minus"].items():
            changes = np.array(row["seed_losses"]["queue"]) - row["seed_losses"][method]
            assert interval == module.feature.swf.interval(changes)
    assert selection == frozen == module.select(validation, name)
    json.dumps({"selection": selection, "test": result, "evaluation": evaluation}, allow_nan=False)


def test_test_artifacts_cannot_be_passed_to_selector():
    source, config, blocks = setup()
    validation = [module.run_block(source, "sdsc", config, b) for b in blocks[:2]]
    with pytest.raises(ValueError, match="validation blocks"):
        module.select([validation[0], dict(validation[1], phase="test")], "sdsc")
    with pytest.raises(ValueError, match="distinct"):
        module.select([validation[0], validation[0]], "sdsc")
    with pytest.raises(ValueError, match="test block"):
        module.evaluate_selection(validation[0], {})


def test_late_source_outcomes_do_not_change_validation():
    source, config, blocks = setup()
    cutoff = blocks[2]["split"]["cutoff"]
    changed = replace(
        source, jobs=tuple(replace(j, runtime=j.runtime * 3) if j.submit >= cutoff else j for j in source.jobs)
    )
    changed_blocks = module.prepare(changed, "sdsc", config)
    for a, b in zip(blocks[:2], changed_blocks[:2]):
        assert a == b
        assert module.run_block(source, "sdsc", config, a) == module.run_block(changed, "sdsc", config, b)


def test_parallel_and_serial_rows_are_identical():
    source, config, blocks = setup()
    assert module.run_block(source, "sdsc", config, blocks[0], workers=2) == module.run_block(
        source, "sdsc", config, blocks[0]
    )


def test_support_drift_is_marginal_and_empty_groups_have_no_support():
    _, config, blocks = setup()
    h, jobs = blocks[0]["history"], blocks[0]["cohort"]
    model = module.feature.fit_models(h, "sdsc", config.minimum)["coarse"]
    details = module.coverage(h, jobs, "sdsc", model)
    assert 0 <= details["need_total_variation"] <= 1
    assert details["context_total_variation"] == 0
    assert sum(g["target_count"] for g in details["groups"]) == len(jobs)
    for group in details["groups"][2:]:
        assert group["target_count"] == 0 and group["support_min"] is None
    changed = [replace(j, need=64, requested_time=1e6) for j in jobs]
    details = module.coverage(h, changed, "sdsc", model)
    assert details["need_total_variation"] == details["context_total_variation"] == 1
    assert details["unseen_context_targets"] == len(jobs)
    assert details["groups"][3]["levels"] == {"pooled": len(jobs)}


def test_queue_and_crps_can_select_different_candidates():
    source, config, blocks = setup()
    validation = [module.run_block(source, "sdsc", config, b) for b in blocks[:2]]
    for result in validation:
        for row in result["rows"]:
            if row["variant"] != "observed":
                ref = next(r for r in result["rows"] if r["variant"] == "observed" and r["policy"] == row["policy"])
                for metric in module.QUEUE_METRICS:
                    row[metric] = ref[metric] * {"coarse": 2, "request_bin": 1, "request_ratio": 3}[row["variant"]]
        for variant, score in result["scores"].items():
            score["crps"] = 0 if variant == "request_ratio" else 1
    selection = module.select(validation, "sdsc")
    assert selection["queue"]["selected"] == "request_bin"
    assert selection["crps"]["selected"] == "request_ratio"
    assert all(s["selected"] == "request_bin" for s in selection["leave_one_seed_out"])


@pytest.mark.parametrize("duplicate", [False, True])
def test_incomplete_or_duplicate_grid_rejected(duplicate):
    source, config, blocks = setup()
    validation = [module.run_block(source, "sdsc", config, b) for b in blocks[:2]]
    if duplicate:
        validation[0]["rows"].append(validation[0]["rows"][-1])
    else:
        validation[0]["rows"].pop()
    with pytest.raises(ValueError, match="grid"):
        module.select(validation, "sdsc")


def test_cli_persists_both_selections_before_any_test_scoring(tmp_path, monkeypatch):
    sdsc, config, _ = setup()
    kalos = setup("kalos")[0]
    monkeypatch.setattr(module, "CONFIGS", {"sdsc": config, "kalos": config})
    monkeypatch.setattr(module.feature.swf, "verified_source", lambda *args: b"synthetic")
    monkeypatch.setattr(module.feature.gpu, "verified_source", lambda *args: b"synthetic")
    monkeypatch.setattr(module.feature, "parse_swf_lifecycle", lambda *args: sdsc)
    monkeypatch.setattr(module.feature, "parse_acme_kalos", lambda *args: kalos)
    original, phases = module.run_block, []

    def checked(source, name, config, block, **kwargs):
        phases.append(block["phase"])
        if block["phase"] == "test":
            saved = json.loads((tmp_path / "selection.json").read_text())
            assert set(saved) == {"sdsc", "kalos"}
            assert len(list(tmp_path.glob("*-validation-*.json"))) == 4
        else:
            assert not (tmp_path / "selection.json").exists()
        return original(source, name, config, block, **kwargs)

    monkeypatch.setattr(module, "run_block", checked)
    monkeypatch.setattr(sys, "argv", ["runner", "--output-dir", str(tmp_path), "--workers", "1"])
    module.main()
    assert phases == ["validation"] * 4 + ["test"] * 4
    assert len(list(tmp_path.glob("*.json"))) == 10
    assert json.loads((tmp_path / "manifest.json").read_text())["scheduler_runs"] == 420


@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_fractions": (0.7,)},
        {"validation_fractions": (0.6, 0.6)},
        {"test_fractions": ()},
        {"test_fractions": (0.5,)},
        {"test_fractions": (1,)},
        {"test_fractions": (np.nan,)},
        {"validation_seed": 62000},
        {"replications": 1},
        {"minimum": 1},
        {"history_limit": 1},
        {"warmup": True},
        {"jobs": 0},
        {"validation_jobs": -1},
    ],
)
def test_invalid_configs(kwargs):
    with pytest.raises(ValueError):
        module.QueueSelectionConfig(**kwargs)
