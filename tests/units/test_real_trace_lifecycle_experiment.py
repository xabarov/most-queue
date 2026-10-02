"""Fixed-cohort sensitivity checks, using only synthetic lifecycle records."""

import json
from dataclasses import replace

import numpy as np
import pytest

from examples import real_trace_lifecycle_experiment as module
from examples.real_trace_temporal_experiment import TemporalConfig, fit_context, prepare_origins
from most_queue.sim.utils.workload_lifecycle import parse_swf_lifecycle


def source_fixture():
    rows = []
    for i in range(400):
        rows.append(f"{i*2+1} {i*5} 0 {2+i%6} {1+i%2} -1 -1 {1+i%2} 4 -1 1 -1 -1 -1 -1 -1 -1 -1")
        if i % 7 == 0:
            rows.append(f"{i*2+2} {i*5+1} 0 2 1 -1 -1 1 4 -1 5 -1 -1 -1 -1 -1 -1 -1")
    return parse_swf_lifecycle(rows, 4)


def setup():
    source = source_fixture()
    config = TemporalConfig(fractions=(0.4,), jobs=20, warmup=5, replications=2)
    prepared = prepare_origins(source.completed, config)[0]
    models, _ = fit_context(*prepared[:2], config)
    forecasts = {k: models["expanding_coarse"].distribution(k)[0].parameters()["p90"] for k in range(1, 5)}
    cohort = prepared[2]
    services = np.array([j.runtime for j in cohort])
    return source, config, prepared, forecasts, services


def test_target_ids_do_not_shift_when_cancelled_competitors_are_inserted():
    source, config, prepared, forecasts, services = setup()
    cases = [
        module.build_case(source, prepared[2], services, forecasts, scenario, warmup=config.warmup)
        for scenario in module.SCENARIOS
    ]
    assert [len(c.target_indices) for c in cases] == [20] * 4
    assert cases[0].audit["cancelled_arrivals"] == 0 < cases[2].audit["cancelled_arrivals"]
    for case in cases:
        offset = len(case.running) + len(case.waiting)
        targets = [case.arrivals[i - offset] for i in case.target_indices]
        np.testing.assert_array_equal([j.service for j in targets], services[5:])
        np.testing.assert_array_equal(
            [j.arrival for j in targets], [j.submit - prepared[2][0].submit for j in prepared[2][5:]]
        )
    assert cases[2].running == cases[3].running and cases[2].waiting == cases[3].waiting
    assert all(j.runtime_limit is None for j in cases[2].arrivals)
    assert all(j.runtime_limit == 4 for j in cases[3].arrivals)


def test_forecasts_and_historical_environment_do_not_leak_generated_target_service():
    source, config, prepared, forecasts, services = setup()
    baseline = module.build_case(source, prepared[2], services, forecasts, "carry_cancelled", warmup=config.warmup)
    changed = module.build_case(
        source, prepared[2], services * 1000, forecasts, "carry_cancelled", warmup=config.warmup
    )
    assert baseline.running == changed.running and baseline.waiting == changed.waiting
    assert [j.estimate for j in baseline.arrivals] == [j.estimate for j in changed.arrivals]
    assert [j for j in baseline.arrivals if j.outcome == "cancelled"] == [
        j for j in changed.arrivals if j.outcome == "cancelled"
    ]


def test_experiment_determinism_strict_json_and_independent_summary_means():
    source, config, prepared, _, _ = setup()
    result = module.run_origin(source, prepared, config)
    assert result == module.run_origin(source, prepared, config)
    json.dumps(result, allow_nan=False)
    assert result["scheduler_runs"] == 4 * (1 + 2 * 2) * 6
    for row in result["rows"]:
        assert sum(g["count"] for g in row["groups"]) == row["target_count"] == 20
        assert row["success_rate"] + row["timeout_rate"] == 1
        assert row["mean_release_t"] - row["mean_w"] == pytest.approx(row["target_consumed_service_mean"])
    for summary in result["summaries"]:
        rows = [r for r in result["rows"] if all(r[k] == summary[k] for k in ("scenario", "model", "policy"))]
        assert summary["metrics"]["mean_release_t"]["mean"] == np.mean([r["mean_release_t"] for r in rows])
    assert len(module.aggregate([result])) == 8


def test_all_timeouts_have_no_success_latency_and_remain_in_fixed_target_denominator():
    source, config, prepared, forecasts, services = setup()
    source = replace(source, jobs=tuple(replace(j, requested_time=0.5) for j in source.jobs))
    case = module.build_case(source, prepared[2], services, forecasts, "requested_limit", warmup=config.warmup)
    result = module.run_case(case, 4, "fcfs")
    assert result["success_rate"] == 0 and result["timeout_rate"] == 1
    assert result["mean_success_t"] is None and result["p99_success_t"] is None
    assert result["target_count"] == 20 and result["successful_target_count"] == 0
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("services", [[1], [float("nan")] * 25, [-1] * 25])
def test_invalid_service_vectors(services):
    source, config, prepared, forecasts, _ = setup()
    with pytest.raises(ValueError):
        module.build_case(source, prepared[2], services, forecasts, "empty_completed", warmup=config.warmup)


def test_invalid_case_contracts():
    source, config, prepared, forecasts, services = setup()
    for scenario, warmup in (("bogus", 5), ("empty_completed", True), ("empty_completed", 25)):
        with pytest.raises(ValueError):
            module.build_case(source, prepared[2], services, forecasts, scenario, warmup=warmup)
    missing = replace(source, jobs=tuple(j for j in source.jobs if j.job_id != prepared[2][0].job_id))
    with pytest.raises(ValueError, match="target IDs"):
        module.build_case(missing, prepared[2], services, forecasts, "empty_completed", warmup=config.warmup)
