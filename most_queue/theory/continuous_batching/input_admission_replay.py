"""Development-only finite replay of specified input admission approximations.

Policy motivation is pinned to vLLM v0.32.0, not an exact engine emulator:
https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/engine/core.py
https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/core/sched/scheduler.py
https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/worker/gpu_model_runner.py
"""

from numbers import Real

import numpy as np

from most_queue.sim.advance_reservation import AdvanceReservationSim
from most_queue.sim.iteration_admission import IterationAdmissionSim
from most_queue.theory.continuous_batching.advance_reservation_validation import (
    compare_joint_slo,
    replay_advance_reservation,
)
from most_queue.theory.continuous_batching.serving_validation import _metric


def _vector(values, count, name):
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f"aligned {name} vector required")
    raw = np.asarray(values, dtype=object)
    if raw.shape != (count,) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw.flat):
        raise ValueError(f"aligned finite numeric {name} required")
    result = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"finite nonnegative {name} required")
    return result


def _observations(benchmark):
    rows = benchmark.requests
    observed = {
        "client_ttft": [r.ttft for r in rows],
        "client_tpot": [r.tpot for r in rows],
        "server_ttft": [r.server_ttft for r in rows],
        "server_tpot": [r.server_tpot for r in rows],
        "server_queue_wait": [r.server_queue_time for r in rows],
        "server_scheduled_to_first_token": [r.server_scheduled_to_first_token for r in rows],
    }
    observed = {name: _vector(values, len(rows), name) for name, values in observed.items()}
    observed["outside_engine"] = observed["client_ttft"] - observed["server_ttft"]
    return observed


def _predictions(
    sim, fit, ready, clients, inputs, outputs, delivery
):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    order = np.argsort(ready, kind="stable")
    inverse = np.argsort(order)
    sim.set_servers(int(inputs[order[0]]), int(outputs[order[0]]), fit.alpha, fit.beta, fit.gamma)
    result = sim.replay(ready[order], inputs[order].tolist(), outputs[order].tolist())
    first = np.asarray([times[0] for times in result.output_times])[inverse]
    last = np.asarray([times[-1] for times in result.output_times])[inverse]
    start = np.asarray(result.start_times)[inverse]
    compute = np.asarray(result.first_compute_times)[inverse]
    entry = np.asarray(result.engine_entry_times)[inverse] if hasattr(result, "engine_entry_times") else ready.copy()
    wait = entry - ready
    tpot = (last - first) / (outputs - 1)
    predictions = {
        "client_ttft": first + delivery - clients,
        "client_tpot": tpot,
        "server_ttft": first - entry,
        "server_tpot": tpot,
        "server_queue_wait": start - entry,
        "server_scheduled_to_first_token": first - start,
        "prequeue": entry - clients,
        "outside_engine": entry - clients + delivery,
    }
    timing = {
        "ready": ready,
        "engine_entry": entry,
        "reservation": start,
        "first_compute": compute,
        "first_output": first,
        "last_output": last,
        "input_wait": wait,
    }
    return predictions, {name: values.tolist() for name, values in timing.items()}


def replay_input_admission(server_fit, input_fit, benchmark, *, batch_cap=4, token_budget=18432):
    """Compare five fixed policies, using client clocks and lengths for new predictions.

    All timing arrays use the minimum client arrival as their common origin and
    retain benchmark order. Observations are used only as evaluation targets;
    the historical baseline alone consumes observed engine arrivals. This API
    neither fits coefficients nor identifies physical frontend components.
    """
    rows, metadata = benchmark.requests, benchmark.metadata
    count = len(rows)
    if not count or any(r.status != "completed" or r.output_tokens <= 1 for r in rows):
        raise ValueError("complete successful multitoken cohort required")
    clients = _vector(metadata.get("arrival_times"), count, "arrival_times")
    engines = _vector(metadata.get("server_arrival_times"), count, "server_arrival_times")
    if np.any(engines < clients):
        raise ValueError("server arrival cannot precede client arrival")
    observed = _observations(benchmark)
    observed["prequeue"] = engines - clients
    for name, values in observed.items():
        _vector(values, count, name)
    baseline = replay_advance_reservation(server_fit, benchmark, batch_cap=batch_cap, token_budget=token_budget)
    inputs = np.asarray([r.input_tokens for r in rows])
    outputs = np.asarray([r.output_tokens for r in rows])
    _vector(
        [
            input_fit.ready_intercept,
            input_fit.ready_per_input_token,
            input_fit.delivery_intercept,
            input_fit.delivery_per_input_token,
        ],
        4,
        "input timing coefficients",
    )
    ready_delay = _vector(input_fit.ready_delay(inputs), count, "ready delay")
    delivery = _vector(input_fit.delivery_delay(inputs), count, "delivery delay")
    clients = clients - clients.min()
    ready = clients + ready_delay
    baseline["metrics"]["outside_engine"] = _metric(observed["outside_engine"], server_fit.client_residual(inputs))
    variants = {"baseline_v2": baseline}
    policies = {
        "continuous_reservation": AdvanceReservationSim(batch_cap=batch_cap, token_budget=token_budget),
        "input_one_batch": IterationAdmissionSim(
            batch_cap=batch_cap, token_budget=token_budget, inflight_batches=1, poll_inputs=True
        ),
        "two_batches_continuous_input": IterationAdmissionSim(
            batch_cap=batch_cap, token_budget=token_budget, inflight_batches=2, poll_inputs=False
        ),
        "input_two_batches": IterationAdmissionSim(
            batch_cap=batch_cap, token_budget=token_budget, inflight_batches=2, poll_inputs=True
        ),
    }
    indices = list(metadata.get("request_indices", range(count)))
    if len(indices) != count:
        raise ValueError("request_indices must align")
    for name, sim in policies.items():
        predicted, timing = _predictions(sim, server_fit, ready, clients, inputs, outputs, delivery)
        variants[name] = {
            **{
                key: baseline[key]
                for key in (
                    "requests",
                    "scope",
                    "measurement_profile",
                    "batch_cap",
                    "token_budget",
                    "outside_training_length_range",
                )
            },
            "scope": "finite empty-server cohort; specified input admission approximation; development only",
            "arrival_basis": "client arrival plus fixed predicted ready delay",
            "request_indices": indices.copy(),
            "metrics": {
                key: _metric(values, predicted[key], queue=key == "server_queue_wait")
                for key, values in observed.items()
            },
            "timing": timing,
            "reservation_to_compute_seconds": (
                np.asarray(timing["first_compute"]) - np.asarray(timing["reservation"])
            ).tolist(),
        }
    for variant in variants.values():
        variant["joint_client_slo"] = [
            compare_joint_slo(benchmark, variant, ttft_limit=ttft, tpot_limit=tpot)
            for ttft in (0.25, 0.5, 1.0)
            for tpot in (0.03, 0.035, 0.04)
        ]
    return {
        "requests": count,
        "request_indices": indices,
        "scope": "finite development cohort; fixed coefficients; no exact engine or capacity claim",
        "data_role": "development",
        "primary_variant": "input_two_batches",
        "variants": variants,
    }
