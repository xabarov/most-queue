"""Replay held-out serving cohorts with fixed isolated-request coefficients.

Server queue-entry timestamps define arrivals. Client start times cannot replace
them: HTTP and frontend delay are outside the scheduler's queue. This compares
the specified PrefillDecodeSim policy with observations, not an exact vLLM model.
"""

import numpy as np

from most_queue.sim.prefill_decode import PrefillDecodeSim


def _metric(observed, predicted, *, queue=False):
    observed, predicted = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    if not np.all(np.isfinite(observed)) or np.any(observed < 0):
        raise ValueError("complete finite nonnegative server measurements are required")
    actual_mean, predicted_mean = float(observed.mean()), float(predicted.mean())
    absolute_mean_error = abs(predicted_mean - actual_mean)
    return {
        "observed_mean_seconds": actual_mean,
        "predicted_mean_seconds": predicted_mean,
        "absolute_mean_error_seconds": absolute_mean_error,
        "relative_mean_error": absolute_mean_error / actual_mean if actual_mean > 0 else None,
        "normalized_mean_error": absolute_mean_error / max(actual_mean, 0.05) if queue else None,
        "mae_seconds": float(np.abs(observed - predicted).mean()),
        "observed": observed.tolist(),
        "predicted": predicted.tolist(),
        "observed_quantiles_seconds": np.percentile(observed, [50, 90, 95]).tolist(),
        "predicted_quantiles_seconds": np.percentile(predicted, [50, 90, 95]).tolist(),
    }


def replay_serving_cohort(fit, benchmark, *, batch_cap=4, token_budget=18432):
    """Evaluate one complete finite cohort from an empty server without refitting.

    Requires aligned ``server_arrival_times`` metadata and queue-entry-to-first
    server TTFT. Results retain original request order even if engine arrival
    order differs from client order. Client TTFT adds the training-only offset;
    queue wait and TPOT come directly from DES. The 50-ms queue denominator floor
    avoids unstable relative errors near zero; absolute errors remain available.
    """
    metadata, rows = benchmark.metadata, benchmark.requests
    if (
        metadata.get("measurement_profile") != fit.measurement_profile
        or metadata.get("latency_basis") != fit.latency_basis
    ):
        raise ValueError("held-out profile and latency definition must match training")
    if metadata.get("server_ttft_basis") != "queue_entry_to_first_token":
        raise ValueError("server TTFT must start at queue entry")
    if any(row.status != "completed" or row.output_tokens <= 1 for row in rows):
        raise ValueError("replay requires complete successful multitoken cohorts; audit failures separately")
    costs = np.asarray([fit.client_offset, fit.alpha, fit.beta, fit.gamma], dtype=float)
    if not np.all(np.isfinite(costs)) or np.any(costs < 0) or fit.alpha <= 0:
        raise ValueError("replay requires positive alpha and finite nonnegative fitted costs")
    recorded = metadata.get("server_arrival_times")
    if not isinstance(recorded, (list, tuple)) or len(recorded) != len(rows):
        raise ValueError("aligned exact server_arrival_times are required; client arrivals are not a substitute")
    if any(value is None or isinstance(value, bool) for value in recorded):
        raise ValueError("every exact server arrival must be present")
    arrivals = np.asarray(recorded, dtype=float)
    if arrivals.ndim != 1 or not np.all(np.isfinite(arrivals)) or np.any(arrivals < 0):
        raise ValueError("server arrivals must be finite and nonnegative")
    observed = {
        "client_ttft": [row.ttft for row in rows],
        "server_ttft": [row.server_ttft for row in rows],
        "server_tpot": [row.server_tpot for row in rows],
        "server_queue_wait": [row.server_queue_time for row in rows],
    }
    if any(value is None for values in observed.values() for value in values):
        raise ValueError("complete server timings are required; missing values cannot be replaced with zero")
    order = np.argsort(arrivals, kind="stable")
    inverse = np.argsort(order)
    ordered_arrivals = arrivals[order] - arrivals.min()
    inputs = [rows[index].input_tokens for index in order]
    outputs = [rows[index].output_tokens for index in order]
    simulator = PrefillDecodeSim(batch_cap=batch_cap, token_budget=token_budget)
    simulator.set_servers(inputs[0], outputs[0], fit.alpha, fit.beta, fit.gamma)
    result = simulator.replay(ordered_arrivals, inputs, outputs)
    first_tokens = np.asarray([times[0] for times in result.output_times])
    last_tokens = np.asarray([times[-1] for times in result.output_times])
    server_ttft = (first_tokens - ordered_arrivals)[inverse]
    predictions = {
        "client_ttft": server_ttft + fit.client_offset,
        "server_ttft": server_ttft,
        "server_tpot": ((last_tokens - first_tokens) / (np.asarray(outputs) - 1))[inverse],
        "server_queue_wait": (np.asarray(result.start_times) - ordered_arrivals)[inverse],
    }
    return {
        "requests": len(rows),
        "scope": "finite cohort from empty server; fixed isolated coefficients; specified DES policy",
        "measurement_profile": fit.measurement_profile,
        "batch_cap": batch_cap,
        "token_budget": token_budget,
        "arrival_basis": "exact server queue entry",
        "request_indices": metadata.get("request_indices", list(range(len(rows)))),
        "metrics": {
            name: _metric(values, predictions[name], queue=name == "server_queue_wait")
            for name, values in observed.items()
        },
        "outside_training_length_range": sum(
            not (
                fit.input_range[0] <= row.input_tokens <= fit.input_range[1]
                and fit.output_range[0] <= row.output_tokens <= fit.output_range[1]
            )
            for row in rows
        ),
    }
