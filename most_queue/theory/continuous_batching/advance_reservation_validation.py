"""Finite-cohort replay and joint client SLO diagnostics, with fixed coefficients."""

import numpy as np

from most_queue.io.serving import summarize_serving_slo
from most_queue.sim.advance_reservation import AdvanceReservationSim
from most_queue.theory.continuous_batching.serving_validation import _metric


def replay_advance_reservation(fit, benchmark, *, batch_cap=4, token_budget=18432):
    """Replay exact engine arrivals; observed scheduling/token times are outcomes.

    Admission reserves a slot immediately if available. GPU execution begins
    at the next boundary. Client TTFT adds the isolated residual; client TPOT
    approximates server TPOT (equal residual at first and last output).
    This is a finite drained cohort, not a steady-state or SLO guarantee.
    """
    metadata, rows = benchmark.metadata, benchmark.requests
    if (
        metadata.get("measurement_profile") != fit.measurement_profile
        or metadata.get("latency_basis") != fit.latency_basis
    ):
        raise ValueError("evaluation profile and latency definition must match training")
    if metadata.get("server_ttft_basis") != "queue_entry_to_first_token":
        raise ValueError("server TTFT must start at queue entry")
    if any(r.status != "completed" or r.output_tokens <= 1 for r in rows):
        raise ValueError("replay requires complete successful multitoken cohorts; audit failures separately")
    costs = np.asarray([fit.alpha, fit.beta, fit.gamma, fit.client_intercept, fit.client_per_input_token])
    if not np.all(np.isfinite(costs)) or np.any(costs < 0) or fit.alpha <= 0:
        raise ValueError("finite nonnegative fitted costs and positive alpha are required")
    recorded = metadata.get("server_arrival_times")
    if not isinstance(recorded, (list, tuple)) or len(recorded) != len(rows):
        raise ValueError("aligned exact server_arrival_times are required")
    if any(v is None or isinstance(v, bool) for v in recorded):
        raise ValueError("every exact server arrival must be present")
    arrivals = np.asarray(recorded, dtype=float)
    if arrivals.ndim != 1 or not np.all(np.isfinite(arrivals)) or np.any(arrivals < 0):
        raise ValueError("server arrivals must be finite and nonnegative")
    observed = {
        "client_ttft": [r.ttft for r in rows],
        "client_tpot": [r.tpot for r in rows],
        "server_ttft": [r.server_ttft for r in rows],
        "server_tpot": [r.server_tpot for r in rows],
        "server_queue_wait": [r.server_queue_time for r in rows],
        "server_scheduled_to_first_token": [r.server_scheduled_to_first_token for r in rows],
    }
    if any(v is None for values in observed.values() for v in values):
        raise ValueError("complete server timings are required; missing values cannot be replaced with zero")
    order = np.argsort(arrivals, kind="stable")
    inverse = np.argsort(order)
    ordered = arrivals[order] - arrivals.min()
    inputs, outputs = [rows[i].input_tokens for i in order], [rows[i].output_tokens for i in order]
    sim = AdvanceReservationSim(batch_cap=batch_cap, token_budget=token_budget)
    sim.set_servers(inputs[0], outputs[0], fit.alpha, fit.beta, fit.gamma)
    result = sim.replay(ordered, inputs, outputs)
    first = np.asarray([t[0] for t in result.output_times])
    last = np.asarray([t[-1] for t in result.output_times])
    server_ttft = (first - ordered)[inverse]
    tpot = ((last - first) / (np.asarray(outputs) - 1))[inverse]
    predictions = {
        "client_ttft": server_ttft + fit.client_residual([r.input_tokens for r in rows]),
        "client_tpot": tpot,
        "server_ttft": server_ttft,
        "server_tpot": tpot,
        "server_queue_wait": (np.asarray(result.start_times) - ordered)[inverse],
        "server_scheduled_to_first_token": (first - np.asarray(result.start_times))[inverse],
    }
    return {
        "requests": len(rows),
        "scope": "finite empty-server cohort; advance reservation approximation; fixed isolated coefficients",
        "measurement_profile": fit.measurement_profile,
        "arrival_basis": "exact server queue entry",
        "batch_cap": batch_cap,
        "token_budget": token_budget,
        "request_indices": metadata.get("request_indices", list(range(len(rows)))),
        "metrics": {k: _metric(v, predictions[k], queue=k == "server_queue_wait") for k, v in observed.items()},
        "reservation_to_compute_seconds": (np.asarray(result.first_compute_times) - np.asarray(result.start_times))[
            inverse
        ].tolist(),
        "outside_training_length_range": sum(
            not (
                fit.input_range[0] <= r.input_tokens <= fit.input_range[1]
                and fit.output_range[0] <= r.output_tokens <= fit.output_range[1]
            )
            for r in rows
        ),
    }


def compare_joint_slo(benchmark, replay, *, ttft_limit, tpot_limit):
    """Compare paired client TTFT/TPOT pass fractions, without an iid CI.

    No predicted goodput is inferred from the observed measurement duration.
    Missing or failed outcomes cannot silently disappear from this paired
    comparison. Use summarize_serving_slo separately to report all offers when
    replay is inapplicable. One-token replies are outside this replay API.
    """
    actual = summarize_serving_slo(benchmark, ttft_limit=ttft_limit, tpot_limit=tpot_limit)
    if actual["completed"] != actual["offered"] or actual["single_token_completed"]:
        raise ValueError("paired replay requires complete successful multitoken cohorts")
    count = len(benchmark.requests)
    masks = []
    for metric, limit in (("client_ttft", ttft_limit), ("client_tpot", tpot_limit)):
        report = replay["metrics"][metric]
        predicted, observed = np.asarray(report["predicted"]), np.asarray(report["observed"])
        values = np.asarray([r.ttft if metric == "client_ttft" else r.tpot for r in benchmark.requests])
        if predicted.shape != (count,) or observed.shape != (count,) or not np.array_equal(values, observed):
            raise ValueError("replay and benchmark requests must align")
        if not np.all(np.isfinite(predicted)) or np.any(predicted < 0):
            raise ValueError("predicted client metrics must be finite and nonnegative")
        masks.append(predicted <= limit)
    predicted_pass = masks[0] & masks[1]
    actual_pass = np.asarray([r.ttft <= ttft_limit and r.tpot <= tpot_limit for r in benchmark.requests])
    predicted_fraction = float(predicted_pass.mean())
    observed_fraction = actual["joint_passed"] / count
    return {
        "offered": count,
        "ttft_limit_seconds": ttft_limit,
        "tpot_limit_seconds": tpot_limit,
        "observed_passed": actual["joint_passed"],
        "predicted_passed": int(predicted_pass.sum()),
        "observed_fraction": observed_fraction,
        "predicted_fraction": predicted_fraction,
        "absolute_fraction_error": abs(predicted_fraction - observed_fraction),
        "false_passes": int(np.sum(predicted_pass & ~actual_pass)),
        "false_failures": int(np.sum(~predicted_pass & actual_pass)),
        "scope": "paired empirical client SLO; no stationarity, capacity or confidence guarantee",
    }
