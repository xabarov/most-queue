"""Server-stage calibration for the specified AdvanceReservationSim policy.

The first output token belongs to the completed prefill. Fit server durations
separately from the client residual; these are effective costs, not GPU kernel
measurements. See docs/research/serving-v2-model-audit.md for pinned sources,
equations, development data, and limitations of advance reservation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from most_queue.theory.continuous_batching.serving_calibration import _validate_profile


def _design(inputs, outputs):
    n, m = np.asarray(inputs, dtype=float), np.asarray(outputs, dtype=float)
    if n.ndim != 1 or m.shape != n.shape or not n.size:
        raise ValueError("length vectors must align and be nonempty")
    if any(not np.all(np.isfinite(v)) or np.any(v < 1) or np.any(v != np.floor(v)) for v in (n, m)):
        raise ValueError("token lengths must be positive finite integers")
    ones = np.ones(len(n))
    return np.column_stack((ones, n, n)), np.column_stack((ones, ones, n + m / 2)), np.column_stack((ones, n))


def _solve(design, target):
    norms = np.linalg.norm(design, axis=0)
    if np.any(norms == 0):
        raise ValueError("rank-deficient calibration: vary prompt and output lengths")
    scaled = design / norms
    if np.linalg.matrix_rank(scaled) < scaled.shape[1]:
        raise ValueError("rank-deficient calibration: vary prompt and output lengths")
    condition = float(np.linalg.cond(scaled))
    if condition > 1e8:
        raise ValueError("calibration design is ill-conditioned")
    return nnls(scaled, target)[0] / norms, condition


@dataclass(frozen=True)
class AdvanceReservationFit:
    """Nonnegative server coefficients and a separate affine client residual."""

    alpha: float
    beta: float
    gamma: float
    client_intercept: float
    client_per_input_token: float
    server_scaled_condition: float
    client_scaled_condition: float
    training_requests: int
    input_range: tuple[int, int]
    output_range: tuple[int, int]
    measurement_profile: str
    latency_basis: str

    def client_residual(self, inputs):
        """Return the isolated-training client-minus-server TTFT approximation."""
        _, _, client = _design(inputs, np.ones(len(inputs)))
        return client @ [self.client_intercept, self.client_per_input_token]

    def predict(self, inputs, outputs):
        """Predict isolated server TTFT/TPOT and client TTFT; no queue prediction."""
        prefill, decode, _ = _design(inputs, outputs)
        costs = [self.alpha, self.beta, self.gamma]
        server_ttft = prefill @ costs
        tpot = decode @ costs
        tpot[np.asarray(outputs) <= 1] = np.nan
        return {
            "server_ttft": server_ttft,
            "server_tpot": tpot,
            "client_ttft": server_ttft + self.client_residual(inputs),
        }


def fit_advance_reservation(benchmarks):
    """Fit server costs and client residual using isolated training cohorts only.

    Server equations: F(n)=alpha+(beta+gamma)*n and
    T(n,m)=alpha+beta+gamma*(n+m/2), m>1. NNLS divides each metric by
    its training mean, then normalizes columns. Client-minus-server TTFT is
    independently fitted as d0+d1*n. Reject missing clocks, mixed profiles,
    incomplete outcomes and unidentifiable designs; never infer queue times.
    """
    benchmarks = list(benchmarks)
    profile, basis = _validate_profile(benchmarks)
    if any(b.metadata.get("server_ttft_basis") != "queue_entry_to_first_token" for b in benchmarks):
        raise ValueError("server TTFT must start at queue entry")
    rows = [r for b in benchmarks for r in b.requests]
    if any(r.status != "completed" for r in rows):
        raise ValueError("calibration requires complete successful cohorts; audit failures separately")
    inputs, outputs = [r.input_tokens for r in rows], [r.output_tokens for r in rows]
    prefill, decode, client = _design(inputs, outputs)
    if any(
        r.server_scheduled_to_first_token is None or r.server_ttft is None or r.server_queue_time is None for r in rows
    ):
        raise ValueError("complete server timings are required")
    if any(
        not np.isclose(r.server_ttft, r.server_queue_time + r.server_scheduled_to_first_token, atol=1e-9, rtol=1e-7)
        for r in rows
    ):
        raise ValueError("server TTFT must equal queue plus scheduled-to-first-token time")
    multi = np.asarray(outputs) > 1
    if not np.any(multi) or any(r.server_tpot is None for r in rows if r.output_tokens > 1):
        raise ValueError("measured multitoken server generation spans are required")
    first = np.asarray([r.server_scheduled_to_first_token for r in rows])
    tpot = np.asarray([r.server_tpot for r in rows if r.output_tokens > 1])
    residual = np.asarray([r.ttft - r.server_ttft for r in rows])
    if min(first.mean(), tpot.mean()) <= 0 or np.any(residual < 0):
        raise ValueError("positive server metric means and nonnegative client residuals are required")
    costs, condition = _solve(
        np.vstack((prefill / first.mean(), decode[multi] / tpot.mean())),
        np.r_[first / first.mean(), tpot / tpot.mean()],
    )
    if costs[0] <= 0:
        raise ValueError("fitted alpha must be positive for the simulation clock")
    offset, client_condition = _solve(client, residual)
    return AdvanceReservationFit(
        *map(float, costs),
        *map(float, offset),
        server_scaled_condition=condition,
        client_scaled_condition=client_condition,
        training_requests=len(rows),
        input_range=(min(inputs), max(inputs)),
        output_range=(min(outputs), max(outputs)),
        measurement_profile=profile,
        latency_basis=basis,
    )
