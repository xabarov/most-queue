"""Fit an isolated-request prefill/decode surrogate, with explicit client offset.

The stage-cost equations follow PrefillDecodeSim for one active request and
an unchunked prefill. They do not identify physical GPU kernel costs or server
queueing from client timings. See docs/serving_measurements.md for the domain,
rank requirement, held-out evaluation, and the distinction from load calibration.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls


def _design(inputs, outputs):
    inputs, outputs = np.asarray(inputs, dtype=float), np.asarray(outputs, dtype=float)
    if inputs.ndim != 1 or outputs.shape != inputs.shape or inputs.size == 0:
        raise ValueError("length vectors must align and be nonempty")
    if any(
        not np.all(np.isfinite(values)) or np.any(values < 1) or np.any(values != np.floor(values))
        for values in (inputs, outputs)
    ):
        raise ValueError("token lengths must be positive finite integers")
    ones = np.ones(len(inputs))
    ttft = np.column_stack((ones, 2 * ones, inputs + 1, 2 * inputs + 1))
    tpot = np.column_stack((0 * ones, ones, ones, inputs + (outputs + 2) / 2))
    return ttft, tpot


@dataclass(frozen=True)
class IsolatedServingFit:
    """Nonnegative effective coefficients in seconds, trained on solo requests."""

    client_offset: float
    alpha: float
    beta: float
    gamma: float
    scaled_condition: float
    training_requests: int
    input_range: tuple[int, int]
    output_range: tuple[int, int]
    measurement_profile: str
    latency_basis: str

    def predict(self, inputs, outputs):
        """Predict isolated TTFT/TPOT; one-token TPOT is returned as NaN.

        Numeric predictions do not imply validity outside the training domain
        or at concurrency >1. The fit supplies no waiting-time prediction.
        """
        ttft, tpot = _design(inputs, outputs)
        coefficients = [self.client_offset, self.alpha, self.beta, self.gamma]
        prediction = tpot @ coefficients
        prediction[np.asarray(outputs) <= 1] = np.nan
        return ttft @ coefficients, prediction


def _validate_profile(benchmarks):
    if not benchmarks or any(bench.metadata.get("max_concurrency") != 1 for bench in benchmarks):
        raise ValueError("calibration requires explicitly isolated max_concurrency=1 runs")
    profile = benchmarks[0].metadata.get("measurement_profile")
    basis = benchmarks[0].metadata.get("latency_basis")
    if not isinstance(profile, str) or not profile or not isinstance(basis, str) or not basis:
        raise ValueError("a common explicit measurement_profile and latency_basis are required")
    if any(
        bench.metadata.get("measurement_profile") != profile or bench.metadata.get("latency_basis") != basis
        for bench in benchmarks
    ):
        raise ValueError("cannot mix measurement profiles or latency definitions")
    return profile, basis


def fit_isolated_serving(benchmarks):
    """Fit four effective coefficients by NNLS using training runs only.

    Requires explicit max_concurrency=1 and complete successful cohorts. This
    establishes client isolation, not proof of an otherwise idle server.
    Each metric is divided by its TRAINING mean; columns are then normalized
    before NNLS. Rank deficiency is refused rather than reporting fake costs.
    """
    benchmarks = list(benchmarks)
    profile, basis = _validate_profile(benchmarks)
    rows = [row for bench in benchmarks for row in bench.requests]
    if any(row.status != "completed" for row in rows):
        raise ValueError("calibration requires complete cohorts without failed or censored requests")
    inputs = [row.input_tokens for row in rows]
    outputs = [row.output_tokens for row in rows]
    ttft, tpot = _design(inputs, outputs)
    multitoken = np.asarray(outputs) > 1
    if not np.any(multitoken):
        raise ValueError("multitoken observations are required to identify decode costs")
    observed_ttft = np.asarray([row.ttft for row in rows])
    observed_tpot = np.asarray([row.tpot for row in rows if row.tpot is not None])
    scales = [observed_ttft.mean(), observed_tpot.mean()]
    if min(scales) <= 0:
        raise ValueError("training metric means must be positive")
    design = np.vstack((ttft / scales[0], tpot[multitoken] / scales[1]))
    target = np.r_[observed_ttft / scales[0], observed_tpot / scales[1]]
    column_norm = np.linalg.norm(design, axis=0)
    scaled = design / column_norm
    if np.linalg.matrix_rank(scaled) < 4:
        raise ValueError("rank-deficient calibration: vary prompt and output lengths")
    condition = float(np.linalg.cond(scaled))
    if condition > 1e8:
        raise ValueError("calibration design is ill-conditioned")
    coefficients = nnls(scaled, target)[0] / column_norm
    return IsolatedServingFit(
        *map(float, coefficients),
        scaled_condition=condition,
        training_requests=len(rows),
        input_range=(min(inputs), max(inputs)),
        output_range=(min(outputs), max(outputs)),
        measurement_profile=profile,
        latency_basis=basis,
    )


def evaluate_isolated_serving(fit, benchmarks):
    """Report held-out errors without refitting or dropping failed outcomes."""
    benchmarks = list(benchmarks)
    profile, basis = _validate_profile(benchmarks)
    if profile != fit.measurement_profile or basis != fit.latency_basis:
        raise ValueError("held-out profile and latency definition must match training")
    rows = [row for bench in benchmarks for row in bench.requests]
    if any(row.status != "completed" for row in rows):
        raise ValueError("evaluation cohorts must be complete; audit failures separately")
    inputs, outputs = [row.input_tokens for row in rows], [row.output_tokens for row in rows]
    ttft, tpot = fit.predict(inputs, outputs)
    observed = [np.asarray([row.ttft for row in rows]), np.asarray([row.tpot for row in rows if row.tpot is not None])]
    predictions = [ttft, tpot[np.asarray(outputs) > 1]]
    report = {"requests": len(rows), "scope": "isolated client timings; no queue or loaded-serving validation"}
    for name, actual, predicted in zip(["ttft", "tpot"], observed, predictions):
        if actual.size == 0 or np.any(actual <= 0):
            raise ValueError("positive held-out metric values are required for relative errors")
        errors = np.abs(actual - predicted)
        report[name] = {
            "mae_seconds": float(errors.mean()),
            "median_absolute_relative_error": float(np.median(errors / actual)),
            "p90_absolute_relative_error": float(np.percentile(errors / actual, 90)),
            "observed": actual.tolist(),
            "predicted": predicted.tolist(),
        }
    report["outside_training_length_range"] = sum(
        not (fit.input_range[0] <= n <= fit.input_range[1] and fit.output_range[0] <= m <= fit.output_range[1])
        for n, m in zip(inputs, outputs)
    )
    return report
