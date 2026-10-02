"""Feature-only runtime regression and one-sided split-conformal calibration.

Calibration follows the rank construction of Lei et al. (2018),
doi:10.1080/01621459.2017.1307116, applied to signed log-runtime residuals.
Marginal coverage requires exchangeable calibration/test observations and a
model fitted independently of them. It is NOT a scheduling or per-class SLO.
Group calibration follows Angelopoulos and Bates, arXiv:2107.07511, Sec. 4.1,
with a fixed, observable resource group and exchangeability within each group.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp


@dataclass(frozen=True)
class RuntimeCalibration:
    """Finite calibration order statistic, not an empirical test guarantee."""

    coverage: float
    samples: int
    rank: int
    log_margin: float


@dataclass(frozen=True)
class RuntimeGroupCalibration:
    """Group rank; None margin means no finite bound at the requested coverage."""

    coverage: float
    samples: int
    rank: int
    log_margin: float | None

    @property
    def is_finite(self) -> bool:
        """Whether this group has enough observations for a finite score bound."""
        return self.log_margin is not None


def _coverage(value):
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(value)
        or not 0 < value < 1
    ):
        raise ValueError("coverage must be finite and strictly between zero and one")


def _groups(values, count):
    # Preserve Python types: coercing [0, True] to int would hide an invalid label.
    labels = np.asarray(values, dtype=object)
    if labels.shape != (count,) or any(
        not isinstance(label, (int, np.integer)) or isinstance(label, (bool, np.bool_)) or label < 0 for label in labels
    ):
        raise ValueError("groups must contain one nonnegative integer label per feature row, not booleans")
    return labels


def _features(values):
    if np.iscomplexobj(values):
        raise ValueError("features must be real")
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("features must be a finite nonempty 2D matrix; zero columns are allowed")
    return matrix


def _durations(values, count):
    if np.iscomplexobj(values):
        raise ValueError("durations must be real")
    durations = np.asarray(values, dtype=float)
    if durations.shape != (count,) or not np.all(np.isfinite(durations)) or np.any(durations <= 0):
        raise ValueError("one finite positive duration per feature row is required")
    return durations


class LogLinearRuntimePredictor:
    """OLS on log service, with training-only smearing and held-out calibration.

    ``fit(X, S)`` learns standardized numeric features plus an intercept and a
    multiplicative mean-residual correction. ``predict(X)`` is a positive point
    estimate, not a guaranteed conditional mean under model misspecification.
    ``calibrate(X_cal, S_cal, coverage)`` must use independent held-out data.
    ``predict(X, upper=True)`` then returns one-sided conformal upper estimates.
    ``calibrate_by_group(X_cal, S_cal, groups)`` adds separate group thresholds;
    ``predict(X, upper=True, groups=groups)`` uses them without pooled fallback.

    Callers must supply only features observable at submission and keep data
    splits disjoint. This API cannot establish feature provenance or detect
    copied/overlapping samples. No actual runtime is accepted by ``predict``.
    """

    def __init__(self):
        self._center = self._scale = self._coefficients = None
        self._log_smearing = None
        self._calibration = None
        self._group_calibrations = {}

    @property
    def calibration(self) -> RuntimeCalibration | None:
        """Return immutable calibration diagnostics, or None before calibration."""
        return self._calibration

    @property
    def group_calibrations(self) -> Mapping[int, RuntimeGroupCalibration]:
        """Return a read-only snapshot of observed groups, empty before calibration."""
        return MappingProxyType(self._group_calibrations)

    def fit(self, features: ArrayLike, durations: ArrayLike) -> "LogLinearRuntimePredictor":
        """Fit on completed historical jobs; a successful refit clears calibration."""
        matrix = _features(features)
        durations = _durations(durations, len(matrix))
        if len(matrix) < 2:
            raise ValueError("at least two training jobs are required")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                center = matrix.mean(axis=0)
                scale = matrix.std(axis=0)
                scale = np.where(scale > 0, scale, 1.0)
                design = np.column_stack((np.ones(len(matrix)), (matrix - center) / scale))
                log_service = np.log(durations)
                coefficients = np.linalg.lstsq(design, log_service, rcond=None)[0]
                residuals = log_service - design @ coefficients
                log_smearing = float(logsumexp(residuals) - math.log(len(residuals)))
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise ValueError("runtime regression is numerically invalid; rescale features") from exc
        if not np.all(np.isfinite(coefficients)) or not np.isfinite(log_smearing):
            raise ValueError("runtime regression produced nonfinite parameters")
        self._center, self._scale, self._coefficients = center, scale, coefficients
        self._log_smearing = log_smearing
        self._calibration = None
        self._group_calibrations = {}
        return self

    def _log_predictions(self, features):
        if self._coefficients is None:
            raise ValueError("fit must be called before prediction or calibration")
        matrix = _features(features)
        if matrix.shape[1] != len(self._center):
            raise ValueError("feature count differs from the fitted model")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                design = np.column_stack((np.ones(len(matrix)), (matrix - self._center) / self._scale))
                values = design @ self._coefficients + self._log_smearing
        except FloatingPointError as exc:
            raise ValueError("prediction overflows; rescale features or restrict extrapolation") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError("nonfinite log predictions")
        return values

    def calibrate(self, features: ArrayLike, durations: ArrayLike, coverage: float = 0.95) -> RuntimeCalibration:
        """Calibrate signed log residuals using the exact ceil((n+1)*coverage) rank.

        If the required rank is n+1, the conformal bound is infinite. Raise
        instead of silently substituting a finite maximum for that bound:
        the MSJ replay API intentionally requires finite duration estimates.
        """
        _coverage(coverage)
        predicted = self._log_predictions(features)
        durations = _durations(durations, len(predicted))
        count = len(durations)
        rank = math.ceil((count + 1) * coverage)
        if rank > count:
            raise ValueError("not enough calibration jobs for a finite bound at this coverage")
        scores = np.log(durations) - predicted
        margin = float(np.partition(scores, rank - 1)[rank - 1])
        calibration = RuntimeCalibration(float(coverage), count, rank, margin)
        self._calibration = calibration
        return calibration

    def calibrate_by_group(
        self, features: ArrayLike, durations: ArrayLike, groups: ArrayLike, coverage: float = 0.95
    ) -> Mapping[int, RuntimeGroupCalibration]:
        """Calibrate a separate signed-score rank for every observed resource group.

        Groups must be fixed before looking at calibration/test labels and be
        observable at submission. Insufficient groups get a None margin, not a
        clipped rank. Predicting them, or unseen groups, raises ValueError.
        This replaces all group diagnostics but preserves pooled calibration.
        Invalid inputs leave both previous calibrations unchanged.
        """
        _coverage(coverage)
        predicted = self._log_predictions(features)
        durations = _durations(durations, len(predicted))
        labels = _groups(groups, len(predicted))
        scores = np.log(durations) - predicted
        calibrations = {}
        for group in sorted(set(labels)):
            group_scores = scores[labels == group]
            count = len(group_scores)
            rank = math.ceil((count + 1) * coverage)
            margin = float(np.partition(group_scores, rank - 1)[rank - 1]) if rank <= count else None
            calibrations[int(group)] = RuntimeGroupCalibration(float(coverage), count, rank, margin)
        self._group_calibrations = calibrations
        return self.group_calibrations

    def _upper_margins(self, groups, count):
        if groups is None:
            if self._calibration is None:
                raise ValueError("calibrate must be called before pooled upper prediction")
            return self._calibration.log_margin
        labels = _groups(groups, count)
        if not self._group_calibrations:
            raise ValueError("calibrate_by_group must be called before grouped upper prediction")
        unknown = sorted(set(labels) - self._group_calibrations.keys())
        if unknown:
            raise ValueError(f"unseen calibration groups: {unknown}; no pooled fallback")
        insufficient = sorted(group for group in set(labels) if not self._group_calibrations[group].is_finite)
        if insufficient:
            raise ValueError(f"not enough calibration jobs for a finite bound in groups: {insufficient}")
        return np.array([self._group_calibrations[group].log_margin for group in labels], dtype=float)

    def predict(self, features: ArrayLike, *, upper: bool = False, groups: ArrayLike | None = None) -> np.ndarray:
        """Predict durations from observable features only; never clips overflow."""
        if not isinstance(upper, (bool, np.bool_)):
            raise ValueError("upper must be a boolean")
        if groups is not None and not upper:
            raise ValueError("groups select a calibration and require upper=True")
        values = self._log_predictions(features)
        margins = self._upper_margins(groups, len(values)) if upper else None
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            if upper:
                # Round outwards: exp(log(s)) can otherwise be just BELOW s,
                # turning ties/constant runtimes into artificial undercoverage.
                margin = np.nextafter(margins, np.inf)
                values = np.nextafter(values + margin, np.inf)
            predicted = np.exp(values)
            if upper:
                predicted = np.where(predicted > 0, np.nextafter(predicted, np.inf), predicted)
        if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0):
            raise ValueError("predicted duration overflows or underflows; rescale data or restrict extrapolation")
        return predicted
