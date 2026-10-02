"""Kaplan-Meier residual-runtime estimates from independently censored history.

The input is (min(S, C), S <= C), never the unobserved lifetime S. Censoring
must be independent of S within the population being fitted. Quantiles are
plug-in estimates, not finite-sample coverage bounds. Unidentified tails are
reported as None, never extrapolated or replaced by a completed observation.
Kaplan and Meier (1958), doi:10.1080/01621459.1958.10501452.
"""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class KaplanMeierPoint:
    """Risk set and right-continuous survival at one observed time."""

    time: float
    at_risk: int
    events: int
    censored: int
    survival: float


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be representable as a finite float")
    return converted


class KaplanMeierRuntimeEstimator:
    """One population's survival curve; fit separate instances for classes.

    Completed events precede censoring at tied times. Survival means P(S > t).
    A flat positive terminal survival is not evidence of an infinite lifetime:
    it means that the remaining tail was not identified by this sample.
    """

    def __init__(self) -> None:
        self._times = self._survival = self._negative_survival = None
        self._curve = ()

    def fit(self, observed_times: ArrayLike, event_observed: ArrayLike) -> "KaplanMeierRuntimeEstimator":
        """Fit positive observed durations and a matching boolean completion mask.

        All observations must already be available before the forecast is used.
        Invalid refits leave the previous fitted state unchanged.
        """
        times, completed = np.asarray(observed_times), np.asarray(event_observed)
        if (
            times.ndim != 1
            or not times.size
            or times.dtype.kind not in "iuf"
            or not np.all(np.isfinite(times))
            or np.any(times <= 0)
        ):
            raise ValueError("observed_times must be a nonempty finite positive real vector")
        if completed.dtype.kind != "b" or completed.shape != times.shape:
            raise ValueError("event_observed must be a matching boolean vector")
        with np.errstate(over="ignore", under="ignore"):
            times = times.astype(float)
        if not np.all(np.isfinite(times)) or np.any(times <= 0):
            raise ValueError("observed_times must be representable as positive finite floats")
        unique, inverse, counts = np.unique(times, return_inverse=True, return_counts=True)
        risk = times.size - np.r_[0, np.cumsum(counts[:-1])]
        events = np.bincount(inverse, weights=completed, minlength=unique.size).astype(int)
        survival = np.cumprod(1.0 - events / risk)
        curve = tuple(
            KaplanMeierPoint(float(t), int(n), int(d), int(c - d), float(s))
            for t, n, d, c, s in zip(unique, risk, events, counts, survival)
        )
        self._times, self._survival = unique, survival
        self._negative_survival = -survival
        self._curve = curve
        return self

    @property
    def curve(self) -> tuple[KaplanMeierPoint, ...]:
        """Return immutable risk-set diagnostics, empty before fitting."""
        return self._curve

    def survival(self, age: float) -> float:
        """Evaluate the right-continuous KM curve (flat after the last observation).

        The flat extension is only the estimator convention, not tail evidence.
        Use the residual methods to detect unsupported tail predictions.
        """
        if self._times is None:
            raise ValueError("fit observed history before prediction")
        age = _real(age, "age")
        if age < 0:
            raise ValueError("age must be nonnegative")
        idx = int(np.searchsorted(self._times, age, side="right")) - 1
        return 1.0 if idx < 0 else float(self._survival[idx])

    def remaining_quantile(self, age: float, probability: float = 0.9) -> float | None:
        """Estimate the conditional residual quantile given S > age.

        Return None if survival at age is zero or the required tail crossing
        is unobserved. A finite answer is positive and never extrapolated.
        """
        probability = _real(probability, "probability")
        if not 0 < probability < 1:
            raise ValueError("probability must be strictly between zero and one")
        survived = self.survival(age)
        if survived == 0:
            return None
        target = (1 - probability) * survived
        side = "right" if target == survived else "left"
        idx = int(np.searchsorted(self._negative_survival, -target, side=side))
        return None if idx == self._times.size else float(self._times[idx] - age)

    def remaining_mean(self, age: float, horizon: float | None = None) -> float | None:
        """Estimate E[S-age | S>age], or its restriction to horizon-age.

        ``horizon`` is an absolute lifetime, strictly greater than age. With a
        positive terminal survival, the unrestricted mean and horizons beyond
        the last observed time are unavailable. Zero survival at age is also
        unavailable. No positive terminal plateau is extrapolated.
        """
        survived = self.survival(age)
        if horizon is not None:
            horizon = _real(horizon, "horizon")
            if horizon <= age:
                raise ValueError("horizon must be greater than age")
        if survived == 0:
            return None
        last = float(self._times[-1])
        if self._survival[-1] > 0 and (horizon is None or horizon > last):
            return None
        stop = last if horizon is None else min(horizon, last)
        interior = self._times[(self._times > age) & (self._times < stop)]
        knots = np.r_[age, interior, stop]
        levels = np.r_[survived, self._survival[np.searchsorted(self._times, interior)]] / survived
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(np.dot(np.diff(knots), levels))
        if not np.isfinite(result):
            raise ValueError("residual mean overflows; rescale durations")
        return result
