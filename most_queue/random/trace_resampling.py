"""Conditional empirical service and circular resampling of historical ranks.

Equal-length circular blocks follow the end-to-start convention described at
https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html.
This module generates workloads; it does not provide population bootstrap CIs.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from most_queue.random.service_calibration import ServiceCalibration


def _count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _needs(values):
    values = np.asarray(values)
    if values.ndim != 1 or not values.size or values.dtype.kind not in "iu" or np.any(values < 1):
        raise ValueError("needs must be a nonempty vector of positive integers")
    return values


def _uniforms(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)) or np.any((values <= 0) | (values >= 1)):
        raise ValueError("uniforms must be a nonempty vector strictly inside (0,1)")
    return values


@dataclass(frozen=True)
class ConditionalEmpirical:
    """History-only fits; exact K -> coarse group -> pooled fallback.

    Resource groups are 1, 2..8, 9..32, >=33. Fallback is determined only by
    historical counts; neither test durations nor test frequency are used.
    """

    pooled: ServiceCalibration
    coarse: tuple[ServiceCalibration | None, ...]
    exact: dict[int, ServiceCalibration]
    minimum: int

    @classmethod
    def fit(cls, needs, services, *, exact=False, minimum=20):
        """Fit positive service observations in the supplied history order."""
        minimum = _count(minimum, "minimum")
        if minimum < 2:
            raise ValueError("minimum must be >= 2")
        if not isinstance(exact, bool):
            raise ValueError("exact must be boolean")
        needs = _needs(needs)
        services = np.asarray(services, dtype=float)
        if services.shape != needs.shape:
            raise ValueError("needs and services must have the same shape")
        pooled = ServiceCalibration.fit(services)
        labels = np.searchsorted([1, 8, 32], needs)
        coarse = tuple(
            ServiceCalibration.fit(services[labels == group]) if np.sum(labels == group) >= minimum else None
            for group in range(4)
        )
        by_need = {}
        if exact:
            for need in np.unique(needs):
                sample = services[needs == need]
                if len(sample) >= minimum:
                    by_need[int(need)] = ServiceCalibration.fit(sample)
        return cls(pooled, coarse, by_need, minimum)

    def distribution(self, need):
        """Return the selected immutable empirical distribution and fit level."""
        need = _count(need, "need")
        if need in self.exact:
            return self.exact[need], "exact"
        coarse = self.coarse[int(np.searchsorted([1, 8, 32], need))]
        if coarse is not None:
            return coarse, "coarse"
        return self.pooled, "pooled"

    def quantiles(self, needs, uniforms):
        """Map a probability tape to empirical durations for exact target K."""
        needs, uniforms = _needs(needs), _uniforms(uniforms)
        if needs.shape != uniforms.shape:
            raise ValueError("needs and uniforms must have the same shape")
        result = np.empty(len(needs))
        for need in np.unique(needs):
            mask = needs == need
            result[mask] = self.distribution(need)[0].quantiles("empirical", uniforms[mask])
        return result

    def ranks(self, needs, services):
        """Compute (F(s-) + F(s))/2, preserving ties without arbitrary jitter.

        Training observations give probabilities strictly inside (0,1).
        Held-out observations may give 0 or 1; these are diagnostics only and
        must not be fed to quantiles as if they were available training ranks.
        """
        needs = _needs(needs)
        services = np.asarray(services, dtype=float)
        if services.shape != needs.shape or not np.all(np.isfinite(services)) or np.any(services <= 0):
            raise ValueError("services must match needs and be positive and finite")
        result = np.empty(len(needs))
        for need in np.unique(needs):
            mask = needs == need
            sample = np.asarray(self.distribution(need)[0].samples)
            left = np.searchsorted(sample, services[mask], side="left")
            right = np.searchsorted(sample, services[mask], side="right")
            result[mask] = (left + right) / (2 * len(sample))
        return result


def circular_block_indices(history_size, uniforms, block_length=1):
    """Draw uniform block starts; wrap within history, truncate the last block.

    Starts use uniforms at output positions 0,L,2L,..., allowing common random
    inputs for different L. Each output position has a uniform donor index;
    L=1 is iid sampling of the SAME tape. No stationarity claim is implied.
    """
    history_size = _count(history_size, "history_size")
    block_length = _count(block_length, "block_length")
    if block_length > history_size:
        raise ValueError("block_length must not exceed history size")
    uniforms = _uniforms(uniforms)
    starts = (uniforms[::block_length] * history_size).astype(np.int64)
    indices = (starts[:, None] + np.arange(block_length)) % history_size
    return indices.ravel()[: len(uniforms)]


def lag_correlation(values, lag=1):
    """Pearson correlation at a positive lag; None for unavailable variance."""
    lag = _count(lag, "lag")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite vector")
    if len(values) <= lag + 1:
        return None
    left, right = values[:-lag], values[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])
