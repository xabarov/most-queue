"""History-only conditional ECDFs with audited fallback and optional scale ratios.

CRPS uses the empirical energy identity, Gneiting & Raftery (2007),
https://doi.org/10.1198/016214506000001437. It scores a full predictive CDF,
not a random sample from it. Feature availability must be established by callers.
"""

from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from most_queue.random.service_calibration import ServiceCalibration
from most_queue.random.trace_resampling import ConditionalEmpirical


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _need(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError("need must be a positive integer")
    return int(value)


def _context(value):
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError("context must be a nonempty string or None")
    return value


def request_bucket(request):
    """Fixed right-closed request bins in seconds; unknown is distinct from zero."""
    if request is None:
        return None
    value = _positive(request, "request")
    return str(int(np.searchsorted([300, 1800, 7200, 28800, 86400], value)))


@dataclass(frozen=True)
class EmpiricalPrediction:
    """An immutable selected ECDF, positive target scale and audited fit level."""

    distribution: ServiceCalibration
    scale: float = 1.0
    level: str = "pooled"

    def __post_init__(self):
        _positive(self.scale, "scale")
        # Check the support before sampling, not only the selected quantile.
        _positive(self.scale * self.distribution.samples[-1], "scaled support")
        _positive(self.scale * self.distribution.samples[0], "scaled support")

    @property
    def mean(self):
        """Predictive mean in service-time units, not ratio units."""
        return self.scale * self.distribution.mean

    def quantile(self, probability):
        """Inverse discrete ECDF; no interpolation or truncation at runtime budget."""
        if isinstance(probability, (bool, np.bool_)) or not isinstance(probability, Real):
            raise ValueError("probability must be a finite real in (0,1)")
        return float(self.scale * self.distribution.quantiles("empirical", [probability])[0])

    def probability_exceeding(self, threshold):
        """Return P(S > threshold), with equality counted as within the budget."""
        threshold = _positive(threshold, "threshold")
        support = np.array(self.distribution.samples) * self.scale
        return float(np.mean(support > threshold))

    def crps(self, observed):
        """Compute E|X-y| - .5 E|X-X'| exactly for a finite equal-weight ECDF.

        For sorted support x_i the second term is sum((2i-n-1)x_i)/n².
        Scaling before evaluation keeps score units in seconds for ratio fits.
        """
        observed = _positive(observed, "observed")
        support = np.asarray(self.distribution.samples) * self.scale
        n = len(support)
        dispersion = np.dot((2 * np.arange(1, n + 1) - n - 1) / n, support) / n
        score = float(np.mean(np.abs(support - observed)) - dispersion)
        if not np.isfinite(score):
            raise ValueError("CRPS overflow; rescale observations")
        return max(0.0, score)


@dataclass(frozen=True)
class FeatureConditionalEmpirical:
    """Joint context/need cells, then coarse/pooled fallback; fit uses history only.

    Optional scales fit service/scale on known positive scales, multiplying by
    the supplied target scale at prediction. Unknown target scale or absent
    ratio history falls back to absolute service. No outcome or test S is used.
    """

    absolute: ConditionalEmpirical
    pooled: ServiceCalibration | None
    coarse: dict
    joint: dict
    exact_joint: dict
    ratio: bool
    audit: dict

    @classmethod
    def fit(
        cls, needs, services, contexts, *, scales=None, exact=False, minimum=20
    ):  # pylint: disable=too-many-arguments
        """Fit context ECDFs with a fixed minimum count and explicit missing scales."""
        needs, contexts = tuple(needs), tuple(contexts)
        if len(contexts) != len(needs):
            raise ValueError("contexts must match needs")
        needs = tuple(_need(value) for value in needs)
        contexts = tuple(_context(value) for value in contexts)
        absolute = ConditionalEmpirical.fit(needs, services, minimum=minimum)
        if not isinstance(exact, bool):
            raise ValueError("exact must be boolean")
        values = np.asarray(services, dtype=float)
        ratio = scales is not None
        scales = tuple(scales) if ratio else (1.0,) * len(needs)
        if len(scales) != len(needs):
            raise ValueError("scales must match needs")
        pools = {"coarse": defaultdict(list), "joint": defaultdict(list), "exact": defaultdict(list)}
        normalized = []
        for need, service, context, scale in zip(needs, values, contexts, scales):
            if scale is None:
                continue
            value = _positive(service / _positive(scale, "scale"), "scaled service")
            group = int(np.searchsorted([1, 8, 32], need))
            normalized.append(value)
            pools["coarse"][group].append(value)
            pools["joint"][group, context].append(value)
            if exact:
                pools["exact"][need, context].append(value)
        fitted = {
            name: {key: ServiceCalibration.fit(sample) for key, sample in cells.items() if len(sample) >= minimum}
            for name, cells in pools.items()
        }
        pooled = ServiceCalibration.fit(normalized) if len(normalized) >= 2 else None
        audit = {
            "history_count": len(needs),
            "scaled_count": len(normalized),
            "missing_scale_count": sum(s is None for s in scales),
            "minimum": int(minimum),
            "exact": exact,
            "ratio": ratio,
            "fitted_cells": {name: len(cells) for name, cells in fitted.items()},
        }
        return cls(absolute, pooled, fitted["coarse"], fitted["joint"], fitted["exact"], ratio, audit)

    def predict(self, need, context, *, scale=None):
        """Return the complete predictive distribution and the fallback used."""
        need, context = _need(need), _context(context)
        if scale is not None:
            _positive(scale, "scale")
        if self.ratio and (scale is None or self.pooled is None):
            distribution, level = self.absolute.distribution(need)
            return EmpiricalPrediction(distribution, 1.0, f"absolute_{level}")
        if not self.ratio and scale is not None:
            raise ValueError("scale is only valid for a ratio fit")
        factor = float(scale) if self.ratio else 1.0
        group = int(np.searchsorted([1, 8, 32], need))
        choices = (
            (self.exact_joint.get((need, context)), "exact_context"),
            (self.joint.get((group, context)), "coarse_context"),
            (self.coarse.get(group), "coarse"),
            (self.pooled, "pooled"),
        )
        for distribution, level in choices:
            if distribution is not None:
                return EmpiricalPrediction(distribution, factor, level)
        raise ValueError("no predictive distribution available")
