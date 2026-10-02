"""Small train-only service-law baselines for trace-driven queue validation.

PH is balanced-means H2 (CV²>1), or a mean-matched integer Erlang (CV²<=1).
This is moment matching, not maximum likelihood or a certified tail fit.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtri
from scipy.stats import gamma

FAMILIES = ("empirical", "exponential", "ph", "lognormal")


@dataclass(frozen=True)
class ServiceCalibration:
    """Immutable empirical sample and parameters shared by four baselines."""

    samples: tuple[float, ...]
    mean: float
    cv2: float
    erlang_phases: int

    @classmethod
    def fit(cls, samples):
        """Fit positive finite observations; preserve the population mean."""
        values = np.asarray(samples, dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("at least two positive finite service observations are required")
        mean = float(np.mean(values))
        cv2 = float(np.mean((values / mean - 1) ** 2))
        if not np.isfinite(mean) or not np.isfinite(cv2):
            raise ValueError("service moments overflow; rescale data")
        phases = min(64, max(1, int(round(1 / max(cv2, 1 / 64)))))
        return cls(tuple(sorted(float(x) for x in values)), mean, cv2, phases)

    def parameters(self) -> dict:
        """Return auditable summaries without redistributing raw observations."""
        sigma2 = float(np.log1p(self.cv2))
        return {
            "count": len(self.samples),
            "mean": self.mean,
            "cv2": self.cv2,
            "p50": float(np.quantile(self.samples, 0.5)),
            "p90": float(np.quantile(self.samples, 0.9)),
            "p99": float(np.quantile(self.samples, 0.99)),
            "ph_kind": "h2_balanced" if self.cv2 > 1 else "erlang",
            "ph_phases": 2 if self.cv2 > 1 else self.erlang_phases,
            "ph_cv2": self.cv2 if self.cv2 > 1 else 1 / self.erlang_phases,
            "lognormal_mu": float(np.log(self.mean) - sigma2 / 2),
            "lognormal_sigma": float(np.sqrt(sigma2)),
        }

    def quantiles(self, family: str, probabilities):
        """Transform common uniforms in (0,1); no clipping or mean rescaling.

        Empirical sampling uses the inverse discrete CDF (not interpolation).
        H2 inverse CDF uses bounded vector bisection. Its two branch means each
        contribute half of E[S], matching the first two moments exactly.
        """
        if family not in FAMILIES:
            raise ValueError("unknown service family")
        u = np.asarray(probabilities, dtype=float)
        if not np.all(np.isfinite(u)) or np.any((u <= 0) | (u >= 1)):
            raise ValueError("probabilities must be finite and strictly between 0 and 1")
        if family == "empirical":
            out = np.asarray(self.samples)[np.ceil(u * len(self.samples)).astype(int) - 1]
        elif family == "exponential":
            out = -self.mean * np.log1p(-u)
        elif family == "lognormal":
            sigma2 = np.log1p(self.cv2)
            out = np.exp(np.log(self.mean) - sigma2 / 2 + np.sqrt(sigma2) * ndtri(u))
        elif self.cv2 <= 1:
            out = gamma.ppf(u, a=self.erlang_phases, scale=self.mean / self.erlang_phases)
        else:
            # Rationalized small probability avoids cancellation when CV is high.
            root = np.sqrt((self.cv2 - 1) / (self.cv2 + 1))
            small = 1 / ((self.cv2 + 1) * (1 + root))
            rates = np.array([2 * (1 - small), 2 * small]) / self.mean
            lower = np.zeros_like(u)
            upper = -np.log1p(-u) / rates.min()
            for _ in range(80):
                mid = (lower + upper) / 2
                cdf = (1 - small) * (-np.expm1(-rates[0] * mid)) + small * (-np.expm1(-rates[1] * mid))
                lower = np.where(cdf < u, mid, lower)
                upper = np.where(cdf >= u, mid, upper)
            out = (lower + upper) / 2
        if not np.all(np.isfinite(out)) or np.any(out <= 0):
            raise ValueError("generated service is not representable; rescale data")
        return out
