"""
Simulation-time predictors for ``SizeBasedQsSim`` (SPJF / PSPJF / SPRPT).

These mirror the stochastic models used in ``most_queue.theory.srpt.utils.predictor``
but sample ``Y`` from the conditional law given the realised service requirement ``X``.
"""

from __future__ import annotations

import math
from typing import Any


class ExpNoiseSimPredictor:
    """Sample ``Y | X = x ~ Exp(scale=x)`` (mean ``x``, rate ``1/x``).

    Matches ``ExpNoisePredictor`` in theory (Mitzenmacher 2020 / Mitzenmacher-Shahout 2025).
    """

    def predict(self, size: float, rng: Any) -> float:
        """Return a sample of the predicted size given true size *size*."""
        x = max(float(size), 1e-12)
        return float(rng.exponential(x))


class LognormalNoiseSimPredictor:
    """Sample ``Y | X = x ~ LogNormal(mean_log=log(x), sigma)``.

    Matches ``LognormalNoisePredictor`` on the theory side.
    """

    def __init__(self, sigma: float = 0.5) -> None:
        """Initialise with log-scale standard deviation *sigma*."""
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.sigma = float(sigma)

    def predict(self, size: float, rng: Any) -> float:
        """Return a sample of the predicted size given true size *size*."""
        x = max(float(size), 1e-12)
        return float(rng.lognormal(math.log(x), self.sigma))


__all__ = ["ExpNoiseSimPredictor", "LognormalNoiseSimPredictor"]
