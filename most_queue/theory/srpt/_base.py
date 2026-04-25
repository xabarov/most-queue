"""
Shared base class for size-based M/G/1 analytical calculators.

Grid precomputation is the key performance optimisation: all inner
integrals (rho_x, int t^2 f dt, int dt/(1-rho_t), CDF) are evaluated once on a
fine mesh over [0, x_max] using cumulative trapezoidal sums, and
then looked up via np.interp.  This eliminates the O(n^3) nested-quad
pattern that arises when rho_x is re-integrated for every outer-quad
evaluation point.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid

from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.srpt.utils.load_below import build_pdf_cdf, get_theory_moments, upper_bound

_N_GRID: int = 2000


class _SizeBasedCalcBase(BaseQueue):
    """
    Abstract base for SRPT / SJF / PSJF calculators.

    Provides:
    - ``set_sources(l)`` / ``set_servers(params, kendall_notation)`` boilerplate.
    - ``_build_grids()`` -- precomputes rho_x, int t^2 f, int dt/(1-rho_t), CDF on a fine grid.
    - ``_rho_interp``, ``_t2f_interp``, ``_cdf_interp``, ``_second_int_interp``
      -- cheap O(log n) interpolation replacements for inner quad calls.
    - ``_check_stability()`` -- validates rho < 1.
    """

    def __init__(self) -> None:
        super().__init__(n=1)
        self.l: float | None = None
        self.b: list[float] | None = None
        self.pdf_fn = None
        self.cdf_fn = None
        self.x_max: float | None = None

        self._grid_xs: np.ndarray | None = None
        self._grid_rho: np.ndarray | None = None
        self._grid_t2f: np.ndarray | None = None
        self._grid_cdf: np.ndarray | None = None
        self._grid_second_int: np.ndarray | None = None

    def set_sources(self, l: float) -> None:  # pylint: disable=arguments-differ
        self.l = float(l)
        self.is_sources_set = True

    def set_servers(self, params, kendall_notation: str = "H") -> None:  # pylint: disable=arguments-differ
        pdf_fn, cdf_fn = build_pdf_cdf(params, kendall_notation)
        self.pdf_fn = pdf_fn
        self.cdf_fn = cdf_fn
        self.b = get_theory_moments(params, kendall_notation, 3)
        self.x_max = upper_bound(self.cdf_fn, p=1e-7)
        self.is_servers_set = True
        self._grid_xs = None  # invalidate stale grids on re-configuration

    def _build_grids(self) -> None:
        """Precompute integration grids over [0, x_max] (called at run time).

        A hybrid grid is used: a fine log-spaced section near 0 is prepended
        to a coarser linear section. This gives good resolution near 0 where
        Gamma / Pareto PDFs can be singular, without exploding the grid size.
        """
        if self._grid_xs is not None:
            return  # already built for current configuration

        x_max = self.x_max
        eps = x_max * 1e-8
        # ~40% of points in the log-section [eps, 0.1*x_max], 60% in linear
        n_log = int(_N_GRID * 0.4)
        n_lin = _N_GRID - n_log

        xs_log = np.geomspace(eps, x_max * 0.1, n_log, endpoint=False)
        xs_lin = np.linspace(x_max * 0.1, x_max, n_lin + 1)
        xs = np.concatenate([[0.0], xs_log, xs_lin])

        pdf_vals = np.vectorize(self.pdf_fn, otypes=[np.float64])(xs)
        cdf_vals = np.vectorize(self.cdf_fn, otypes=[np.float64])(xs)

        # rho_x = l * integral_0^x t f(t) dt
        rho_vals = self.l * np.concatenate([[0.0], cumulative_trapezoid(xs * pdf_vals, xs)])
        # clip to [0, 1) to avoid accidental overshoot from numerical errors
        rho_vals = np.clip(rho_vals, 0.0, 1.0 - 1e-12)

        # integral_0^x t^2 f(t) dt
        t2f_vals = np.concatenate([[0.0], cumulative_trapezoid(xs * xs * pdf_vals, xs)])

        # integral_0^x dt / (1 - rho_t)  -- used by SRPT second term
        second_int_vals = np.concatenate([[0.0], cumulative_trapezoid(1.0 / (1.0 - rho_vals), xs)])

        self._grid_xs = xs
        self._grid_rho = rho_vals
        self._grid_t2f = t2f_vals
        self._grid_cdf = cdf_vals
        self._grid_second_int = second_int_vals

    # ------------------------------------------------------------------
    # Fast interpolated accessors (replace inner quad calls)
    # ------------------------------------------------------------------

    def _rho_interp(self, x: float) -> float:
        return float(np.interp(x, self._grid_xs, self._grid_rho))

    def _t2f_interp(self, x: float) -> float:
        return float(np.interp(x, self._grid_xs, self._grid_t2f))

    def _cdf_interp(self, x: float) -> float:
        return float(np.interp(x, self._grid_xs, self._grid_cdf))

    def _second_int_interp(self, x: float) -> float:
        return float(np.interp(x, self._grid_xs, self._grid_second_int))

    def _check_stability(self) -> float:
        utilization = self.l * self.b[0]
        if utilization >= 1.0:
            raise ValueError("System is unstable: utilization must be < 1")
        return utilization
