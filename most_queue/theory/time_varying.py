"""
Non-stationary Mt/M/c queues: time-dependent arrival rate lambda(t).

Two classic approximations for the time-varying blocking / delay probability:

* **PSA (pointwise stationary approximation)** — plug the instantaneous offered
  load a(t) = lambda(t)/mu into the stationary Erlang formula at every t. Exact
  in the limit of slow variation (and large c).
* **MOL (modified offered load)** — first smooth lambda(t) through an M/M/infinity
  response to obtain the offered load m(t) (dm/dt = lambda(t) - mu*m(t)), then plug
  m(t) into the stationary Erlang formula. Captures the lag/damping that PSA
  misses, and is markedly more accurate under fast variation.

Supported systems: "loss" (Mt/M/c/c, blocking = Erlang B) and "delay"
(Mt/M/c, probability of waiting = Erlang C).
"""

import time
from dataclasses import dataclass

import numpy as np

from most_queue.theory.base_queue import BaseQueue


@dataclass
class TimeVaryingResults:
    """Time-varying results over the analysis grid."""

    t: np.ndarray
    psa: np.ndarray  # PSA blocking/delay probability
    mol: np.ndarray  # MOL blocking/delay probability
    offered_load: np.ndarray  # MOL offered load m(t)
    duration: float = 0.0


def erlang_b(a: float, c: int) -> float:
    """Erlang B (loss) probability for offered load a and c servers."""
    b = 1.0
    for k in range(1, c + 1):
        b = a * b / (k + a * b)
    return b


def erlang_c(a: float, c: int) -> float:
    """Erlang C (probability of waiting) for offered load a and c servers (a < c)."""
    if a >= c:
        return 1.0
    b = erlang_b(a, c)
    return c * b / (c - a * (1 - b))


class TimeVaryingMMcCalc(BaseQueue):
    """
    PSA and MOL approximations for a non-stationary Mt/M/c queue.

    :param n: number of servers c.
    :param kind: "loss" (Erlang B blocking) or "delay" (Erlang C wait probability).
    """

    def __init__(self, n: int, kind: str = "loss"):
        super().__init__(n=n)
        self.c = n
        self.kind = kind.lower()
        if self.kind not in ("loss", "delay"):
            raise ValueError("kind must be 'loss' or 'delay'")
        self.lam_fn = None
        self.mu = None

    def set_sources(self, lam_fn):  # pylint: disable=arguments-differ
        """:param lam_fn: callable t -> lambda(t), the time-varying arrival rate."""
        self.lam_fn = lam_fn
        self.is_sources_set = True

    def set_servers(self, mu: float):  # pylint: disable=arguments-differ
        """:param mu: per-server service rate."""
        self.mu = mu
        self.is_servers_set = True

    def _prob(self, a: float) -> float:
        return erlang_b(a, self.c) if self.kind == "loss" else erlang_c(a, self.c)

    def run(self, t_grid, mol_warmup: float = 0.0, dt: float = 0.0) -> TimeVaryingResults:
        """
        Evaluate PSA and MOL over `t_grid`.

        :param t_grid: array of time points to report.
        :param mol_warmup: lead time before t_grid[0] over which to integrate the
            MOL offered-load ODE so it reaches periodic steady state (use a few / mu).
        :param dt: integration step for the MOL ODE (default: min grid step / 10).
        """
        self._check_if_servers_and_sources_set()
        start = time.process_time()
        t_grid = np.asarray(t_grid, dtype=float)
        mu = self.mu

        psa = np.array([self._prob(self.lam_fn(t) / mu) for t in t_grid])

        # MOL: integrate dm/dt = lambda(t) - mu*m from (t0 - warmup) to t_end
        if dt <= 0:
            dt = (t_grid[1] - t_grid[0]) / 10.0 if len(t_grid) > 1 else 1.0 / (10 * mu)
        t0 = t_grid[0] - mol_warmup
        m = self.lam_fn(t0) / mu  # start at the pointwise offered load
        # dense integration, sampling onto t_grid
        offered = np.empty_like(t_grid)
        gi = 0
        t = t0
        n_steps = int(np.ceil((t_grid[-1] - t0) / dt)) + 1
        for _ in range(n_steps):
            while gi < len(t_grid) and t_grid[gi] <= t + 1e-12:
                offered[gi] = m
                gi += 1
            m += dt * (self.lam_fn(t) - mu * m)
            t += dt
        while gi < len(t_grid):  # any trailing points
            offered[gi] = m
            gi += 1

        mol = np.array([self._prob(a) for a in offered])

        res = TimeVaryingResults(t=t_grid, psa=psa, mol=mol, offered_load=offered)
        res.duration = time.process_time() - start
        return res
