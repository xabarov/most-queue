"""
EPIC-073: exact waiting-time distribution for an occupancy-dependent queue
with a hard concurrency cap -- the Markovian core of LLM-inference continuous
batching.

Modern inference engines (vLLM/SGLang-style "continuous"/iteration-level
batching) do not serve a fixed group atomically: at every decode step a
request can join the currently-executing batch as soon as a slot frees, and
leave independently of the others. The number of requests that can be
resident at once (`k`) is capped by available accelerator memory (KV-cache
pages), and the per-request completion rate itself commonly depends on the
CURRENT occupancy, since all resident requests share one iteration's compute/
memory-bandwidth budget (see e.g. Ramani & Tantawi 2026, who build the same
"state-dependent Markov chain for batch occupancy" but solve it only
approximately, via mean-value analysis, for the MEAN TTFT/ITL).

This module gives the EXACT result for the same system class: any state-
dependent per-request rate ``mu(occupancy)`` for occupancy in ``1..k``, any
``k``, Poisson arrivals, FCFS beyond the cap. Two facts make this tractable
in closed form:

1. The chain (total number in system `j`) is an ordinary one-dimensional
   birth-death process -- even though the DEATH rate depends on `j` (through
   occupancy = min(j, k)), detailed balance always holds for a birth-death
   chain, so the stationary distribution ``pi`` has an explicit, non-
   iterative product form (``_solve_pi``).
2. Occupancy is PINNED at exactly `k` in every state with `j >= k` (one
   departure always triggers one immediate admission from the queue while
   anyone waits) -- so the departure rate seen by a PASTA-arriving tagged
   customer who finds `j >= k` is the FIXED rate ``k * mu(k)``, regardless of
   how `mu` behaves below the cap. Conditional on arriving to find `j >= k`,
   the wait before admission is therefore an exact ``Erlang(j - k + 1, k *
   mu(k))`` -- a classical Erlang-C-type tail, but exact (not MVA-
   approximated) and explicitly weighted by the state-dependent ``pi_j``.

Exact regression check: at ``mu(occupancy) == const``, this collapses to the
classical M/M/k/N queue (``most_queue.theory.fifo.mmnr.MMnrCalc``).

Scope (see docs/epics/EPIC-073*.md for the fuller discussion and the
dissertation roadmap, docs/диссертация/план-моделей.md, M1): a fixed,
occupancy-INDEPENDENT per-request memory footprint is assumed (`k` itself is
exogenous and constant) -- a growing, per-request footprint (real KV-cache
behavior) is an explicit, separate reserve (M3 in the roadmap, closest
existing structural analogue: ``most_queue.theory.msj``). `W` here is the
queueing delay before admission into the active pool only, not the full
sojourn (post-admission service duration is itself occupancy-dependent and
out of scope here -- same scoping convention as the rest of the library's
batch-service family, where `W` excludes the tagged customer's own service).
"""

import numpy as np

from most_queue.random.distributions import ErlangDistribution
from most_queue.random.utils.params import ErlangParams
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue

_NEGLIGIBLE_PROB = 1e-14  # below this, skip the state (also avoids math.factorial overflow at large r)


class OccupancyDependentQueueCalc(BaseQueue):
    """
    M/M(n)/k/N queue with occupancy-dependent per-request completion rate.

    :param k: concurrency cap (occupancy ceiling) -- e.g. the number of
        request slots that fit in available accelerator memory under a fixed
        per-request footprint assumption.
    :param queue_truncation: cap `N` on total number in system for the exact
        (closed-form, but normalized over a finite range) solve; must be >= k.
    """

    def __init__(self, k: int, queue_truncation: int = 300):
        super().__init__(n=k)
        if queue_truncation < k:
            raise ValueError(f"queue_truncation must be >= k, got {queue_truncation} < {k}")
        self.k = k
        self.N = queue_truncation
        self.l = None
        self.mu_fn = None
        self._pi: np.ndarray | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        self.l = l
        self.is_sources_set = True

    def set_servers(self, mu):  # pylint: disable=arguments-differ
        """
        :param mu: occupancy-dependent per-request completion rate. A scalar
            (occupancy-independent, reduces exactly to classical M/M/k/N), or
            a callable ``mu(occupancy) -> rate`` for occupancy in ``1..k``.
        """
        if callable(mu):
            self.mu_fn = mu
        else:
            if mu <= 0:
                raise ValueError(f"mu must be positive, got {mu}")
            self.mu_fn = lambda _occ, _mu=float(mu): _mu
        self.is_servers_set = True

    def _death_rate(self, j: int) -> float:
        """Total completion rate out of state j (occupancy = min(j, k))."""
        occ = min(j, self.k)
        return occ * self.mu_fn(occ)

    def _solve_pi(self) -> np.ndarray:
        """
        Closed (product) form for the state-dependent birth-death chain,
        truncated at N: pi_j = pi_0 * prod_{i=1}^{j} (lam / death_rate(i)).
        Computed in log-space to avoid under/overflow for large N.
        """
        if self._pi is not None:
            return self._pi
        self._check_if_servers_and_sources_set()
        lam = self.l
        log_pi = np.zeros(self.N + 1)
        log_ratio = 0.0
        for j in range(1, self.N + 1):
            log_ratio += np.log(lam) - np.log(self._death_rate(j))
            log_pi[j] = log_ratio
        pi = np.exp(log_pi - log_pi.max())
        pi /= pi.sum()
        self._pi = pi
        return pi

    def get_n_moments(self, num: int = 4) -> list[float]:
        """Exact raw moments of N (number in system), direct summation over pi."""
        pi = self._solve_pi()
        j = np.arange(len(pi))
        return [float((pi * j**p).sum()) for p in range(1, num + 1)]

    def get_p_wait(self) -> float:
        """Exact probability that a PASTA-arriving tagged customer must wait (occupancy already at cap)."""
        pi = self._solve_pi()
        return float(pi[self.k :].sum())

    def get_w(self, num: int = 4) -> list[float]:
        """
        Exact raw moments of the waiting time W (queueing delay before a
        tagged customer is admitted into the active pool) -- a mixture of
        Erlang(j-k+1, k*mu(k)) moments over j >= k, weighted by the PASTA-seen
        stationary occupancy pi_j (see module docstring).
        """
        pi = self._solve_pi()
        k = self.k
        rate = self._death_rate(k)  # = k * mu(k), constant for every j >= k
        w = [0.0] * num
        for j in range(k, len(pi)):
            if pi[j] < _NEGLIGIBLE_PROB:
                continue
            moments = ErlangDistribution.calc_theory_moments(ErlangParams(r=j - k + 1, mu=rate), num)
            for idx in range(num):
                w[idx] += pi[j] * moments[idx]
        return w

    def get_tail(self, t: float) -> float:
        """Exact P(W > t)."""
        pi = self._solve_pi()
        k = self.k
        rate = self._death_rate(k)
        total = 0.0
        for j in range(k, len(pi)):
            if pi[j] < _NEGLIGIBLE_PROB:
                # also protects ErlangDistribution.get_cdf's math.factorial(r-1)
                # from OverflowError at large r -- the skipped mass is already
                # negligible (pi decays geometrically for j > k in a stable system).
                continue
            total += pi[j] * ErlangDistribution.get_tail(ErlangParams(r=j - k + 1, mu=rate), t)
        return total

    def get_cdf(self, t: float) -> float:
        """Exact P(W <= t)."""
        return 1.0 - self.get_tail(t)

    def run(self, num_of_moments: int = 4) -> QueueResults:
        """Run the full calculation; returns raw W-moments, state probabilities, utilization."""
        start = self._measure_time()
        pi = self._solve_pi()
        w = self.get_w(num_of_moments)
        utilization = self.l / self._death_rate(self.k)
        result = QueueResults(p=pi.tolist(), w=w, utilization=utilization)
        self._set_duration(result, start)
        return result
