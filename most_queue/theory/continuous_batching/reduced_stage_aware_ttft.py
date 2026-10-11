"""Exact iid waiting-mark integration and decoder symmetry for token TTFT.

Deferred decisions preserve the product law of unconsumed FIFO marks, not
strong lumpability for arbitrary waiting words. Symmetric phase counting is
classical; see He and Alfa (2018), doi:10.1287/ijoc.2017.0759, Sections 1--2.
Their independent PH-server construction is NOT our shared-cost token model.
The policy, physical costs, rewards and B=1 PK shortcut remain those of
StageAwareTtftCalc. B>1 remains finite loss, not an infinite-queue certificate.
"""

# Explicit physical costs and complexity guards are part of the API.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.special import gammainc
from scipy.stats import poisson

from most_queue.theory.continuous_batching.stage_aware_ttft import StageAwareTtftCalc, StageAwareTtftResults


@dataclass
class ReducedStageAwareTtftResults(StageAwareTtftResults):
    """Same accepted-flow rewards, plus enumerated admission-word count."""

    transition_words: int = 0


class ReducedStageAwareTtftCalc(StageAwareTtftCalc):
    """Track waiting count and active stages without approximating iid marks.

    Only decoders may be reordered. FCFS prefills retain their relative order.
    canonical_decoders=False isolates waiting-mark integration for ablation.
    The reduction is for unconditional iid-law predictions, not conditioning
    on actually observed queued lengths. No correlated or mark-based admission.
    Complexity remains combinatorial in active cap and attained-token support.
    """

    def __init__(
        self,
        batch_cap=2,
        token_budget=4,
        queue_capacity=6,
        max_states=20000,
        max_transition_words=2000000,
        canonical_decoders=True,
    ):
        if not isinstance(canonical_decoders, bool):
            raise ValueError("canonical_decoders must be a bool")
        super().__init__(batch_cap, token_budget, queue_capacity, max_states, max_transition_words)
        self.canonical_decoders = canonical_decoders

    def run(self):
        """Return exact finite-model rewards, or the inherited infinite B1 PK."""
        result = super().run()
        if isinstance(result, ReducedStageAwareTtftResults):
            return result
        return ReducedStageAwareTtftResults(**vars(result))

    def _canonical(self, active):
        if not self.canonical_decoders:
            return active
        decoders = sorted((k, q) for k, q in active if q >= self.law[k][0])
        prefills = tuple((k, q) for k, q in active if q < self.law[k][0])
        return tuple(decoders) + prefills

    def _kernel(self, probs):
        """Build reachable kernel and six rewards; private diagnostic interface."""
        states, index = [((), 0)], {((), 0): 0}
        rows, cols, entries, rewards = [], [], [], []
        word_cache, iteration_cache, arrival_cache = {}, {}, {}
        words_used = 0
        cursor = 0
        while cursor < len(states):
            active, waiting = states[cursor]
            targets = defaultdict(float)
            if not active:
                reward = (1 / self.l, 1.0, 0.0, 0.0, 0.0, 0.0)
                words_used += len(probs)
                for k, prob in enumerate(probs):
                    targets[(((k, 0),), 0)] += prob
            else:
                if active not in iteration_cache:
                    iteration_cache[active] = self._iteration(active)
                duration, survivors = iteration_cache[active]
                space = self.queue_capacity - waiting
                # Every arrival-count branch costs at least one planned word.
                # Reject impossible budgets before allocating O(K) arrays.
                if words_used + space + 1 > self.max_transition_words:
                    raise ValueError("transition word budget exceeded; reduce capacity")
                cache_key = (duration, space)
                if cache_key not in arrival_cache:
                    x = self.l * duration
                    if not np.isfinite(x):
                        raise ValueError("arrival holding product overflow")
                    js = np.arange(1, space + 1)
                    tails = gammainc(js, x)
                    accepted = float(tails.sum())
                    new_area = max(0.0, float(np.sum(duration * tails - js / self.l * gammainc(js + 1, x))))
                    masses = poisson.pmf(np.arange(space + 1), x)
                    masses[-1] = poisson.sf(space - 1, x)
                    arrival_cache[cache_key] = accepted, new_area, masses
                accepted, new_area, masses = arrival_cache[cache_key]
                unfirst = waiting + sum(q <= self.law[k][0] for k, q in active)
                reward = (
                    duration,
                    accepted,
                    unfirst * duration + new_area,
                    waiting * duration + new_area,
                    (waiting + len(active)) * duration + new_area,
                    duration,
                )
                vacancies = self.batch_cap - len(survivors)
                for count, mass in enumerate(masses):
                    fill = min(vacancies, waiting + count)
                    planned = len(probs) ** fill
                    words_used += planned
                    if words_used > self.max_transition_words:
                        raise ValueError("transition word budget exceeded; reduce support or active cap")
                    if mass == 0:
                        continue
                    if fill not in word_cache:
                        word_cache[fill] = tuple(
                            (tuple((k, 0) for k in word), math.prod(probs[k] for k in word))
                            for word in itertools.product(range(len(probs)), repeat=fill)
                        )
                    for admitted, probability in word_cache[fill]:
                        target = (self._canonical(survivors + admitted), waiting + count - fill)
                        targets[target] += float(mass * probability)
            if words_used > self.max_transition_words:
                raise ValueError("transition word budget exceeded; reduce support or active cap")
            for target, probability in targets.items():
                if probability == 0:
                    continue
                if target not in index:
                    if len(states) >= self.max_states:
                        raise ValueError("state budget exceeded; reduce support or active cap")
                    index[target] = len(states)
                    states.append(target)
                rows.append(cursor)
                cols.append(index[target])
                entries.append(probability)
            rewards.append(reward)
            cursor += 1
        matrix = csr_matrix((entries, (rows, cols)), shape=(len(states), len(states)))
        return states, matrix, np.asarray(rewards), words_used

    def _finite(self, probs, load, started):
        states, matrix, rewards, words_used = self._kernel(probs)
        transient = matrix[1:, 1:]
        visits = spsolve(eye(len(states) - 1, format="csc") - transient.T.tocsc(), matrix[0, 1:].toarray().ravel())
        if not np.all(np.isfinite(visits)) or np.min(visits) < -1e-9:
            raise ValueError("ill-conditioned regenerative solve")
        pi = np.r_[1.0, np.maximum(visits, 0.0)]
        pi /= pi.sum()
        residual = float(np.max(np.abs(pi @ matrix - pi)))
        if residual > 1e-9 or np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1)) > 1e-10:
            raise ValueError("stationary or stochastic balance failed")
        holding, accepted, ttft_area, wait_area, population_area, busy = pi @ rewards
        if accepted <= 0 or holding <= 0:
            raise ValueError("invalid regenerative normalization")
        return ReducedStageAwareTtftResults(
            w=[float(wait_area / accepted)],
            v=[float(population_area / accepted)],
            utilization=float(busy / holding),
            ttft_mean=float(ttft_area / accepted),
            blocking_probability=float(np.clip(1 - accepted / (self.l * holding), 0.0, 1.0)),
            accepted_rate=float(accepted / holding),
            states=len(states),
            stationary_residual=residual,
            conservative_load=load,
            model="finite waiting-capacity loss semi-Markov, exact iid reduction",
            duration=time.perf_counter() - started,
            transition_words=words_used,
        )
