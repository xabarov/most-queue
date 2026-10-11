"""Infinite iid token queue with regenerative model-truncation bounds.

Classical ratio/exit methodology: Infanger and Glynn (2022),
https://arxiv.org/abs/2208.14406, Section 2; Glynn and Zheng (2025),
https://arxiv.org/abs/2505.03157, Section 2. The service-credit domination
and its application to this explicit token scheduler are derived in the
companion model documentation. Bounds are evaluated in ordinary floating
point: NOT machine-verified rounding enclosures, NOT TTFT quantiles.
"""

# Physical costs, truncation and explicit complexity budgets are API inputs.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve
from scipy.stats import poisson

from most_queue.theory.continuous_batching.reduced_stage_aware_ttft import (
    ReducedStageAwareTtftCalc,
    ReducedStageAwareTtftResults,
)


@dataclass
class InfiniteStageAwareTtftResults(ReducedStageAwareTtftResults):
    """Mean interval midpoints and analytic, floating-evaluated error bounds."""

    mean_bounds: dict = field(default_factory=dict)
    utilization_bounds: tuple = (0.0, 0.0)
    truncation: int | None = None
    dominating_load: float | None = None
    cycle_exit_probability: float | None = None
    waiting_tail_upper: float | None = None
    cycle_time_lower: float | None = None
    cycle_time_remainder_upper: float | None = None
    roundoff_certified: bool = False


class InfiniteStageAwareTtftCalc(ReducedStageAwareTtftCalc):
    """Infinite waiting room, killed count kernel and busy-period bounds.

    truncation is a NUMERICAL domain, never a physical loss capacity. Requests
    beyond it are not discarded from the model: their future cycle rewards
    are bounded. For B>1 require lambda*E[D]<1, where D is the per-token
    service-credit upper bound. This is sufficient, NOT the exact stability
    boundary. A refusal cannot diagnose instability of the batch scheduler.
    Only iid finite-support marks and mark-blind FIFO boundary admission.
    """

    def __init__(self, batch_cap=2, token_budget=4, truncation=8, max_states=20000, max_transition_words=2000000):
        super().__init__(batch_cap, token_budget, truncation, max_states, max_transition_words)
        self.truncation = truncation

    def dominating_work(self):
        """Return per-type total credits D=(alpha+beta)L+gamma*L(L+1)/2."""
        if not self.is_servers_set:
            raise ValueError("set_servers must be called first")
        lengths = np.array([n + m for n, m, _ in self.law], dtype=float)
        work = (self.alpha + self.beta) * lengths + self.gamma * lengths * (lengths + 1) / 2
        if not np.all(np.isfinite(work)) or np.any(work <= 0):
            raise ValueError("dominating work overflow")
        return work

    @staticmethod
    def _overshoot_moments(mean, threshold, offset):
        """Unconditional E[1,H,H^2; N>=threshold], H=N+offset, N Poisson."""
        probability = float(poisson.sf(threshold - 1, mean))
        first = mean * float(poisson.sf(threshold - 2, mean))
        second = mean**2 * float(poisson.sf(threshold - 3, mean)) + first
        return np.array(
            (probability, first + offset * probability, second + 2 * offset * first + offset**2 * probability)
        )

    def _remainder(self, moments, work, probs):
        """Bound continuation until idle using a dominating workload busy period."""
        probability, first, second = moments
        mean = float(probs @ work)
        second_work = float(probs @ work**2)
        variance = max(0.0, second_work - mean**2)
        active = self.batch_cap * float(work.max())
        x = active * probability + mean * first
        x2 = active**2 * probability + (2 * active * mean + variance) * first + mean**2 * second
        slack = 1 - self.l * mean
        duration = x / slack
        area = x2 / (2 * slack) + self.l * second_work * x / (2 * slack**2)
        waiting = area / float(work.min())
        population = waiting + self.batch_cap * duration
        return np.array((duration, population, waiting, population))

    def _killed_kernel(self, probs, work):
        """Build the genuine restricted infinite kernel, rewards and exit bounds."""
        states, index = [((), 0)], {((), 0): 0}
        rows, cols, entries, rewards, exits, tail_times = [], [], [], [], [], []
        word_cache, iteration_cache, arrival_cache = {}, {}, {}
        words_used, cursor = 0, 0
        while cursor < len(states):
            active, waiting = states[cursor]
            targets = defaultdict(float)
            if not active:
                reward = (1 / self.l, 0.0, 0.0, 0.0)
                exit_reward = np.zeros(5)
                tail_time = 0.0
                words_used += len(probs)
                for k, probability in enumerate(probs):
                    targets[(((k, 0),), 0)] += probability
            else:
                if active not in iteration_cache:
                    iteration_cache[active] = self._iteration(active)
                duration, survivors = iteration_cache[active]
                vacancies = self.batch_cap - len(survivors)
                # Counts <= N-waiting+vacancies remain inside; larger counts
                # all fill vacancies, and their unbounded overshoot is retained.
                last = self.truncation - waiting + vacancies
                if words_used + last + 1 > self.max_transition_words:
                    raise ValueError("transition word budget exceeded; reduce truncation")
                mean = self.l * duration
                if not np.isfinite(mean):
                    raise ValueError("arrival holding product overflow")
                cache_key = (duration, waiting, vacancies)
                if cache_key not in arrival_cache:
                    masses = poisson.pmf(np.arange(last + 1), mean)
                    moments = self._overshoot_moments(mean, last + 1, waiting - vacancies)
                    arrival_cache[cache_key] = masses, np.r_[moments[0], self._remainder(moments, work, probs)]
                masses, exit_reward = arrival_cache[cache_key]
                new_area = self.l * duration**2 / 2
                unfirst = waiting + sum(q <= self.law[k][0] for k, q in active)
                reward = (
                    duration,
                    unfirst * duration + new_area,
                    waiting * duration + new_area,
                    (waiting + len(active)) * duration + new_area,
                )
                # Real waiting count may exceed N INSIDE a retained iteration.
                # Counting its whole duration on any such burst is conservative.
                tail_time = duration * float(poisson.sf(self.truncation - waiting, mean))
                for count, mass in enumerate(masses):
                    fill = min(vacancies, waiting + count)
                    words_used += len(probs) ** fill
                    if words_used > self.max_transition_words:
                        raise ValueError("transition word budget exceeded; reduce support or active cap")
                    if fill not in word_cache:
                        word_cache[fill] = tuple(
                            (tuple((k, 0) for k in word), math.prod(probs[k] for k in word))
                            for word in itertools.product(range(len(probs)), repeat=fill)
                        )
                    # Keep structurally reachable states even for underflowed
                    # positive probabilities, so underflow cannot hide complexity.
                    for admitted, probability in word_cache[fill]:
                        target = (self._canonical(survivors + admitted), waiting + count - fill)
                        targets[target] += float(mass * probability)
            if words_used > self.max_transition_words:
                raise ValueError("transition word budget exceeded; reduce support or active cap")
            for target, probability in targets.items():
                if target not in index:
                    if len(states) >= self.max_states:
                        raise ValueError("state budget exceeded; reduce support or active cap")
                    index[target] = len(states)
                    states.append(target)
                rows.append(cursor)
                cols.append(index[target])
                entries.append(probability)
            rewards.append(reward)
            exits.append(exit_reward)
            tail_times.append(tail_time)
            cursor += 1
        matrix = csr_matrix((entries, (rows, cols)), shape=(len(states), len(states)))
        return states, matrix, np.asarray(rewards), np.asarray(exits), np.asarray(tail_times), words_used

    def _prepare_bound(self, probs, work):
        """Validate the workload condition; subclasses may supply another bound."""
        load = self.l * float(probs @ work)
        if not np.isfinite(load) or load >= 1:
            raise ValueError("sufficient stability certificate requires lambda*E[dominating_work]<1; NOT instability")
        return load

    def run(self):
        """Return mean midpoints and bounds; B1 uses exact infinite FCFS PK."""
        self._check_if_servers_and_sources_set()
        started = time.perf_counter()
        if self.batch_cap == 1:
            result = super().run()
            bounds = {"w": (result.w[0], result.w[0]), "v": (result.v[0], result.v[0])}
            bounds["ttft"] = (result.ttft_mean, result.ttft_mean)
            return InfiniteStageAwareTtftResults(
                **vars(result), mean_bounds=bounds, utilization_bounds=(result.utilization, result.utilization)
            )
        probs = self._mark_probabilities()
        work = self.dominating_work()
        dominating_load = self._prepare_bound(probs, work)
        states, matrix, rewards, exits, tail_times, words_used = self._killed_kernel(probs, work)
        transient = matrix[1:, 1:]
        equation = eye(len(states) - 1, format="csc") - transient.T.tocsc()
        source = matrix[0, 1:].toarray().ravel()
        visits = spsolve(equation, source)
        if not np.all(np.isfinite(visits)) or np.min(visits) < -1e-9:
            raise ValueError("ill-conditioned regenerative solve")
        residual = float(np.max(np.abs(equation @ visits - source)))
        mass_residual = float(np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() + exits[:, 0] - 1)))
        if residual > 1e-9 or mass_residual > 1e-10:
            raise ValueError("regenerative or killed mass balance failed")
        visits = np.r_[1.0, np.maximum(visits, 0.0)]
        retained = visits @ rewards
        exit_probability, *remainder = visits @ exits
        remainder = np.asarray(remainder)
        if not np.all(np.isfinite(retained)) or not np.all(np.isfinite(remainder)):
            raise ValueError("cycle rewards overflow")
        lower_time, upper_time = retained[0], retained[0] + remainder[0]
        bounds = {
            key: (
                float(retained[i] / (self.l * upper_time)),
                float((retained[i] + remainder[i]) / (self.l * lower_time)),
            )
            for key, i in (("ttft", 1), ("w", 2), ("v", 3))
        }
        # Idle holding is exactly 1/lambda and occurs once in every full cycle.
        utilization = (max(0.0, 1 - 1 / (self.l * lower_time)), min(1.0, 1 - 1 / (self.l * upper_time)))
        full, _ = self.isolated_times()
        return InfiniteStageAwareTtftResults(
            w=[sum(bounds["w"]) / 2],
            v=[sum(bounds["v"]) / 2],
            ttft_mean=sum(bounds["ttft"]) / 2,
            utilization=sum(utilization) / 2,
            accepted_rate=self.l,
            states=len(states),
            stationary_residual=residual,
            conservative_load=self.l * float(probs @ full),
            model="infinite iid queue; analytic truncation bounds, floating-point evaluation",
            duration=time.perf_counter() - started,
            transition_words=words_used,
            mean_bounds=bounds,
            utilization_bounds=utilization,
            truncation=self.truncation,
            dominating_load=dominating_load,
            cycle_exit_probability=float(exit_probability),
            waiting_tail_upper=float(min(1.0, (visits @ tail_times + remainder[0]) / lower_time)),
            cycle_time_lower=float(lower_time),
            cycle_time_remainder_upper=float(remainder[0]),
        )
