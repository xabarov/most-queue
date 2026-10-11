"""Stage-resolved semi-Markov mean TTFT for a specified boundary scheduler.

General methods are classical: renewal rewards, Little's law and M/G/1 PK;
see Bekker (2005), doi:10.6100/IR598926, Ch.1--2, and Zhang (2009),
https://repository.gatech.edu/bitstreams/7e8688af-edf1-4c43-a81f-edae8dc4d4c2/download.
Iteration costs/policy match the explicit L3 DES contract, not a real GPU.
B=1 is infinite FCFS; B>1 is EXACT finite LOSS, not exact infinite queue.
"""

# Explicit physical costs and state budgets are part of the model contract.
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

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue


@dataclass
class StageAwareTtftResults(QueueResults):
    """Accepted-arrival mean rewards; no tail/quantile or truncation certificate."""

    ttft_mean: float = 0.0
    blocking_probability: float = 0.0
    accepted_rate: float = 0.0
    states: int = 0
    stationary_residual: float = 0.0
    conservative_load: float = 0.0
    model: str = ""


class StageAwareTtftCalc(BaseQueue):
    """Ordered attained-token states with Poisson arrivals during iterations.

    queue_capacity counts waiting jobs during an iteration; even an unused
    active slot cannot admit until its boundary. Output length is latent in
    the stochastic state and NEVER used to choose an action. max_states and
    max_transition_words fail explicitly instead of silently coarsening marks.
    """

    def __init__(self, batch_cap=2, token_budget=4, queue_capacity=6, max_states=20000, max_transition_words=2000000):
        for name, value in (
            ("batch_cap", batch_cap),
            ("queue_capacity", queue_capacity),
            ("max_states", max_states),
            ("max_transition_words", max_transition_words),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == "queue_capacity" else 1):
                raise ValueError(f"{name} must be a valid integer")
        if token_budget is not None and (
            isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < batch_cap
        ):
            raise ValueError("token_budget must be None or an integer >= batch_cap")
        super().__init__(n=1, buffer=queue_capacity)
        self.batch_cap, self.token_budget = batch_cap, token_budget
        self.queue_capacity, self.max_states = queue_capacity, max_states
        self.max_transition_words = max_transition_words
        self.l = None
        self.law = None
        self.alpha = self.beta = self.gamma = None

    def set_sources(self, l):
        """Set a positive finite Poisson rate, independent of iid joint marks."""
        if not np.isfinite(l) or l <= 0:
            raise ValueError("arrival rate must be positive and finite")
        self.l = float(l)
        self.is_sources_set = True

    def set_servers(self, law, alpha, beta, gamma):
        """Set finite support [(input_tokens, output_tokens, probability), ...]."""
        values = np.asarray(law, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
            raise ValueError("law must be a nonempty three-column array")
        if not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("lengths and probabilities must be positive and finite")
        if (
            np.any(values[:, :2] != np.floor(values[:, :2]))
            or np.any(values[:, :2] > 1000000)
            or not math.isclose(float(values[:, 2].sum()), 1.0, abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError("law must have positive integer lengths and probabilities summing to one")
        costs = np.array([alpha, beta, gamma], dtype=float)
        if not np.all(np.isfinite(costs)) or alpha <= 0 or min(beta, gamma) < 0:
            raise ValueError("finite alpha>0 and beta,gamma>=0 required")
        combined = defaultdict(float)
        for n, m, p in values:
            combined[int(n), int(m)] += float(p)
        self.law = tuple((n, m, p) for (n, m), p in sorted(combined.items()))
        self.alpha, self.beta, self.gamma = costs.tolist()
        self.is_servers_set = True

    def isolated_times(self):
        """Return full service and time through first decode for each joint type."""
        if not self.is_servers_set:
            raise ValueError("set_servers must be called first")
        full, first = [], []
        for n, m, _ in self.law:
            chunks = 1 if self.token_budget is None else math.ceil(n / self.token_budget)
            budget = n if self.token_budget is None else self.token_budget
            cache_sum = budget * (chunks - 1) * chunks / 2 + n
            pre = chunks * self.alpha + self.beta * n + self.gamma * cache_sum
            one = self.alpha + self.beta + self.gamma * (n + 1)
            total = pre + m * (self.alpha + self.beta) + self.gamma * (m * n + m * (m + 1) / 2)
            if not np.isfinite(total):
                raise ValueError("isolated service overflow")
            full.append(total)
            first.append(pre + one)
        return np.array(full), np.array(first)

    def _iteration(self, active):
        """Advance ordered attained tokens; return holding time and survivors."""
        decoders = sum(q >= self.law[k][0] for k, q in active)
        remaining = math.inf if self.token_budget is None else self.token_budget - decoders
        duration = self.alpha
        survivors = []
        for k, q in active:
            n, m, _ = self.law[k]
            if q >= n:
                amount = 1
            else:
                amount = int(min(n - q, remaining))
                remaining -= amount
            if amount:
                duration += self.beta * amount + self.gamma * (q + amount)
            if q + amount < n + m:
                survivors.append((k, q + amount))
        if not np.isfinite(duration):
            raise ValueError("iteration duration overflow")
        return duration, tuple(survivors)

    def run(self):
        """Compute first W/V moments and mean TTFT under the explicit contract."""
        self._check_if_servers_and_sources_set()
        started = time.perf_counter()
        full, first = self.isolated_times()
        probs = self._mark_probabilities()
        load = self.l * float(probs @ full)
        if self.batch_cap == 1:
            if load >= 1:
                raise ValueError("infinite B=1 FCFS requires lambda*E[full_service]<1")
            wait = self.l * float(probs @ full**2) / (2 * (1 - load))
            return StageAwareTtftResults(
                w=[wait],
                v=[wait + float(probs @ full)],
                utilization=load,
                ttft_mean=wait + float(probs @ first),
                accepted_rate=self.l,
                conservative_load=load,
                model="infinite B=1 M/G/1 FCFS",
                duration=time.perf_counter() - started,
            )
        return self._finite(probs, load, started)

    def _mark_probabilities(self):
        """Return configured iid masses (overridable normalization policy)."""
        return np.array([row[2] for row in self.law])

    def _finite(self, probs, load, started):
        states, index = [((), ())], {((), ()): 0}
        rows, cols, entries, rewards = [], [], [], []
        words_used = 0
        cursor = 0
        while cursor < len(states):
            active, queue = states[cursor]
            targets = defaultdict(float)
            if not active:
                duration = 1 / self.l
                reward = (duration, 1.0, 0.0, 0.0, 0.0, 0.0)
                for k, prob in enumerate(probs):
                    targets[(((k, 0),), ())] += prob
            else:
                duration, survivors = self._iteration(active)
                space = self.queue_capacity - len(queue)
                x = self.l * duration
                accepted = sum(gammainc(j, x) for j in range(1, space + 1))
                new_area = sum(duration * gammainc(j, x) - j / self.l * gammainc(j + 1, x) for j in range(1, space + 1))
                unfirst = len(queue) + sum(q <= self.law[k][0] for k, q in active)
                reward = (
                    duration,
                    accepted,
                    unfirst * duration + max(0.0, new_area),
                    len(queue) * duration + max(0.0, new_area),
                    (len(queue) + len(active)) * duration + max(0.0, new_area),
                    duration,
                )
                planned = sum(len(probs) ** n for n in range(space + 1))
                words_used += planned
                if words_used > self.max_transition_words:
                    raise ValueError("transition word budget exceeded; reduce support or capacity")
                for n in range(space + 1):
                    mass = poisson.pmf(n, x) if n < space else poisson.sf(n - 1, x)
                    for word in itertools.product(range(len(probs)), repeat=n):
                        probability = float(mass * np.prod([probs[k] for k in word]))
                        if probability == 0:
                            continue
                        waiting = queue + word
                        fill = min(self.batch_cap - len(survivors), len(waiting))
                        target = (survivors + tuple((k, 0) for k in waiting[:fill]), waiting[fill:])
                        targets[target] += probability
            for target, probability in targets.items():
                if target not in index:
                    if len(states) >= self.max_states:
                        raise ValueError("state budget exceeded; reduce support or capacity")
                    index[target] = len(states)
                    states.append(target)
                rows.append(cursor)
                cols.append(index[target])
                entries.append(probability)
            rewards.append(reward)
            cursor += 1
        matrix = csr_matrix((entries, (rows, cols)), shape=(len(states), len(states)))
        # One visit to idle per regenerative cycle; expected nonidle visits.
        transient = matrix[1:, 1:]
        visits = spsolve(eye(len(states) - 1, format="csc") - transient.T.tocsc(), matrix[0, 1:].toarray().ravel())
        if not np.all(np.isfinite(visits)) or np.min(visits) < -1e-9:
            raise ValueError("ill-conditioned regenerative solve")
        pi = np.r_[1.0, np.maximum(visits, 0.0)]
        pi /= pi.sum()
        residual = float(np.max(np.abs(pi @ matrix - pi)))
        if residual > 1e-9 or np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1)) > 1e-10:
            raise ValueError("stationary or stochastic balance failed")
        holding, accepted, ttft_area, wait_area, population_area, busy = pi @ np.asarray(rewards)
        return StageAwareTtftResults(
            w=[float(wait_area / accepted)],
            v=[float(population_area / accepted)],
            utilization=float(busy / holding),
            ttft_mean=float(ttft_area / accepted),
            blocking_probability=float(np.clip(1 - accepted / (self.l * holding), 0.0, 1.0)),
            accepted_rate=float(accepted / holding),
            states=len(states),
            stationary_residual=residual,
            conservative_load=load,
            model="finite waiting-capacity loss semi-Markov",
            duration=time.perf_counter() - started,
        )
