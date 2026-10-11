"""Finite SMDP approximation of Xu et al., arXiv:2301.12865v3, Eq.5--22.

Separate overflow state represents state_max, not state_max+1. RVI uses
compressed Poisson correlations, not a dense state/action transition tensor.
Finite Bellman optimality and overflow cost do not certify infinite optimality.
"""

import math
from numbers import Integral, Real

import numpy as np
from scipy.stats import poisson

from most_queue.structs import BatchSmdpResults
from most_queue.theory.base_queue import BaseQueue


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


class BatchSmdpCalc(BaseQueue):
    """Optimize the finite overflow approximation with explicit diagnostics.

    Profiles list deterministic times and energies for batches 1,...,B.
    Source affine profiles are a special case; no threshold structure is imposed.
    Action zero means wait until the next Poisson arrival, not a fixed timer.
    """

    def __init__(self, state_max=100, *, delay_weight=1.0, energy_weight=1.0, overflow_cost=100.0):
        super().__init__(n=1)
        self.state_max = _integer(state_max, "state_max")
        if self.state_max > 2000:
            raise ValueError("state_max exceeds the numerical resource limit of 2000")
        self.delay_weight = _real(delay_weight, "delay_weight")
        self.energy_weight = _real(energy_weight, "energy_weight")
        if self.delay_weight == self.energy_weight == 0:
            raise ValueError("at least one objective weight must be positive")
        self.overflow_cost = _real(overflow_cost, "overflow_cost")
        self.rate = None
        self.times = None
        self.energies = None

    def set_sources(self, rate):
        """Set a strictly positive Poisson rate in requests/time unit."""
        self.rate = _real(rate, "rate", positive=True)
        self.is_sources_set = True

    def set_servers(self, batch_times, batch_energies):
        """Set deterministic finite profiles, indexed by batch size minus one."""
        times = [_real(x, "batch_time", positive=True) for x in batch_times]
        energies = [_real(x, "batch_energy") for x in batch_energies]
        if not times or len(times) != len(energies):
            raise ValueError("nonempty time/energy profiles must have equal length")
        if len(times) > min(self.state_max, 128):
            raise ValueError("batch limit must be <= state_max and <= 128")
        self.times, self.energies = np.array(times), np.array(energies)
        self.is_servers_set = True

    def _prepare(self):
        self._check_if_servers_and_sources_set()
        batches = np.arange(1, len(self.times) + 1)
        capacity = float(max(batches / self.times))
        if self.rate >= capacity:
            raise ValueError("arrival rate must be strictly below maximum batch capacity")
        counts = np.arange(self.state_max + 1)
        pmf = poisson.pmf(counts[None, :], (self.rate * self.times)[:, None])
        effective = np.minimum(np.arange(self.state_max + 2), self.state_max)
        sojourn = np.r_[1 / self.rate, self.times]
        area = effective[:, None] * sojourn[None, :]
        area[:, 1:] += self.rate * self.times[None, :] ** 2 / 2
        energy = np.r_[0, self.energies]
        costs = self.delay_weight * area / self.rate + self.energy_weight * energy[None, :]
        costs[-1] += self.overflow_cost * sojourn
        if not np.all(np.isfinite(costs)) or not np.all(np.isfinite(sojourn)) or not np.all(np.isfinite(area)):
            raise ValueError("nonfinite reward/sojourn arithmetic; rescale")
        return pmf, effective, sojourn, area, energy, costs

    def _action(self, state, action):
        state = _integer(state, "state", minimum=0)
        action = _integer(action, "action", minimum=0)
        if state > self.state_max + 1 or action > min(state, len(self.times)):
            raise ValueError("state/action outside finite feasible domain")
        return state, action

    def _row(self, state, action, pmf):
        row = np.zeros(self.state_max + 2)
        if action == 0:
            row[min(state + 1, self.state_max + 1)] = 1
        else:
            base = min(state, self.state_max) - action
            row[base : self.state_max + 1] = pmf[action - 1, : self.state_max - base + 1]
            row[-1] = poisson.sf(self.state_max - base, self.rate * self.times[action - 1])
        if abs(sum(row) - 1) > 1e-11:
            raise ValueError("Poisson transition row lost probability mass")
        return row

    def epoch(self, state, action):
        """Return (transition row, mean time, cost, population area, energy).

        state_max+1 is the abstract overflow label, with effective population
        state_max. Cost includes overflow_cost*time only at that label.
        """
        pmf, _, sojourn, area, energy, costs = self._prepare()
        state, action = self._action(state, action)
        return self._row(state, action, pmf), sojourn[action], costs[state, action], area[state, action], energy[action]

    def _operator(self, bias, tables, eta):
        pmf, effective, sojourn, _, _, costs = tables
        n = self.state_max
        q = np.full_like(costs, np.inf)
        successor = np.minimum(np.arange(n + 2) + 1, n + 1)
        q[:, 0] = costs[:, 0] / sojourn[0] + bias + eta / sojourn[0] * (bias[successor] - bias)
        centered = bias[: n + 1] - bias[-1]
        for action in range(1, len(self.times) + 1):
            # correlation at lag b = sum_k p_k * (h[b+k]-h[overflow]).
            correlation = np.correlate(centered, pmf[action - 1], mode="full")[n:]
            states = np.arange(action, n + 2)
            expectation = bias[-1] + correlation[effective[states] - action]
            q[states, action] = (
                costs[states, action] / sojourn[action]
                + bias[states]
                + eta / sojourn[action] * (expectation - bias[states])
            )
        policy = np.argmin(q, axis=1)
        return q[np.arange(n + 2), policy], policy, q

    def _eta_bound(self):
        bounds = [1 / self.rate]
        for action, period in enumerate(self.times, start=1):
            mean = self.rate * period
            leave = 1 - poisson.pmf(action, mean)
            cumulative = poisson.cdf(action, mean)
            bounds.extend([period / leave if leave else math.inf, period / cumulative if cumulative else math.inf])
        return float(min(bounds))

    def evaluate_policy(self, policy):
        """Evaluate a fixed finite policy by embedded stationary reward ratios.

        Above state_max the proposed REAL queue extension uses the overflow
        action. Its service-epoch drift is checked separately from finite cost.
        Decision and interval-time weights are not a continuous queue PMF.
        """
        tables = self._prepare()
        pmf, _, sojourn, area, energy, costs = tables
        policy = list(policy)
        if len(policy) != self.state_max + 2:
            raise ValueError("policy must include state_max+2 actions, including overflow")
        policy = np.array([self._action(s, a)[1] for s, a in enumerate(policy)])
        transition = np.array([self._row(s, int(a), pmf) for s, a in enumerate(policy)])
        system = transition.T - np.eye(len(policy))
        system[-1] = 1
        rhs = np.zeros(len(policy))
        rhs[-1] = 1
        try:
            stationary = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("stationary policy system is singular; increase precision or revise truncation") from exc
        if min(stationary) < -1e-9 or not np.all(np.isfinite(stationary)):
            raise ValueError("invalid stationary policy solution")
        stationary = np.maximum(stationary, 0)
        stationary /= sum(stationary)
        balance = float(max(abs(stationary @ transition - stationary)))
        if balance > 1e-9:
            raise ValueError("stationary policy balance residual is too large")
        states = np.arange(len(policy))
        periods = sojourn[policy]
        mean_time = float(stationary @ periods)
        selected_costs = costs[states, policy]
        tail_action = int(policy[-1])
        tail_drift = self.rate * self.times[tail_action - 1] - tail_action if tail_action else 1.0
        result = BatchSmdpResults(
            state_max=self.state_max,
            policy=policy.tolist(),
            decision_pmf=stationary.tolist(),
            interval_time_weights=(stationary * periods / mean_time).tolist(),
            mean_epoch_time=mean_time,
            average_cost=float(stationary @ selected_costs / mean_time),
            response_proxy=float(stationary @ area[states, policy] / (mean_time * self.rate)),
            power=float(stationary @ energy[policy] / mean_time),
            overflow_cost_rate=float(stationary[-1] * selected_costs[-1] / mean_time),
            overflow_epoch_probability=float(stationary[-1]),
            overflow_time_probability=float(stationary[-1] * periods[-1] / mean_time),
            tail_drift=float(tail_drift),
            tail_extension_stable=bool(tail_action and tail_drift < 0),
            balance_residual=balance,
            diagnostics=[
                "Finite overflow approximation, NOT certified infinite-queue optimality.",
                "Overflow cost contribution is a diagnostic, not a truncation error bound.",
                "Decision PMF and interval-time weights are not the real queue's time PMF.",
                "Response proxy comes from finite reward area / rate, not an exact latency distribution.",
                "Constant tail service with negative drift stabilizes the specified Poisson extension only.",
                "No hard SLO, idle baseline power, random service, KV dynamics, or GPU calibration.",
            ],
        )
        return result

    def run(self, *, tolerance=1e-8, max_iterations=10000, eta=None):
        """Relative value iteration; ties choose the smallest feasible action.

        Gain brackets concern ONLY the finite approximation. Nonconvergence
        remains explicit; the last policy is evaluated but not called optimal.
        """
        tolerance = _real(tolerance, "tolerance", positive=True)
        max_iterations = _integer(max_iterations, "max_iterations")
        start = self._measure_time()
        tables = self._prepare()
        bound = self._eta_bound()
        eta = 0.9 * min(tables[2]) if eta is None else _real(eta, "eta", positive=True)
        if eta >= bound:
            raise ValueError("eta must be strictly below every uniformization bound")
        bias = np.zeros(self.state_max + 2)
        for iteration in range(1, max_iterations + 1):
            raw, _, _ = self._operator(bias, tables, eta)
            increments = raw - bias
            span = float(np.ptp(increments))
            bias = raw - raw[0]
            if not np.all(np.isfinite(bias)):
                raise ValueError("nonfinite relative values; rescale")
            if span < tolerance:
                break
        raw, policy, q = self._operator(bias, tables, eta)
        increments = raw - bias
        result = self.evaluate_policy(policy)
        result.iterations = iteration
        result.converged = bool(np.ptp(increments) < tolerance)
        result.gain_lower = float(min(increments))
        result.gain_upper = float(max(increments))
        result.uniformization_eta = float(eta)
        result.bias = (eta * bias).tolist()
        # min_a(c-g*y+P*h-h), computed through the transformed operator.
        residuals = (q - bias[:, None] - result.average_cost) * tables[2][None, :]
        result.bellman_residual = float(max(abs(np.min(residuals, axis=1))))
        if not result.converged:
            result.diagnostics.append("RVI iteration limit reached; policy is NOT certified finite-optimal.")
        if not result.tail_extension_stable:
            result.diagnostics.append(
                "Overflow action has nonnegative real-queue drift: extension NOT certified stable."
            )
        self._set_duration(result, start)
        return result
