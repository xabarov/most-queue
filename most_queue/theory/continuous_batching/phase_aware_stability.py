"""Phase-corrected Foster drift for the specified iid token scheduler.

Poisson-corrector methodology is classical: Glynn and Infanger (2022),
https://arxiv.org/html/2202.10404v1, Section 1 equations 1.1--1.2.
Reachability, multichain extension and physical-time normalization are derived
in docs/research/phase-aware-stability-validation-2026.md. Float linear solves
produce a candidate; Fraction arithmetic verifies each drift inequality for
the exact binary inputs and normalized binary mark probabilities.
"""

# Explicit physical costs and complexity budgets follow the token-model API.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from most_queue.structs import QueueResults
from most_queue.theory.continuous_batching.reduced_stage_aware_ttft import ReducedStageAwareTtftCalc


@dataclass
class PhaseStabilityResults(QueueResults):
    """A stability/escape certificate; no stationary delay estimates."""

    classification: str = "unresolved"
    stable: bool = False
    drift_verified: bool = False
    verification: str = "exact rational row inequalities on normalized binary inputs"
    phase_count: int = 0
    reachable_active_count: int = 0
    transient_phase_count: int = 0
    closed_classes: list = field(default_factory=list)
    capacity: float = 0.0
    saturated_load: float = 0.0
    drift_margin: float | None = None
    drift_margin_exact: str | None = None
    corrector: list = field(default_factory=list)
    row_drifts: list = field(default_factory=list)
    row_drifts_exact: list = field(default_factory=list)
    poisson_residual: float = 0.0
    transition_words: int = 0


class PhaseAwareStabilityCalc(ReducedStageAwareTtftCalc):
    """Check every reachable saturated class, with a finite phase corrector.

    At h>=B the active pool is full and all vacancies are refilled. The
    count increment is Poisson(lambda*d)-departures. All closed SCCs and
    transient absorption drifts are considered; no aperiodicity assumption.
    Strict negative rational row drift certifies positive recurrence from
    empty. Strict positive drift on a closed class certifies a possible
    escape. Critical/ill-conditioned cases remain explicitly unresolved.
    """

    def __init__(self, batch_cap=2, token_budget=4, max_states=20000, max_transition_words=2000000):
        super().__init__(batch_cap, token_budget, 0, max_states, max_transition_words)
        self.buffer = None
        self._phases = self._matrix = self._durations = self._departures = None
        self._exact_rows = self._exact_durations = None

    def _exact_iteration(self, active):
        """Evaluate real cost sums exactly at the represented binary inputs."""
        remaining = (
            math.inf if self.token_budget is None else self.token_budget - sum(q >= self.law[k][0] for k, q in active)
        )
        duration = Fraction(self.alpha)
        survivors = []
        for k, attained in active:
            prompt, output, _ = self.law[k]
            amount = 1 if attained >= prompt else int(min(prompt - attained, remaining))
            if attained < prompt:
                remaining -= amount
            if amount:
                duration += Fraction(self.beta) * amount + Fraction(self.gamma) * (attained + amount)
            if attained + amount < prompt + output:
                survivors.append((k, attained + amount))
        # Fraction -> float raises before isfinite when the exact sum is too big.
        if duration > Fraction(np.finfo(float).max):
            raise ValueError("iteration duration overflow")
        return duration, tuple(survivors)

    def _saturated_kernel(self):
        """Enumerate exact reachable active phases and their full-refill kernel."""
        probabilities = [Fraction(p) for _, _, p in self.law]
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        active_states, seen = [()], {()}
        iterations, words, words_used = {}, {}, 0
        cursor = 0
        while cursor < len(active_states):
            active = active_states[cursor]
            if not active:
                targets = [((k, 0),) for k in range(len(self.law))]
                words_used += len(targets)
            else:
                duration, survivors = self._exact_iteration(active)
                iterations[active] = duration, survivors
                vacancies = self.batch_cap - len(survivors)
                if vacancies + 1 > self.max_transition_words - words_used:
                    raise ValueError("phase transition word budget exceeded")
                targets = []
                for fill in range(vacancies + 1):
                    planned = len(self.law) ** fill
                    words_used += planned
                    if words_used > self.max_transition_words:
                        raise ValueError("phase transition word budget exceeded")
                    if fill not in words:
                        words[fill] = tuple(
                            (tuple((k, 0) for k in word), math.prod(probabilities[k] for k in word))
                            for word in itertools.product(range(len(self.law)), repeat=fill)
                        )
                    targets.extend(self._canonical(survivors + admitted) for admitted, _ in words[fill])
            if words_used > self.max_transition_words:
                raise ValueError("phase transition word budget exceeded")
            for target in targets:
                if target not in seen:
                    if len(active_states) >= self.max_states:
                        raise ValueError("reachable active phase state budget exceeded")
                    seen.add(target)
                    active_states.append(target)
            cursor += 1
        phases = [active for active in active_states if len(active) == self.batch_cap]
        index = {active: i for i, active in enumerate(phases)}
        rows, cols, entries, exact_rows, durations, departures = [], [], [], [], [], []
        for i, active in enumerate(phases):
            duration, survivors = iterations[active]
            vacancies = self.batch_cap - len(survivors)
            targets = defaultdict(Fraction)
            for admitted, probability in words[vacancies]:
                targets[index[self._canonical(survivors + admitted)]] += probability
            assert sum(targets.values()) == 1
            exact_rows.append(dict(targets))
            durations.append(duration)
            departures.append(vacancies)
            for j, probability in targets.items():
                mass = float(probability)
                if mass == 0:
                    raise ValueError("phase probability underflow; exact graph cannot be silently pruned")
                rows.append(i)
                cols.append(j)
                entries.append(mass)
        matrix = csr_matrix((entries, (rows, cols)), shape=(len(phases), len(phases)))
        return (
            phases,
            matrix,
            durations,
            np.asarray(departures, dtype=float),
            exact_rows,
            len(active_states),
            words_used,
        )

    @staticmethod
    def _class_analysis(matrix, durations, departures, rate):
        """Class stationary drifts and harmonic extension to transient phases."""
        _, labels = connected_components(matrix, directed=True, connection="strong")
        classes, anchors, delta = [], [], np.zeros(matrix.shape[0])
        closed = np.zeros(matrix.shape[0], dtype=bool)
        for label in sorted(set(labels.tolist())):
            indices = np.flatnonzero(labels == label)
            if any(labels[j] != label for i in indices for j in matrix.getrow(i).indices):
                continue
            sub = matrix[indices][:, indices]
            equation = (eye(len(indices), format="csc") - sub.T).tolil()
            equation[0, :] = np.ones(len(indices))
            rhs = np.zeros(len(indices))
            rhs[0] = 1
            stationary = np.asarray(spsolve(equation.tocsc(), rhs))
            if not np.all(np.isfinite(stationary)) or np.min(stationary) < -1e-10:
                raise ValueError("ill-conditioned class stationary solve")
            stationary = np.maximum(stationary, 0)
            stationary /= stationary.sum()
            holding = float(stationary @ durations[indices])
            completed = float(stationary @ departures[indices])
            if holding <= 0 or completed <= 0:
                raise ValueError("invalid saturated class throughput")
            drift = rate * holding - completed
            delta[indices] = drift
            closed[indices] = True
            anchors.append(int(indices[0]))
            classes.append(
                {
                    "indices": indices.tolist(),
                    "stationary": stationary.tolist(),
                    "mean_holding": holding,
                    "mean_departures": completed,
                    "capacity": completed / holding,
                    "mean_count_drift": drift,
                }
            )
        transient = np.flatnonzero(~closed)
        if len(transient):
            block = matrix[transient][:, transient]
            delta[transient] = spsolve(eye(len(transient), format="csc") - block, matrix[transient] @ delta)
        return classes, anchors, delta, len(transient)

    def run(self):
        """Return an exactly checked row-drift certificate or explicit ambiguity."""
        self._check_if_servers_and_sources_set()
        started = time.perf_counter()
        phases, matrix, exact_durations, departures, exact_rows, active_count, words = self._saturated_kernel()
        durations = np.array([float(value) for value in exact_durations])
        classes, anchors, delta, transient_count = self._class_analysis(matrix, durations, departures, self.l)
        raw = self.l * durations - departures
        equation = (eye(len(phases), format="csc") - matrix).tolil()
        rhs = raw - delta
        for anchor in anchors:
            equation[anchor, :] = 0
            equation[anchor, anchor] = 1
            rhs[anchor] = 0
        corrector = np.asarray(spsolve(equation.tocsc(), rhs))
        if not np.all(np.isfinite(corrector)) or not np.all(np.isfinite(delta)):
            raise ValueError("ill-conditioned phase corrector solve")
        rational = [Fraction(float(value)) for value in corrector]
        exact_drifts = [
            Fraction(self.l) * exact_durations[i]
            - int(departures[i])
            + sum(probability * rational[j] for j, probability in exact_rows[i].items())
            - rational[i]
            for i in range(len(phases))
        ]
        capacity = min(item["capacity"] for item in classes)
        band = 1e-10 * capacity
        classification, verified, margin = "critical_or_numerically_unresolved", False, None
        if self.l < capacity - band and max(exact_drifts) < 0:
            classification, verified, margin = "positive_recurrence", True, -max(exact_drifts)
        elif self.l > capacity + band:
            positive = [min(exact_drifts[i] for i in item["indices"]) for item in classes]
            if max(positive) > 0:
                classification, verified, margin = "positive_drift_escape", True, max(positive)
        self._phases, self._matrix = phases, matrix
        self._durations, self._departures = durations, departures
        self._exact_rows, self._exact_durations = exact_rows, exact_durations
        return PhaseStabilityResults(
            classification=classification,
            stable=classification == "positive_recurrence",
            drift_verified=verified,
            phase_count=len(phases),
            reachable_active_count=active_count,
            transient_phase_count=transient_count,
            closed_classes=classes,
            capacity=capacity,
            saturated_load=self.l / capacity,
            drift_margin=float(margin) if margin is not None else None,
            drift_margin_exact=str(margin) if margin is not None else None,
            corrector=corrector.tolist(),
            row_drifts=[float(value) for value in exact_drifts],
            row_drifts_exact=[str(value) for value in exact_drifts],
            poisson_residual=float(np.max(np.abs(raw + matrix @ corrector - corrector - delta))),
            transition_words=words,
            duration=time.perf_counter() - started,
        )
