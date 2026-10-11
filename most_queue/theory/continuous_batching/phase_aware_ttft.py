"""Infinite mean intervals using phase-corrected exterior return potentials.

The exact rational drift certificate is PhaseAwareStabilityCalc. Delay bounds
use linear/quadratic potentials and a uniform no-arrival clearing trial in the
finite core. Regenerative ratio methodology: Glynn and Zheng (2025),
https://arxiv.org/html/2505.03157v1, Section 2 equations 3--12.
Only the phase drift is verified exactly; full mean-bound arithmetic is float.
"""

# Physical parameters and explicit domain/complexity budgets follow EPIC-095.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import math
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np

from most_queue.theory.continuous_batching.infinite_stage_aware_ttft import (
    InfiniteStageAwareTtftCalc,
    InfiniteStageAwareTtftResults,
)
from most_queue.theory.continuous_batching.phase_aware_stability import PhaseAwareStabilityCalc


@dataclass
class PhaseAwareTtftResults(InfiniteStageAwareTtftResults):
    """Floating mean bounds alongside a separately verified drift certificate."""

    phase_certificate: dict = field(default_factory=dict)
    return_bound_constants: dict = field(default_factory=dict)


class PhaseAwareTtftCalc(InfiniteStageAwareTtftCalc):
    """Replace the workload restriction by a checked saturated phase drift.

    All model/policy/support restrictions remain. Exterior potentials bound
    return to h<B; explicit clearing trials bound return from this finite core
    to idle. Thus continuation accounts for complete cycles and arbitrary
    overshoot. Bounds can be loose or exceed budgets near critical load.
    """

    def __init__(self, batch_cap=2, token_budget=4, truncation=8, max_states=20000, max_transition_words=2000000):
        super().__init__(batch_cap, token_budget, truncation, max_states, max_transition_words)
        self._phase_calc = self._certificate = None
        self._constants = {}

    def _mark_probabilities(self):
        """Float evaluation of the same exactly normalized binary input law."""
        masses = [Fraction(p) for _, _, p in self.law]
        total = sum(masses)
        return np.array([float(p / total) for p in masses])

    def _prepare_bound(self, probs, work):
        """Construct exterior time/area and finite-core clearing bounds."""
        try:
            return self._construct_bound(probs, work)
        except OverflowError as exc:
            raise ValueError("continuation potential overflow; stability certificate remains valid") from exc

    def _construct_bound(self, probs, work):
        """Float evaluation with explicit refusal if intermediate values overflow."""
        corrector = np.asarray(self._certificate.corrector)
        shifts = corrector - corrector.min() + 1
        epsilon = self._certificate.drift_margin
        durations = self._phase_calc._durations
        departures = self._phase_calc._departures
        second = []
        for i, row in enumerate(self._phase_calc._exact_rows):
            mean = self.l * durations[i]
            second.append(
                mean + sum(float(p) * (mean - departures[i] + shifts[j] - shifts[i]) ** 2 for j, p in row.items())
            )
        m2, shift = max(second), float(shifts.max())
        # Core can contain a partial pool; use maximum over ALL reachable active
        # phases, not only saturated durations. A tagged older prefill may stall.
        max_duration = self.alpha + self.batch_cap * (
            self.beta * max(n for n, _, _ in self.law) + self.gamma * max(n + m for n, m, _ in self.law)
        )
        steps = (2 * self.batch_cap - 1) * max(n + m for n, m, _ in self.law)
        trial_time = steps * max_duration
        mean_count = self.l * trial_time
        probability = math.exp(-mean_count)
        if probability == 0 or not math.isfinite(mean_count):
            raise ValueError("finite-core clearing bound overflow; stability certificate remains valid")
        y_mean = self.batch_cap - 1 + mean_count + shift
        y_second = y_mean**2 + mean_count
        g_mean = (y_second + (m2 / epsilon) * y_mean) / (2 * epsilon)
        c_wait = self.l * max_duration**2 / 2
        c_population = self.batch_cap * max_duration + c_wait
        core_time = (trial_time + max_duration * y_mean / epsilon) / probability
        core_area = (
            trial_time * (2 * self.batch_cap - 1 + mean_count) + max_duration * g_mean + c_population * y_mean / epsilon
        ) / probability
        self._constants = {
            "epsilon": epsilon,
            "shift_max": shift,
            "increment_second_max": m2,
            "duration_max": max_duration,
            "clearing_steps": steps,
            "clearing_probability_lower": probability,
            "core_time_upper": core_time,
            "core_area_upper": core_area,
            "waiting_step_constant": c_wait,
            "population_step_constant": c_population,
        }
        if not all(math.isfinite(value) for value in self._constants.values()):
            raise ValueError("continuation potential overflow; stability certificate remains valid")
        # This field remains a workload diagnostic, not the accepted condition.
        return self.l * float(probs @ work)

    def _remainder(self, moments, work, probs):
        """Integrate full-cycle return bounds over exact Poisson overshoot moments."""
        del work, probs
        probability, first, second = moments
        values = self._constants
        epsilon, shift = values["epsilon"], values["shift_max"]
        y = first + shift * probability
        y2 = second + 2 * shift * first + shift**2 * probability
        steps = y / epsilon
        area_steps = (y2 + values["increment_second_max"] * y / epsilon) / (2 * epsilon)
        duration = values["duration_max"] * steps + values["core_time_upper"] * probability
        waiting = (
            values["duration_max"] * area_steps
            + values["waiting_step_constant"] * steps
            + values["core_area_upper"] * probability
        )
        population = (
            values["duration_max"] * area_steps
            + values["population_step_constant"] * steps
            + values["core_area_upper"] * probability
        )
        return np.array((duration, population, waiting, population))

    def run(self):
        """Check phase drift and return infinite mean bounds or a clear refusal."""
        self._check_if_servers_and_sources_set()
        started = time.perf_counter()
        self._constants = {}
        phase = PhaseAwareStabilityCalc(self.batch_cap, self.token_budget, self.max_states, self.max_transition_words)
        phase.set_servers(self.law, self.alpha, self.beta, self.gamma)
        phase.set_sources(self.l)
        certificate = phase.run()
        if not certificate.stable:
            raise ValueError(f"phase stability certificate not available: {certificate.classification}")
        self._phase_calc, self._certificate = phase, certificate
        result = super().run()
        return PhaseAwareTtftResults(
            **{
                **vars(result),
                "duration": time.perf_counter() - started,
                "model": "infinite iid queue; rational phase drift and floating regenerative mean bounds",
            },
            phase_certificate=asdict(certificate),
            return_bound_constants=self._constants.copy(),
        )
