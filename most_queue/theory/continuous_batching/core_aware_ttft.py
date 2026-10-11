"""Phase-specific overshoot and finite-domain full-return continuation.

The finite solve retains exterior excursions, then closes returns to h<B
with a scalar maximum. This is an application of classical regeneration and
Lyapunov comparison, not a new general truncation theorem. See Glynn--Zheng
(2025), https://arxiv.org/html/2505.03157v1, Section 2 equations 13--21.
Model-specific derivation: docs/research/core-aware-continuation-validation-2026.md.
Only phase drift uses verified rational arithmetic; mean intervals are float.
"""

# Physical configuration and combined complexity budgets are explicit inputs.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import itertools
import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from most_queue.theory.continuous_batching.infinite_stage_aware_ttft import InfiniteStageAwareTtftCalc
from most_queue.theory.continuous_batching.phase_aware_ttft import PhaseAwareTtftCalc, PhaseAwareTtftResults


@dataclass
class CoreAwareTtftResults(PhaseAwareTtftResults):
    """Floating mean intervals plus the cost and checks of the auxiliary solve."""

    continuation_diagnostics: dict = field(default_factory=dict)


class CoreAwareTtftCalc(PhaseAwareTtftCalc):
    """Tighter full-cycle continuation for the same infinite iid token queue.

    mode='phase' isolates exit-phase shifts with the old clearing bound.
    mode='core' additionally solves the killed auxiliary domain h<=K.
    K is core_truncation (default 8), not a physical buffer. Count-domain
    state/word budgets include BOTH finite solves; phase certification retains
    its separate legacy budget. No reflection or lost exterior rewards.
    """

    def __init__(
        self,
        batch_cap=2,
        token_budget=4,
        truncation=8,
        max_states=20000,
        max_transition_words=2000000,
        core_truncation=8,
        mode="core",
    ):
        super().__init__(batch_cap, token_budget, truncation, max_states, max_transition_words)
        if mode not in ("phase", "core"):
            raise ValueError("mode must be 'phase' or 'core'")
        if isinstance(core_truncation, bool) or not isinstance(core_truncation, int) or core_truncation < batch_cap - 1:
            raise ValueError("core_truncation must be an integer >= batch_cap-1")
        self.core_truncation, self.mode = core_truncation, mode
        self._diagnostics = {}
        self._shift_cache = {}
        self._phase_shifts = {}
        self._core_upper = np.zeros(4)
        self._building_core = False

    def _prepare_bound(self, probs, work):
        load = super()._prepare_bound(probs, work)
        b = np.asarray(self._certificate.corrector)
        self._phase_shifts = dict(zip(self._phase_calc._phases, b - b.min() + 1))
        values = self._constants
        self._core_upper = np.array([values["core_time_upper"]] + [values["core_area_upper"]] * 3)
        self._diagnostics["global_core_upper"] = self._core_upper.tolist()
        if self.mode == "core":
            self._solve_core(probs, work)
        self._diagnostics["core_upper"] = self._core_upper.tolist()
        return load

    def _exit_shifts(self, survivors, probs):
        """E[c(a')], E[c(a')^2] under full iid refill, independent of J."""
        if survivors not in self._shift_cache:
            fill = self.batch_cap - len(survivors)
            if len(probs) ** fill > self.max_transition_words:
                raise ValueError("exit phase word budget exceeded")
            first = second = 0.0
            for word in itertools.product(range(len(probs)), repeat=fill):
                mass = math.prod(probs[k] for k in word)
                active = self._canonical(survivors + tuple((k, 0) for k in word))
                shift = self._phase_shifts[active]
                first += mass * shift
                second += mass * shift**2
            self._shift_cache[survivors] = first, second
        return self._shift_cache[survivors]

    def _exit_remainder(self, survivors, moments, work, probs):
        del work
        probability, first, second = moments
        c1, c2 = self._exit_shifts(survivors, probs)
        values = self._constants
        eps = values["epsilon"]
        y, y2 = first + c1 * probability, second + 2 * c1 * first + c2 * probability
        steps = y / eps
        area_steps = (y2 + values["increment_second_max"] * y / eps) / (2 * eps)
        area = values["duration_max"] * area_steps
        population = area + values["population_step_constant"] * steps
        remainder = np.array(
            (
                values["duration_max"] * steps,
                population,
                area + values["waiting_step_constant"] * steps,
                population,
            )
        )
        if not self._building_core:
            remainder += probability * self._core_upper
        if not np.all(np.isfinite(remainder)) or np.min(remainder) < 0:
            raise ValueError("exit continuation overflow or invalid moments")
        return remainder

    @staticmethod
    def _return_system(matrix, rewards, exits, core_indices):
        """Solve all finite-domain rewards and idle/exit absorption probabilities."""
        equation = eye(matrix.shape[0] - 1, format="csc") - matrix[1:, 1:].tocsc()
        rhs = np.column_stack((rewards[1:] + exits[1:, 1:], exits[1:, 0], matrix[1:, 0].toarray().ravel()))
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                solved = np.asarray(spsolve(equation, rhs))
            except MatrixRankWarning as exc:
                raise ValueError("singular finite-core return solve") from exc
        if not np.all(np.isfinite(solved)) or np.min(solved) < -1e-9:
            raise ValueError("ill-conditioned finite-core return solve")
        residual = float(np.max(np.abs(equation @ solved - rhs) / (1 + np.abs(rhs))))
        mass = float(np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() + exits[:, 0] - 1)))
        absorption = float(np.max(np.abs(solved[:, 4] + solved[:, 5] - 1)))
        if residual > 1e-9 or mass > 1e-10 or absorption > 1e-9:
            raise ValueError("finite-core return balance failed")
        solved = np.maximum(solved, 0)
        success = solved[core_indices, 5]
        if len(success) == 0 or np.min(success) <= 0 or np.max(solved[:, 4]) >= 1:
            raise ValueError("finite-core return contraction unresolved")
        # In exact arithmetic success=1-exit. Row-wise ratios dominate the
        # classical max(cost)/min(success) geometric bound, often strictly.
        upper = np.max(solved[core_indices, :4] / success[:, None], axis=0)
        if not np.all(np.isfinite(upper)):
            raise ValueError("finite-core return bound overflow")
        return upper, {
            "solve_relative_residual": residual,
            "mass_residual": mass,
            "absorption_residual": absorption,
            "restart_idle_probability_min": float(success.min()),
            "restart_exit_probability_max": float(solved[core_indices, 4].max()),
            "separate_max_geometric_upper": (solved[core_indices, :4].max(axis=0) / success.min()).tolist(),
        }

    def _solve_core(self, probs, work):
        started = time.perf_counter()
        truncation = self.truncation
        self.truncation, self._building_core = self.core_truncation, True
        try:
            states, matrix, rewards, exits, _, words = InfiniteStageAwareTtftCalc._killed_kernel(self, probs, work)
        finally:
            self.truncation, self._building_core = truncation, False
        core_indices = [i - 1 for i, (_, h) in enumerate(states) if i and h < self.batch_cap]
        upper, checks = self._return_system(matrix, rewards, exits, core_indices)
        # Two independently valid uniform bounds on return from h<B.
        self._core_upper = np.minimum(self._core_upper, upper)
        self._diagnostics.update(
            checks,
            core_states=len(states),
            core_transition_words=words,
            core_kernel_storage_bytes=matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
            core_solve_seconds=time.perf_counter() - started,
            uncapped_core_upper=upper.tolist(),
        )

    def _killed_kernel(self, probs, work):
        """Reserve the auxiliary work against the SAME count-system budgets."""
        budgets = self.max_states, self.max_transition_words
        self.max_states -= self._diagnostics.get("core_states", 0)
        self.max_transition_words -= self._diagnostics.get("core_transition_words", 0)
        try:
            if min(self.max_states, self.max_transition_words) <= 0:
                raise ValueError("combined continuation state/word budget exceeded")
            return super()._killed_kernel(probs, work)
        finally:
            self.max_states, self.max_transition_words = budgets

    def run(self):
        """Reset derived state; compute same queue with the chosen continuation."""
        if (
            isinstance(self.truncation, bool)
            or not isinstance(self.truncation, int)
            or self.truncation < self.batch_cap - 1
        ):
            raise ValueError("truncation must be an integer >= batch_cap-1")
        if self.mode not in ("phase", "core"):
            raise ValueError("mode must be 'phase' or 'core'")
        if (
            isinstance(self.core_truncation, bool)
            or not isinstance(self.core_truncation, int)
            or self.core_truncation < self.batch_cap - 1
        ):
            raise ValueError("core_truncation must be an integer >= batch_cap-1")
        self._diagnostics = {"mode": self.mode, "core_truncation": self.core_truncation}
        self._shift_cache, self._phase_shifts = {}, {}
        self._building_core = False
        result = super().run()
        self._diagnostics.update(
            total_count_states=result.states + self._diagnostics.get("core_states", 0),
            total_count_transition_words=result.transition_words + self._diagnostics.get("core_transition_words", 0),
            total_kernel_storage_bytes=result.kernel_storage_bytes
            + self._diagnostics.get("core_kernel_storage_bytes", 0),
        )
        return CoreAwareTtftResults(
            **{**vars(result), "model": "infinite iid queue; phase/core continuation, floating mean bounds"},
            continuation_diagnostics=self._diagnostics.copy(),
        )
