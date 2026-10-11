"""Ao, Dong, Luo, Simchi-Levi, arXiv:2606.15555v1, Sections 2--5.

Continuous request masses; saturated proportional admissions and LPF eviction.
Spectrum describes the no-eviction recurrence, not stochastic queue recurrence.
Homogeneous throughput formulae are the authors'; root computations are numerical.
"""

import math
from numbers import Integral, Real

import numpy as np

from most_queue.structs import KvCacheCongestionResults, KvCacheFluidResults
from most_queue.theory.base_queue import BaseQueue


def _integer(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < int(not zero):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return int(value)


def _real(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _spectrum(coefficients):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = np.asarray(coefficients, dtype=float) / coefficients[0]
    if not np.all(np.isfinite(normalized)) or np.any(normalized <= 0):
        raise ValueError("nonfinite or underflowed characteristic coefficients; rescale")
    roots = np.roots(normalized)
    if not np.all(np.isfinite(roots)):
        raise ValueError("nonfinite characteristic roots; reduce degree or rescale")
    residual = 0.0
    for root in roots:
        numerator = abs(np.polyval(normalized, root))
        denominator = np.polyval(abs(normalized), abs(root))
        if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("unreliable polynomial residual; reduce degree or rescale")
        residual = max(residual, float(numerator / denominator))
    return float(max(abs(roots), default=0.0)), residual


class KvCacheCongestionCalc(BaseQueue):
    """Equilibria and numerical finite-input spectrum of saturated KV dynamics.

    set_sources specifies positive proportions, NOT a Poisson arrival rate.
    set_servers specifies deterministic class lengths. A root near the unit
    circle is explicitly inconclusive. Degree is bounded to protect resources.
    """

    def __init__(self, memory_tokens, *, max_decode_length=256, spectral_tolerance=1e-8):
        super().__init__(n=1)
        self.memory_tokens = _real(memory_tokens, "memory_tokens")
        self.max_decode_length = _integer(max_decode_length, "max_decode_length")
        self.spectral_tolerance = _real(spectral_tolerance, "spectral_tolerance")
        if self.spectral_tolerance >= 1:
            raise ValueError("spectral_tolerance must be smaller than one")
        self.proportions = None
        self.input_tokens = None
        self.decode_tokens = None
        self.input_ratios = None

    def set_sources(self, proportions):
        """Set strictly positive class proportions summing to one."""
        values = [_real(value, "proportion") for value in proportions]
        if not values or not math.isclose(math.fsum(values), 1, rel_tol=0, abs_tol=1e-10):
            raise ValueError("positive proportions must sum to one")
        if self.decode_tokens is not None and len(values) != len(self.decode_tokens):
            raise ValueError("proportions and classes must have equal length")
        self.proportions = [value / math.fsum(values) for value in values]
        self.is_sources_set = True

    def set_servers(self, input_tokens, decode_tokens, *, input_ratios=None):
        """Set class lengths and optional positive asymptotic input ratios.

        Ratios default to one (common-input scaling), independently of finite
        inputs. Supply explicit ratios for heterogeneous-input limiting analysis.
        All supported requests must fit strictly at their largest stage.
        """
        inputs = [_integer(value, "input_tokens", zero=True) for value in input_tokens]
        decodes = [_integer(value, "decode_tokens") for value in decode_tokens]
        if not inputs or len(inputs) != len(decodes):
            raise ValueError("input/decode vectors must have equal nonzero length")
        if self.proportions is not None and len(inputs) != len(self.proportions):
            raise ValueError("proportions and classes must have equal length")
        if max(decodes) > self.max_decode_length:
            raise ValueError("decode length exceeds configured max_decode_length")
        if any(s + d >= self.memory_tokens for s, d in zip(inputs, decodes)):
            raise ValueError("memory_tokens must exceed each request's peak footprint")
        ratios = [1.0] * len(inputs) if input_ratios is None else [_real(x, "input_ratio") for x in input_ratios]
        if len(ratios) != len(inputs):
            raise ValueError("input_ratios and classes must have equal length")
        self.input_tokens, self.decode_tokens, self.input_ratios = inputs, decodes, ratios
        self.is_servers_set = True

    def run(self):
        """Return exact algebraic equilibrium and numerical recurrence spectrum."""
        self._check_if_servers_and_sources_set()
        start = self._measure_time()
        try:
            costs = [d * (s + (d + 1) / 2) for s, d in zip(self.input_tokens, self.decode_tokens)]
            mean_cost = math.fsum(p * cost for p, cost in zip(self.proportions, costs))
            nominal = self.memory_tokens / mean_cost
            degree = max(self.decode_tokens)
            finite, limiting = [], []
            for age in range(degree):
                alive = [k for k, d in enumerate(self.decode_tokens) if d > age]
                finite.append(math.fsum(self.proportions[k] * (self.input_tokens[k] + 1 + age) for k in alive))
                limiting.append(math.fsum(self.proportions[k] * self.input_ratios[k] for k in alive))
            radius, residual = _spectrum(finite)
            limiting_radius, limiting_residual = _spectrum(limiting)
        except (OverflowError, ZeroDivisionError) as exc:
            raise ValueError("length/weight arithmetic overflow; rescale") from exc
        if not math.isfinite(mean_cost) or nominal <= 0 or not math.isfinite(nominal):
            raise ValueError("nonfinite or underflowed equilibrium; rescale")
        status = "near_boundary"
        if radius < 1 - self.spectral_tolerance:
            status = "schur_stable_numerically"
        elif radius > 1 + self.spectral_tolerance:
            status = "linearly_unstable_numerically"
        homogeneous = len(set(zip(self.input_tokens, self.decode_tokens))) == 1
        s, d = self.input_tokens[0], self.decode_tokens[0]
        worst = self.memory_tokens / (d * (s + d)) if homogeneous else None
        result = KvCacheCongestionResults(
            nominal_throughput=nominal,
            worst_cycle_throughput=worst,
            worst_cycle_ratio=worst / nominal if homogeneous else None,
            fixed_point=[[p * nominal] * d for p, d in zip(self.proportions, self.decode_tokens)],
            coefficients=finite,
            limiting_coefficients=limiting,
            spectral_radius=radius,
            limiting_spectral_radius=limiting_radius,
            root_residual=max(residual, limiting_residual),
            decode_gcd=math.gcd(*self.decode_tokens),
            linear_stability=status,
            diagnostics=[
                "Completions per iteration; deterministic saturated continuous masses.",
                "Proportional admissions and proportional within-stage LPF eviction.",
                "Finite root test is numerical, not a certified or open-queue stability theorem.",
                "GCD is an input-dominated diagnostic; finite input length still matters.",
                "Limiting input ratios are explicit scaling assumptions, not estimated from finite lengths.",
                "No prefill/swap recovery time, GPU fit, latency distribution, or stationary PMF.",
            ],
        )
        if result.root_residual > 1e-8:
            result.linear_stability = "near_boundary"
            result.diagnostics.append("Large polynomial backward residual: classification suppressed.")
        self._set_duration(result, start)
        return result

    def evolve(  # pylint: disable=too-many-arguments
        self, iterations, *, initial_state=None, admission_cap=None, warmup=0, record_trace=False
    ):
        """Execute the saturated continuous map; no integer rounding or arrivals.

        admission_cap=None is greedy; otherwise it is aggregate request mass
        per iteration. The cap need not be an integer. Initial states may have
        slack, but must be nonnegative, correctly shaped and memory-feasible.
        """
        self._check_if_servers_and_sources_set()
        iterations = _integer(iterations, "iterations")
        warmup = _integer(warmup, "warmup", zero=True)
        if warmup >= iterations or not isinstance(record_trace, bool):
            raise ValueError("warmup must be below iterations and record_trace must be bool")
        cap = math.inf if admission_cap is None else _real(admission_cap, "admission_cap", zero=True)
        weights = [np.arange(s + 1, s + 1 + d, dtype=float) for s, d in zip(self.input_tokens, self.decode_tokens)]
        state = [np.zeros(d) for d in self.decode_tokens]
        if initial_state is not None:
            if len(initial_state) != len(state):
                raise ValueError("initial_state must have one row per class")
            for k, row in enumerate(initial_state):
                if len(row) != self.decode_tokens[k]:
                    raise ValueError("initial_state row must match decode length")
                state[k] = np.array([_real(x, "mass", zero=True) for x in row])

        def memory(values):
            return math.fsum(float(np.dot(w, x)) for w, x in zip(weights, values))

        if memory(state) > self.memory_tokens * (1 + 1e-12):
            raise ValueError("initial_state exceeds memory capacity")
        start = self._measure_time()
        result = KvCacheFluidResults(initial_work=math.fsum(float(np.dot(np.arange(len(x)), x)) for x in state))
        stage0_cost = math.fsum(p * (s + 1) for p, s in zip(self.proportions, self.input_tokens))
        observed_completions = 0.0
        observed_memory = 0.0
        for step in range(iterations):
            completions = math.fsum(float(x[-1]) for x in state)
            result.executed_work += math.fsum(float(sum(x)) for x in state)
            result.completed_work += math.fsum(float(x[-1]) * len(x) for x in state)
            shifted = [np.concatenate(([0.0], x[:-1])) for x in state]
            post_execution_memory = memory(shifted)
            evicted = 0.0
            for age in range(1, max(self.decode_tokens)):
                excess = memory(shifted) - self.memory_tokens
                if excess <= 0:
                    break
                members = [k for k, d in enumerate(self.decode_tokens) if d > age]
                stage_memory = math.fsum(float(weights[k][age] * shifted[k][age]) for k in members)
                if stage_memory == 0:
                    continue
                fraction = min(1.0, excess / stage_memory)
                for k in members:
                    removed = float(shifted[k][age] * fraction)
                    evicted += removed
                    result.discarded_work += age * removed
                    shifted[k][age] *= 1 - fraction
            free = max(0.0, self.memory_tokens - memory(shifted))
            admission = min(cap, free / stage0_cost)
            for k, p in enumerate(self.proportions):
                shifted[k][0] = p * admission
            state = shifted
            used = memory(state)
            result.evicted_mass += evicted
            if step >= warmup:
                observed_completions += completions
                observed_memory += used
            if record_trace:
                result.iteration_trace.append(
                    {
                        "iteration": step,
                        "state": [x.tolist() for x in state],
                        "completed": completions,
                        "evicted": evicted,
                        "admitted": admission,
                        "memory": used,
                        "post_execution_memory": post_execution_memory,
                    }
                )
        result.final_state = [x.tolist() for x in state]
        result.remaining_work = math.fsum(float(np.dot(np.arange(len(x)), x)) for x in state)
        result.throughput = observed_completions / (iterations - warmup)
        result.memory_fraction = observed_memory / ((iterations - warmup) * self.memory_tokens)
        self._set_duration(result, start)
        return result
