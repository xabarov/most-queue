"""Three-parameter continuous batching of Ramani and Tantawi (2026).

Source: arXiv:2609.20957v1, Sections IV--V and Appendices A--B.
The birth--death chain is solved exactly, INCLUDING its geometric tail.
Homogeneous lengths, exponential completions, effective prefill stages and
mean-occupancy TTFT/ITL remain approximations to token-level execution.
"""

# Three independently calibrated costs and two workload lengths form the public contract.
# pylint: disable=too-many-arguments,too-many-positional-arguments

import copy
import math

import numpy as np
from scipy.special import logsumexp

from most_queue.structs import PrefillDecodeResults
from most_queue.theory.base_queue import BaseQueue


class PrefillDecodeCalc(BaseQueue):
    """State-dependent Markovian surrogate for mixed prefill/decode iterations.

    All costs use one common time unit; the arrival rate uses its inverse.
    None token_budget means unlimited tokens, NOT unlimited active requests.
    max_output_states limits only the displayed PMF, never the calculation.
    """

    def __init__(self, batch_cap: int = 256, token_budget: int | None = 8192, max_output_states: int = 4096):
        if isinstance(batch_cap, bool) or not isinstance(batch_cap, int) or not 1 <= batch_cap <= 4096:
            raise ValueError("batch_cap must be an integer in 1..4096")
        if token_budget is not None and (
            isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget < batch_cap
        ):
            raise ValueError("token_budget must be None or an integer >= batch_cap")
        if (
            isinstance(max_output_states, bool)
            or not isinstance(max_output_states, int)
            or not batch_cap + 1 <= max_output_states <= 1_000_000
        ):
            raise ValueError("max_output_states must be in batch_cap+1..1000000")
        super().__init__(n=1)
        self.batch_cap = batch_cap
        self.token_budget = token_budget
        self.max_output_states = max_output_states
        self.l = None
        self.input_tokens = self.output_tokens = None
        self.alpha = self.beta = self.gamma = None
        self._result = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """Set the positive finite Poisson arrival rate."""
        if not np.isfinite(l) or l <= 0:
            raise ValueError("arrival rate must be positive and finite")
        self.l = float(l)
        self.is_sources_set = True
        self._result = None

    def set_servers(self, input_tokens, output_tokens, alpha, beta, gamma):  # pylint: disable=arguments-differ
        """Set mean lengths and baseline, compute and KV-access time costs."""
        values = np.asarray([input_tokens, output_tokens, alpha, beta, gamma], dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values[:3] <= 0) or np.any(values[3:] < 0):
            raise ValueError("lengths and alpha must be positive; beta/gamma nonnegative; all finite")
        self.input_tokens, self.output_tokens, self.alpha, self.beta, self.gamma = values.tolist()
        self.is_servers_set = True
        self._result = None
        # Detect overflow in the largest profile before performing a stationary solve.
        try:
            self.stage_profile(self.batch_cap)
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            self.is_servers_set = False
            raise ValueError("invalid or overflowing stage profile") from exc

    def prefill_chunks(self, occupancy: float) -> int:
        """Appendix A's smallest feasible effective chunk count.

        Real occupancies in [0,B] extend the formula for the paper's final
        mean-occupancy evaluation; only integer occupancies 1..B define rates.
        A rationalized quadratic root avoids cancellation for large budgets.
        """
        if not self.is_servers_set:
            raise ValueError("set_servers must be called first")
        if not np.isfinite(occupancy) or not 0 <= occupancy <= self.batch_cap:
            raise ValueError("occupancy must be finite and in [0, batch_cap]")
        if self.token_budget is None:
            return 1
        n, m, budget = self.input_tokens, self.output_tokens, self.token_budget
        # Divide the polynomial by M before solving to reduce overflow risk.
        phi = m * (1 - (occupancy - 1) / budget) - occupancy * (n / budget)
        product = (n / budget) * m
        discriminant = math.hypot(phi, 2 * math.sqrt(product))
        root = 2 * product / (discriminant + phi) if phi >= 0 else (discriminant - phi) / 2
        if not math.isfinite(root):
            raise ValueError("effective chunk calculation overflowed; rescale lengths")
        chunks = max(1, math.ceil(root))

        def load(count):
            return (occupancy - 1) * (n + m) / (count + m) + n / count

        # Correct only floating-point rounding at an exact integer boundary.
        if chunks > 1 and load(chunks - 1) <= budget:
            chunks -= 1
        if load(chunks) > budget:
            chunks += 1
        return chunks

    def stage_profile(self, occupancy: float) -> dict[str, float]:
        """Return the paper's per-state work, iteration, and tagged latencies."""
        c = self.prefill_chunks(occupancy)
        n, m = self.input_tokens, self.output_tokens
        pref = self.beta * n + self.gamma * n * (c + 1) / 2
        dec = self.beta * m + self.gamma * m * (n + (m + 1) / 2)
        delta = (pref + dec) / (c + m)
        iteration = self.alpha + occupancy * delta
        background = self.alpha + (occupancy - 1) * delta
        values = {
            "chunks": c,
            "prefill_work": pref,
            "decode_work": dec,
            "delta": delta,
            "iteration": iteration,
            "prefill": c * background + pref,
            "itl": background + dec / m,
            "service": (c + m) * iteration,
        }
        if not all(math.isfinite(value) for value in values.values()) or values["service"] <= 0:
            raise ValueError("work or service calculation overflowed; rescale costs/lengths")
        return values

    def run(self, num_of_moments: int = 1) -> PrefillDecodeResults:
        """Compute exact surrogate occupancy/means and literal paper TTFT/ITL.

        Only the first W/V moment is exposed; no analytic TTFT/ITL tail.
        A negative mean-field latency is flagged, not silently clamped.
        """
        if num_of_moments != 1:
            raise ValueError("only first latency moments are supported")
        self._check_if_servers_and_sources_set()
        if self._result is not None:
            return copy.deepcopy(self._result)
        start = self._measure_time()
        cap = self.batch_cap
        rates = np.array([i / self.stage_profile(i)["service"] for i in range(1, cap + 1)])
        rho = self.l / rates[-1]
        if not np.all(np.isfinite(rates)) or np.any(rates <= 0) or not 0 < rho < 1:
            raise ValueError("stationary surrogate requires lambda * tau(B) / B < 1")
        log_weights = np.r_[0.0, np.cumsum(math.log(self.l) - np.log(rates))]
        log_norm = logsumexp(np.r_[log_weights[:-1], log_weights[-1] - math.log1p(-rho)])
        head = np.exp(log_weights - log_norm)  # individual probabilities 0..B
        at_cap = math.exp(log_weights[-1] - log_norm - math.log1p(-rho))  # P(N>=B)
        batch = float(head[1:cap] @ np.arange(1, cap)) + cap * at_cap
        queue = at_cap * rho / (1 - rho)
        waiting, service = queue / self.l, batch / self.l
        profile = self.stage_profile(min(float(cap), batch))
        pref = profile["prefill"]
        itl = (service - pref) / self.output_tokens
        ttft = waiting + pref + itl
        warnings = []
        if batch < 1:
            warnings.append("mean occupancy < 1: tagged-background formula is extrapolated below its physical domain")
        valid = min(pref, itl, ttft) >= 0
        if not valid:
            warnings.append("negative mean-field latency: literal paper estimate is not a physical prediction")
        # Prefix only. The geometric tail is included in normalization and ALL means.
        extra = self.max_output_states - cap - 1
        prefix = np.r_[head, head[-1] * rho ** np.arange(1, extra + 1)]
        omitted = at_cap * rho ** (extra + 1)
        residual = float(np.max(np.abs(head[1:] * rates - self.l * head[:-1]))) / self.l
        result = PrefillDecodeResults(
            w=[waiting],
            v=[waiting + service],
            p=prefix.tolist(),
            utilization=rho,
            ttft_mean=ttft,
            itl_mean=itl,
            prefill_mean=pref,
            service_mean=service,
            batch_mean=batch,
            jobs_mean=batch + queue,
            queue_mean=queue,
            waiting_probability=at_cap,
            queue_nonempty_probability=at_cap * rho,
            busy_probability=1 - head[0],
            capacity=float(rates[-1]),
            effective_prefill_chunks=profile["chunks"],
            tail_ratio=rho,
            omitted_probability=omitted,
            stationary_residual=residual,
            metrics_valid=valid,
            diagnostics=warnings,
        )
        if not all(math.isfinite(x) for x in (waiting, service, batch, queue, itl, ttft)):
            raise ValueError("stationary metrics overflowed; rescale time/rate")
        self.ro, self.w, self.v, self.p = rho, result.w.copy(), result.v.copy(), result.p.copy()
        self.mean_jobs_on_queue, self.mean_jobs_in_system = queue, batch + queue
        self._set_duration(result, start)
        self._result = copy.deepcopy(result)
        return result

    def get_p(self) -> list[float]:
        """Return an unrenormalized PMF prefix (run().omitted_probability gives its remainder)."""
        return self.run().p

    def get_w(self, num: int = 1) -> list[float]:
        """Return the surrogate mean waiting time only."""
        return self.run(num).w

    def get_v(self, num: int = 1) -> list[float]:
        """Return the surrogate mean sojourn time only."""
        return self.run(num).v
