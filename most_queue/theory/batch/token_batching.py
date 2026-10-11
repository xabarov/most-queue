"""Variable-token batching: Yang, Jiao, Xu, arXiv:2407.05347v1 (2024).

The random batch model is retained rather than replaced by its conditional
mean. Fixed batches use a departure-epoch queue chain and renewal rewards.
The paper's deterministic-mean surrogate is available separately, as is its
equation (25), whose b=1 limit is sojourn, not waiting time.
"""

import time

import numpy as np
from scipy.special import gammaln
from scipy.stats import poisson

from most_queue.structs import FixedTokenBatchingResults, QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.batch.callback_batching import CallbackBatchingCalc
from most_queue.theory.batch.dynamic_batching import _positive_integer
from most_queue.theory.batch.token_service import TokenBatchService, TokenLengthDistribution


class TokenBatchingCalc(CallbackBatchingCalc):
    """All-waiting dynamic or gated elastic batching with iid output lengths."""

    service = None

    def set_servers(self, tokens, k1, k2, k3, k4, *, elastic=False, max_fft_points=262_144):
        # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments
        """Configure equation (18), or random-sum equation (26) with early release."""
        service = TokenBatchService(tokens, k1, k2, k3, k4, elastic, max_fft_points)
        mean, second = service.envelope
        super().set_servers(
            service.moments, service.counts, mean_bound=mean, second_bound=second, job_mean=service.job_mean
        )
        self.service = service
        return self

    def paper_bound_report(self):
        """Report equation (16) applied to a pathwise duration envelope.

        Both bounds hold for a PATHWISE affine envelope H<=a*b+c, not
        merely for an affine bound on E[H|b]. See the validation report
        for the second-moment derivation of phi1 for random durations.
        """
        self._check_domain()
        lam, a, c = self.rate, self.alpha, self.tau0
        denominator = 2 * (1 - (lam * a) ** 2)
        phi0 = lam * (a + c) ** 2 / denominator
        phi1 = (lam * a * c + lam * a * a + c) / denominator
        return {
            "waiting_mean": self.get_waiting_mean(),
            "envelope_phi0": phi0,
            "paper_phi1_substitution": phi1,
            "paper_min_substitution": min(phi0, phi1),
            "phi1_certified": True,
        }


class FixedTokenBatchingCalc(BaseQueue):
    """Wait until b jobs are present; serve exactly b, retaining all excess.

    Numeric truncation applies to the post-departure backlog, not batch size.
    For b=1 this is ordinary M/G/1; for b>1 the batch arrival process is
    Erlang, so treating batches as a Poisson stream would be incorrect.
    """

    def __init__(self, batch_size, queue_truncation=None, tolerance=1e-8, max_states=4096, max_entries=2_000_000):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        super().__init__(n=1)
        self.batch_size = _positive_integer(batch_size, "batch_size")
        self.max_states = _positive_integer(max_states, "max_states")
        self.max_entries = _positive_integer(max_entries, "max_entries")
        self.queue_truncation = (
            None if queue_truncation is None else _positive_integer(queue_truncation, "queue_truncation")
        )
        if self.queue_truncation is not None and self.queue_truncation + 1 > self.max_states:
            raise ValueError("queue_truncation+1 must not exceed max_states")
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("tolerance must lie in (0,1)")
        self.tolerance = float(tolerance)
        self.rate = self.service = None
        self._cache = None
        self.mean_service_surrogate = False

    def set_sources(self, rate):
        """Set finite positive Poisson request rate."""
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be finite and positive")
        self.rate, self.is_sources_set, self._cache = float(rate), True, None
        return self

    def set_servers(self, tokens, k1, k2, k3, k4, *, elastic=False, mean_service_surrogate=False):
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        """Set random token service; surrogate=True replaces H by E[H|b].

        The surrogate is only meaningful for padded (not early-release)
        service, and is precisely the M/D^b/1 approximation in section IV-C.
        """
        if elastic and mean_service_surrogate:
            raise ValueError("the deterministic-mean surrogate is padded, not elastic")
        service = TokenBatchService(tokens, k1, k2, k3, k4, elastic)
        self.service, self.is_servers_set, self._cache = service, True, None
        self.mean_service_surrogate = bool(mean_service_surrogate)
        return self

    def _solve(self, limit):
        # pylint: disable=too-many-locals
        if (limit + 1) ** 2 > self.max_entries:
            raise MemoryError("fixed-batch transition matrix exceeds max_entries")
        b, lam = self.batch_size, self.rate
        mean, second = self.service.moments(b)
        if not np.isfinite(mean) or not np.isfinite(second):
            raise ValueError("fixed-batch service moments must be finite")
        if self.mean_service_surrogate:
            second = mean**2
            dist = poisson(lam * mean)
        else:
            dist = self.service.counts(b, lam)[0][1]
        q = np.arange(limit + 1, dtype=float)
        remains = np.maximum(q - b, 0)
        matrix = np.zeros((limit + 1, limit + 1))
        for index, remaining in enumerate(remains.astype(int)):
            counts = np.arange(limit - remaining)
            matrix[index, remaining:limit] = dist.pmf(counts)
            matrix[index, limit] = dist.sf(limit - remaining - 1)
        system = matrix.T - np.eye(limit + 1)
        system[0] = 1
        rhs = np.zeros(limit + 1)
        rhs[0] = 1
        pi = np.linalg.solve(system, rhs)
        if not np.all(np.isfinite(pi)) or pi.min() < -1e-10:
            raise RuntimeError("unusable fixed-batch stationary distribution")
        pi = np.maximum(pi, 0)
        pi /= pi.sum()
        for _ in range(10_000):
            updated = pi @ matrix
            updated /= updated.sum()
            change = float(np.abs(updated - pi) @ (1 + q**2)) / float(updated @ (1 + q**2))
            pi = updated
            if change < min(self.tolerance * 0.01, 1e-14):
                break
        else:
            raise RuntimeError("fixed-batch stationary refinement did not converge")
        filling = np.maximum(b - q, 0) / lam
        fill_area = np.where(q < b, (b * (b - 1) - q * (q - 1)) / (2 * lam), 0)
        cycle = mean + float(pi @ filling)
        area = float(pi @ (fill_area + remains * mean)) + lam * second / 2
        wait = area / (lam * cycle)
        job_service = mean if self.mean_service_surrogate else self.service.job_mean(b)
        flow = abs(lam * cycle - b) / b
        expected_second = float(pi @ (remains**2 + 2 * remains * lam * mean)) + lam * mean + lam**2 * second
        moment = abs(float(pi @ q**2) - expected_second) / max(expected_second, np.finfo(float).tiny)
        boundary = float(pi[q >= 0.8 * limit] @ q[q >= 0.8 * limit] ** 2) / max(float(pi @ q**2), 1e-300)
        residual = float(np.linalg.norm(pi @ matrix - pi, ord=1))
        return pi, np.array([wait, wait + job_service]), float(pi @ filling) / cycle, (flow, moment, boundary, residual)

    def run(self):
        """Return first waiting/sojourn moments and explicit truncation diagnostics."""
        if self.rate is None or self.service is None:
            raise ValueError("set_sources/set_servers must be called first")
        if self.rate * self.service.moments(self.batch_size)[0] >= self.batch_size:
            raise ValueError("fixed batching requires lambda*E[H|b]<b")
        if self._cache is not None:
            return self._copy_result(self._cache)
        start = time.process_time()
        limit = self.queue_truncation or min(63, self.max_states - 1)
        if limit < 1:
            raise ValueError("max_states must permit at least two queue states")
        previous = None
        while True:
            pi, metrics, idle, diagnostics = self._solve(limit)
            refinement = (
                None if previous is None else float(np.max(np.abs(metrics - previous) / np.maximum(metrics, 1e-300)))
            )
            converged = refinement is not None and max(refinement, *diagnostics) <= self.tolerance
            if self.queue_truncation is not None or converged:
                break
            if limit + 1 == self.max_states:
                raise ValueError("fixed-batch truncation did not converge within max_states")
            previous, limit = metrics, min(2 * (limit + 1), self.max_states) - 1
        flow, moment, boundary, residual = diagnostics
        self._cache = FixedTokenBatchingResults(
            w=[float(metrics[0])],
            v=[float(metrics[1])],
            utilization=1 - idle,
            batch_size=self.batch_size,
            queue_distribution=pi,
            queue_truncation=limit,
            idle_probability=idle,
            flow_residual=flow,
            moment_residual=moment,
            boundary_second_moment_fraction=boundary,
            stationary_residual=residual,
            refinement_error=refinement,
            converged=converged,
            mean_service_surrogate=self.mean_service_surrogate,
            duration=time.process_time() - start,
        )
        return self._copy_result(self._cache)

    @staticmethod
    def _copy_result(result):
        return FixedTokenBatchingResults(
            **{
                **vars(result),
                "w": result.w.copy(),
                "v": result.v.copy(),
                "queue_distribution": result.queue_distribution.copy(),
            }
        )

    def get_w(self, num=1):
        """Only the first waiting moment is implemented."""
        if _positive_integer(num, "num") != 1:
            raise ValueError("only the first waiting moment is implemented")
        return self.run().w

    def get_v(self, num=1):
        """Only the first sojourn moment is implemented."""
        if _positive_integer(num, "num") != 1:
            raise ValueError("only the first sojourn moment is implemented")
        return self.run().v


class TokenClippingCalc(BaseQueue):
    """Single-request M/G/1 with S=a*total_output_tokens+c, equations (1)--(5)."""

    def __init__(self):
        super().__init__(n=1)
        self.rate = self.tokens = None
        self.a = self.c = 0.0

    def set_sources(self, rate):
        """Set Poisson arrival rate."""
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("rate must be finite and positive")
        self.rate, self.is_sources_set = float(rate), True
        return self

    def set_servers(self, tokens, a, c=0.0):
        """Set linear single-request cost and a clipped or original token PMF."""
        if not isinstance(tokens, TokenLengthDistribution):
            raise ValueError("tokens must be a TokenLengthDistribution")
        if not np.isfinite(a) or not np.isfinite(c) or a < 0 or c < 0 or a + c <= 0:
            raise ValueError("a/c must be finite, non-negative, not both zero")
        self.tokens, self.a, self.c, self.is_servers_set = tokens, float(a), float(c), True
        return self

    def run(self):
        """Pollaczek--Khinchine waiting and sojourn means, not delay distributions."""
        if self.rate is None or self.tokens is None:
            raise ValueError("set_sources/set_servers must be called first")
        mean, second = self.tokens.moments()
        service = self.a * mean + self.c
        service_second = self.a**2 * second + 2 * self.a * self.c * mean + self.c**2
        if not np.isfinite(service) or not np.isfinite(service_second):
            raise ValueError("service moments overflow floating-point representation")
        rho = self.rate * service
        if rho >= 1:
            raise ValueError("M/G/1 requires lambda*E[S]<1")
        wait = self.rate * service_second / (2 * (1 - rho))
        return QueueResults(w=[wait], v=[wait + service], utilization=rho)


def optimize_token_limit(tokens, candidates, rate, a, c=0.0, theta=0.99):
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Exhaustive finite candidate search for paper equation (10), patient users.

    Utility is E[min(N,limit)/N] from an uncensored finite original PMF.
    Delay units affect the utility tradeoff; theta is not dimension-free.
    Unstable candidates are excluded and reported, not assigned finite delays.
    """
    if not isinstance(tokens, TokenLengthDistribution) or not np.isfinite(tokens.utility) or tokens.clipped_fraction:
        raise ValueError("optimization requires an original, uncensored finite token PMF")
    if not np.isfinite(theta) or not 0 <= theta <= 1:
        raise ValueError("theta must lie in [0,1]")
    rows = []
    calc = TokenClippingCalc().set_sources(rate)
    calc.set_servers(tokens, a, c)  # validate parameters before evaluating candidates
    for cap in candidates:
        law = TokenLengthDistribution(tokens.lengths, tokens.probabilities, cap)
        calc.set_servers(law, a, c)
        if rate * (a * law.moments()[0] + c) >= 1:
            rows.append({"max_tokens": int(cap), "stable": False})
            continue
        wait = calc.run().w[0]
        rows.append(
            {
                "max_tokens": int(cap),
                "stable": True,
                "waiting_mean": wait,
                "utility": law.utility,
                "clipped_fraction": law.clipped_fraction,
                "objective": theta * law.utility - (1 - theta) * wait,
            }
        )
    feasible = [row for row in rows if row["stable"]]
    if not feasible:
        raise ValueError("no stable candidate token limit")
    return {"best": max(feasible, key=lambda row: row["objective"]).copy(), "candidates": rows}


def paper_fixed_formula(rate, duration, batch_size, terms=20):
    """Literal equation (25): sojourn MINUS batch formation (b-1)/(2*rate).

    Its b=1 limit is mean SOJOURN, not waiting. Finite terms are a series
    approximation, not an exact root calculation.
    Use FixedTokenBatchingCalc(..., mean_service_surrogate=True) as a reference.
    """
    b, terms = _positive_integer(batch_size, "batch_size"), _positive_integer(terms, "terms")
    if not np.isfinite(rate) or rate <= 0 or not np.isfinite(duration) or duration <= 0 or rate * duration >= b:
        raise ValueError("require finite positive rate/duration and rate*duration<b")
    x = rate * duration
    m = np.arange(1, terms + 1)
    coefficients = np.exp(-x * m / b + (m - 1) * np.log(x * m / b) - gammaln(m + 1))
    roots = np.exp(2j * np.pi * np.arange(1, b) / b)
    z = np.sum(coefficients[:, None] * roots[None, :] ** m[:, None], axis=0)
    value = ((b - (b - x) ** 2) / (2 * (b - x)) + np.sum(1 / (1 - z))) / rate
    if abs(value.imag) > 1e-9 * max(1, abs(value.real)):
        raise RuntimeError("fixed-batch series failed conjugate symmetry")
    return float(value.real)
