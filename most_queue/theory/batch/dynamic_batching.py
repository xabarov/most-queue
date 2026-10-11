"""Numerical analysis of Inoue's unbounded dynamic-batching queue.

Source: Y. Inoue, Performance Evaluation 147 (2021), 102183,
doi:10.1016/j.peva.2020.102183, arXiv:1912.06322v3, equations (6), (15),
(35), (38), (41), (42). The model, identities and bounds belong to Inoue.
Section 3.1 already discusses truncation; this is an implementation, not a
new exact solution of an infinite chain.

Poisson arrivals wait until the current batch completes. All waiting jobs then
start together; an empty server starts the next arrival alone. Each batch has
mean processing time alpha*b + tau0. The embedded chain is B' = max(1, A),
where A counts arrivals during processing. Counts are Poisson (deterministic),
negative binomial (gamma, including Erlang/exponential), or a mixture of two
geometrics (H2). Numerical overflow is lumped into the last state; this is NOT
a physical batch cap (excess customers would remain queued under a cap).

Automatic truncation checks refinement, upper-tail second-moment weight and
the untruncated moment identities. These are diagnostics, NOT certified error
bounds. A fixed truncation returns its diagnostics without claiming convergence.
"""

import time
from numbers import Integral

import numpy as np
from scipy.sparse import csc_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.stats import nbinom, poisson

from most_queue.structs import DynamicBatchingResults
from most_queue.theory.base_queue import BaseQueue

SERVICE_KINDS = ("deterministic", "exponential", "erlang", "gamma", "h2")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return int(value)


class DynamicBatchingCalc(BaseQueue):  # pylint: disable=too-many-public-methods
    """Solve the embedded chain with explicitly reported truncation diagnostics.

    max_batch=None refines until every convergence diagnostic meets tolerance.
    max_batch=N pins a numerical truncation (not a batch cap). max_states and
    max_nonzeros limit matrix construction, not sparse-LU fill-in. Mean
    waiting/latency require rho**2 * (1 + service_scv) < 1.
    Only the first waiting/sojourn moments are implemented.
    """

    def __init__(self, max_batch=None, tolerance=1e-9, max_states=20_000, max_nonzeros=2_000_000):
        super().__init__(n=1)
        self.max_states = _positive_integer(max_states, "max_states")
        self.max_nonzeros = _positive_integer(max_nonzeros, "max_nonzeros")
        self.max_batch = None if max_batch is None else _positive_integer(max_batch, "max_batch")
        if self.max_batch is not None and self.max_batch > self.max_states:
            raise ValueError("max_batch must not exceed max_states")
        if not np.isfinite(tolerance) or not 0 < tolerance < 1:
            raise ValueError("tolerance must be finite and lie in (0, 1)")
        self.tolerance = float(tolerance)
        self.rate = None
        self.kind = None
        self.alpha = self.tau0 = 0.0
        self.shape = 1.0
        self.service_scv = 0.0
        self.h2_probability = 0.5
        self._cache = None

    def set_sources(self, rate):
        """Set a finite, strictly positive Poisson arrival rate."""
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("arrival rate must be finite and positive")
        self.rate = float(rate)
        self.is_sources_set = True
        self._cache = None
        return self

    def set_servers(self, alpha, tau0, *, kind="deterministic", shape=1, coev=1.0):
        # pylint: disable=too-many-arguments
        """Configure affine-mean service; H2 uses balanced means and CV coev.

        Gamma's shape may be any positive real; Erlang's must be an integer.
        H2 branch means are m/(2p), m/(2(1-p)), p=(1+sqrt((CV²-1)/(CV²+1)))/2.
        """
        if kind not in SERVICE_KINDS:
            raise ValueError(f"kind must be one of {SERVICE_KINDS}")
        if not np.isfinite(alpha) or not np.isfinite(tau0) or alpha < 0 or tau0 < 0:
            raise ValueError("alpha and tau0 must be finite and non-negative")
        if alpha == 0 and tau0 == 0:
            raise ValueError("a batch cannot take zero time for every size")
        if kind == "erlang":
            shape = _positive_integer(shape, "shape")
        if kind in ("erlang", "gamma") and (not np.isfinite(shape) or shape <= 0):
            raise ValueError("shape must be finite and positive")
        if kind == "h2" and (not np.isfinite(coev) or coev < 1 or coev > np.sqrt(np.finfo(float).max)):
            raise ValueError("H2 coev must be finite and at least 1")
        scv = {"deterministic": 0.0, "exponential": 1.0}.get(kind)
        if scv is None:
            scv = coev**2 if kind == "h2" else 1.0 / shape
        probability = 0.5 * (1 + np.sqrt((scv - 1) / (scv + 1))) if kind == "h2" else 0.5
        if probability >= 1 or not np.isfinite(scv):
            raise ValueError("service variability is too large for floating-point representation")
        self.kind, self.alpha, self.tau0 = kind, float(alpha), float(tau0)
        self.shape = float(shape) if kind in ("gamma", "erlang") else 1.0
        self.service_scv, self.h2_probability = scv, probability
        self.is_servers_set = True
        self._cache = None
        return self

    def set_deterministic_service(self, alpha, tau0):
        """Set deterministic batch duration alpha*b + tau0."""
        return self.set_servers(alpha, tau0)

    def set_exponential_service(self, alpha, tau0):
        """Set exponential batch duration with affine mean."""
        return self.set_servers(alpha, tau0, kind="exponential")

    def set_erlang_service(self, alpha, tau0, shape):
        """Set Erlang batch duration with affine mean and integer shape."""
        return self.set_servers(alpha, tau0, kind="erlang", shape=shape)

    def set_gamma_service(self, alpha, tau0, shape):
        """Set gamma batch duration with affine mean and positive real shape."""
        return self.set_servers(alpha, tau0, kind="gamma", shape=shape)

    def set_h2_service(self, alpha, tau0, coev):
        """Set balanced-means H2 batch duration with affine mean and CV >= 1."""
        return self.set_servers(alpha, tau0, kind="h2", coev=coev)

    def _check_ready(self):
        if self.rate is None:
            raise ValueError("set_sources must be called first")
        if self.kind is None:
            raise ValueError("a service law must be set first")

    @property
    def load(self):
        """Marginal-throughput load rho=lambda*alpha, NOT busy-time utilization."""
        self._check_ready()
        return self.rate * self.alpha

    def _check_domain(self):
        self._check_ready()
        if self.load >= 1:
            raise ValueError("unstable finite-mean regime: rho = lambda * alpha must be below 1")
        if self.load**2 * (1 + self.service_scv) >= 1:
            raise ValueError("mean waiting/latency is infinite: rho**2 * (1 + service_scv) must be below 1")

    def _count_components(self, mean):
        if self.kind == "deterministic":
            return [(1.0, poisson(self.rate * mean))]
        if self.kind == "h2":
            p = self.h2_probability
            return [(weight, nbinom(1, 1 / (1 + self.rate * mean / (2 * weight)))) for weight in (p, 1 - p)]
        return [(1.0, nbinom(self.shape, 1 / (1 + self.rate * mean / self.shape)))]

    def _support(self, mean, limit):
        components = self._count_components(mean)
        low = min(float(dist.ppf(1e-15)) for _, dist in components)
        high = max(float(dist.isf(1e-15)) for _, dist in components)
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("arrival-count support is not representable")
        return max(2, min(limit, int(low) - 2)), max(1, min(limit - 1, int(high) + 2))

    def _solve_chain(self, size):
        if 4 * size > self.max_nonzeros:
            raise MemoryError(f"matrix construction exceeds the entry budget {self.max_nonzeros:,}")
        sizes = np.arange(1, size + 1, dtype=float)
        means = self.alpha * sizes + self.tau0
        if not np.all(np.isfinite(self.rate * means)):
            raise ValueError("arrival/service scale overflows floating-point representation")
        supports = []
        # Stop pricing as soon as the budget is exhausted, before allocating P.
        estimate = 2 * size
        for mean in means:
            low, high = self._support(mean, size)
            estimate += max(0, high - low + 1) + 2
            if estimate > self.max_nonzeros:
                raise MemoryError(f"matrix construction exceeds the entry budget {self.max_nonzeros:,}")
            supports.append((low, high))
        rows, cols, values, zeros = [], [], [], []
        for index, (mean, (low, high)) in enumerate(zip(means, supports)):
            components = self._count_components(mean)
            counts = np.arange(low, high + 1)
            middle = sum(weight * dist.pmf(counts) for weight, dist in components)
            first = sum(weight * dist.cdf(low - 1) for weight, dist in components)
            last = sum(weight * dist.sf(high) for weight, dist in components)
            zeros.append(sum(weight * dist.pmf(0) for weight, dist in components))
            if size == 1:
                targets, weights = np.array([0]), np.array([1.0])
            else:
                targets = np.concatenate(([0], counts - 1, [size - 1]))
                weights = np.concatenate(([first], middle, [last]))
                weights /= weights.sum()
            cols.append(targets)
            rows.append(np.full(targets.size, index))
            values.append(weights)
        matrix = csc_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(size, size))
        system = (matrix.T - identity(size, format="csc")).tolil()
        system[0, :] = 1.0
        rhs = np.zeros(size)
        rhs[0] = 1.0
        pi = np.asarray(spsolve(csc_matrix(system), rhs)).ravel()
        if not np.all(np.isfinite(pi)) or pi.min() < -1e-10 or pi.sum() <= 0:
            raise RuntimeError("the embedded chain has no usable stationary distribution")
        pi = np.maximum(pi, 0)
        pi /= pi.sum()
        # Sparse LU can leave tiny positive mass in otherwise empty far-tail
        # states. Their b² weight matters. Positive iterations remove this
        # numerical contamination without arbitrarily clipping rare states.
        weights = 1 + sizes**2
        for _ in range(10_000):
            updated = np.asarray(matrix.T @ pi).ravel()
            updated /= updated.sum()
            change = float(np.abs(updated - pi) @ weights) / float(updated @ weights)
            pi = updated
            if change < min(self.tolerance * 0.01, 1e-14):
                break
        else:
            raise RuntimeError("stationary iteration did not converge")
        residual = float(np.linalg.norm(matrix.T @ pi - pi, ord=1))
        if residual > max(self.tolerance, 1e-12):
            raise RuntimeError(f"stationary residual is too large: {residual:.3g}")
        return pi, sizes, np.asarray(zeros), residual

    def _diagnostics(self, pi, sizes, zeros):
        mean, second = float(pi @ sizes), float(pi @ sizes**2)
        means = self.alpha * sizes + self.tau0
        # Untruncated identity: E[B(B-1)] = lambda² E[H²].
        factorial = float(pi @ (sizes * (sizes - 1)))
        expected = self.rate**2 * (1 + self.service_scv) * float(pi @ means**2)
        flow = abs(mean - self.rate * float(pi @ means) - float(pi @ zeros)) / mean
        moment = abs(factorial - expected) / max(expected, np.finfo(float).tiny)
        upper = sizes >= 0.8 * sizes[-1]
        tail = float(pi[upper] @ sizes[upper] ** 2) / second
        wait = factorial / (2 * self.rate * mean)
        service = float(pi @ (sizes * means)) / mean
        idle = float(pi @ zeros) / mean  # positive, avoids cancellation in 1-busy
        metrics = np.array([mean, second, wait, service, wait + service])
        return (
            metrics,
            idle,
            {"boundary_second_moment_fraction": tail, "flow_residual": flow, "moment_residual": moment},
        )

    def _stationary(self):
        self._check_domain()
        if self._cache is not None:
            return self._cache
        size = self.max_batch or min(64, self.max_states)
        previous = None
        while True:
            pi, sizes, zeros, residual = self._solve_chain(size)
            metrics, idle, diagnostics = self._diagnostics(pi, sizes, zeros)
            refinement = (
                None
                if previous is None
                else float(np.max(np.abs(metrics - previous) / np.maximum(np.abs(metrics), np.finfo(float).tiny)))
            )
            converged = refinement is not None and max(refinement, residual, *diagnostics.values()) <= self.tolerance
            diagnostics.update(stationary_residual=residual, refinement_error=refinement)
            cache = (pi, sizes, metrics, idle, diagnostics, converged)
            if self.max_batch is not None or converged:
                self._cache = cache
                return cache
            if size == self.max_states:
                raise ValueError(f"truncation did not converge within max_states={size}; diagnostics={diagnostics}")
            previous = metrics
            size = min(2 * size, self.max_states)

    def get_batch_distribution(self):
        """Return a copy of the batch-epoch PMF (index 0 is batch size 1)."""
        return self._stationary()[0].copy()

    def get_batch_moments(self, num=2):
        """Return first one or two raw batch moments; higher orders are unsupported."""
        num = _positive_integer(num, "num")
        if num > 2:
            raise ValueError("only the first two batch moments are supported")
        return self._stationary()[2][:num].tolist()

    def get_waiting_mean(self):
        """Mean arrival-to-batch-start time from Inoue's Lemma 2."""
        return float(self._stationary()[2][2])

    def get_w(self, num=1):
        """Return [mean waiting time]; higher waiting moments are not implemented."""
        if _positive_integer(num, "num") != 1:
            raise ValueError("only the first waiting moment is supported")
        return [self.get_waiting_mean()]

    def get_service_time_mean(self):
        """Size-biased processing mean for a random job, not a random batch."""
        return float(self._stationary()[2][3])

    def get_latency(self):
        """Mean sojourn time (called E[W] in the source paper)."""
        return float(self._stationary()[2][4])

    def get_v(self, num=1):
        """Return [mean sojourn time]; higher sojourn moments are not implemented."""
        if _positive_integer(num, "num") != 1:
            raise ValueError("only the first sojourn moment is supported")
        return [self.get_latency()]

    def get_idle_probability(self):
        """Time-stationary idle probability from renewal reward P(A=0)/E[B]."""
        return self._stationary()[3]

    def get_n_mean(self):
        """Mean number of requests in the system, using Little's law."""
        latency = self.get_latency()
        return self.rate * latency

    def _require_deterministic(self):
        self._check_domain()
        if self.kind != "deterministic":
            raise ValueError("Theorem 2 requires Assumption 4: deterministic affine service")

    def _lemma4(self, idle):
        lam, alpha, tau0 = self.rate, self.alpha, self.tau0
        inner = 2 * alpha * tau0 + alpha**2 + (1 - idle - lam * alpha) * tau0 / lam
        return alpha + tau0 + lam * (1 + 2 * lam * alpha) * inner / (2 * (1 - (lam * alpha) ** 2))

    def bound_phi0(self):
        """Theorem 2, equation (41): upper bound on mean LATENCY."""
        self._require_deterministic()
        lam, alpha, tau0 = self.rate, self.alpha, self.tau0
        return (alpha + tau0) / (2 * (1 - lam * alpha)) * (1 + 2 * lam * tau0 + (1 - lam * tau0) / (1 + lam * alpha))

    def bound_phi1(self):
        """Theorem 2, equation (42): latency bound obtained by setting idle=0."""
        self._require_deterministic()
        rho = self.load
        return 1.5 * self.tau0 / (1 - rho) + 0.5 * self.alpha * (rho + 2) / (1 - rho**2)

    def bound_report(self):
        """Compare published bounds to the numerical chain, not to a new exact formula."""
        self._require_deterministic()
        latency, idle = self.get_latency(), self.get_idle_probability()
        phi0, phi1 = self.bound_phi0(), self.bound_phi1()
        lower = max(0.0, 1 - self.rate * (self.alpha + self.tau0))
        coefficient = (1 + 2 * self.load) * self.tau0 / (2 * (1 - self.load**2))
        return {
            "latency": latency,
            "phi0": phi0,
            "phi1": phi1,
            "best_bound": min(phi0, phi1),
            "best_bound_ratio": min(phi0, phi1) / latency,
            "idle_probability": idle,
            "idle_lower_bound": lower,
            "lemma4_at_idle": self._lemma4(idle),
            "slack_from_idle": coefficient * (idle - lower),
        }

    def run(self):
        """Return QueueResults-compatible means plus numerical diagnostics."""
        start = time.process_time()
        pi, sizes, metrics, idle, diagnostics, converged = self._stationary()
        mean, second, waiting, service, latency = map(float, metrics)
        return DynamicBatchingResults(
            w=[waiting],
            v=[latency],
            waiting_mean=waiting,
            latency_mean=latency,
            service_mean=service,
            batch_mean=mean,
            batch_second_moment=second,
            idle_probability=idle,
            utilization=1 - idle,
            jobs_mean=self.rate * latency,
            max_batch=int(sizes[-1]),
            batch_distribution=pi.copy(),
            converged=converged,
            duration=time.process_time() - start,
            **diagnostics,
        )

    def __repr__(self):
        return f"DynamicBatchingCalc(kind={self.kind!r}, alpha={self.alpha}, tau0={self.tau0})"
