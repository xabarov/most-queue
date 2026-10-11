"""Independent event simulation of Inoue's unbounded dynamic batching.

Y. Inoue, Performance Evaluation 147 (2021), 102183,
doi:10.1016/j.peva.2020.102183. Arrivals and actual completion times, not
the calculator's transition probabilities, determine the trajectory.
All statistics exclude warmup at whole-batch completion boundaries.
A fixed warmup fraction is a convention, not a guarantee of stationarity.
"""

import time
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from most_queue.structs import QueueResults


@dataclass
class DynamicBatchingSimResults(QueueResults):
    """Post-warmup batch-cohort samples and time averages on the same interval.

    served can exceed the requested count: the last batch is kept whole.
    w/v contain three empirical raw moments; p is the time-occupancy PMF.
    """

    waiting_mean: float = 0.0
    latency_mean: float = 0.0
    service_mean: float = 0.0
    batch_mean: float = 0.0
    idle_fraction: float = 0.0
    served: int = 0
    batches: int = 0
    warmup_served: int = 0
    observation_time: float = 0.0
    waits: np.ndarray = field(default_factory=lambda: np.zeros(0))
    latencies: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def cdf(self, t, of="w"):
        """Empirical waiting (w) or sojourn (v) CDF."""
        if of not in ("w", "v"):
            raise ValueError("of must be 'w' or 'v'")
        if np.isnan(t):
            raise ValueError("CDF time must not be NaN")
        sample = self.waits if of == "w" else self.latencies
        if sample.size == 0:
            raise ValueError("no samples were collected")
        return float((sample <= t).mean())


class DynamicBatchingSim:
    """Poisson arrivals, all-waiting dispatch, affine-mean batch processing."""

    def __init__(self, seed=None):
        self.generator = np.random.default_rng(seed)
        self.rate = None
        self.kind = None
        self.alpha = self.tau0 = 0.0
        self.shape = 1.0
        self.h2_probability = 0.5

    def set_sources(self, rate):
        """Set a finite positive Poisson arrival rate."""
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("arrival rate must be finite and positive")
        self.rate = float(rate)
        return self

    def set_servers(self, alpha, tau0, *, kind="deterministic", shape=1, coev=1.0):
        # pylint: disable=too-many-arguments
        """Set affine mean and deterministic, exponential, Erlang, gamma or H2 law.

        H2 uses balanced branch means with CV=coev >= 1.
        Gamma accepts a positive real shape; Erlang requires an integer.
        """
        if kind not in ("deterministic", "exponential", "erlang", "gamma", "h2"):
            raise ValueError("unknown service kind")
        if not np.isfinite(alpha) or not np.isfinite(tau0) or alpha < 0 or tau0 < 0:
            raise ValueError("alpha and tau0 must be finite and non-negative")
        if alpha == 0 and tau0 == 0:
            raise ValueError("a batch cannot take zero time for every size")
        if kind == "erlang" and (isinstance(shape, bool) or not isinstance(shape, Integral)):
            raise ValueError("Erlang shape must be a positive integer")
        if kind in ("erlang", "gamma") and (not np.isfinite(shape) or shape <= 0):
            raise ValueError("shape must be finite and positive")
        if kind == "h2" and (not np.isfinite(coev) or coev < 1 or coev > np.sqrt(np.finfo(float).max)):
            raise ValueError("H2 coev must be finite and at least 1")
        p = 0.5 * (1 + np.sqrt((coev**2 - 1) / (coev**2 + 1))) if kind == "h2" else 0.5
        if p >= 1:
            raise ValueError("service variability is too large for floating-point representation")
        self.kind, self.alpha, self.tau0 = kind, float(alpha), float(tau0)
        self.shape = float(shape) if kind in ("erlang", "gamma") else 1.0
        self.h2_probability = p
        return self

    def set_deterministic_service(self, alpha, tau0):
        """Set deterministic duration alpha*b + tau0."""
        return self.set_servers(alpha, tau0)

    def set_exponential_service(self, alpha, tau0):
        """Set exponential duration with affine mean."""
        return self.set_servers(alpha, tau0, kind="exponential")

    def set_erlang_service(self, alpha, tau0, shape):
        """Set Erlang duration with affine mean."""
        return self.set_servers(alpha, tau0, kind="erlang", shape=shape)

    def set_gamma_service(self, alpha, tau0, shape):
        """Set gamma duration with affine mean."""
        return self.set_servers(alpha, tau0, kind="gamma", shape=shape)

    def set_h2_service(self, alpha, tau0, coev):
        """Set balanced-means H2 duration with affine mean."""
        return self.set_servers(alpha, tau0, kind="h2", coev=coev)

    def _draw(self, size):
        mean = self.alpha * size + self.tau0
        if self.kind == "deterministic":
            return mean
        if self.kind == "exponential":
            return float(self.generator.exponential(mean))
        if self.kind == "h2":
            p = self.h2_probability
            weight = p if self.generator.random() < p else 1 - p
            return float(self.generator.exponential(mean / (2 * weight)))
        return float(self.generator.gamma(self.shape, mean / self.shape))

    def run(self, total_served=200_000, warmup_fraction=0.2, *, num_of_jobs=None):
        """Collect at least total_served jobs, finishing the last batch.

        num_of_jobs is the common simulator-interface alias. Discard at least
        floor(total_served*f/(1-f)) completed warmup jobs, then collect whole
        batches. Idle, occupancy and batch means use this same observation
        window. Repeat with larger warmup/run budgets when diagnosing bias.
        """
        if num_of_jobs is not None:
            total_served = num_of_jobs
        if self.rate is None or self.kind is None:
            raise ValueError("set_sources and a service law must be set first")
        if self.rate * self.alpha >= 1:
            raise ValueError("require rho = lambda * alpha < 1 for stationary simulation")
        if isinstance(total_served, bool) or not isinstance(total_served, Integral) or total_served < 1:
            raise ValueError("total_served must be a positive integer")
        if not np.isfinite(warmup_fraction) or not 0 <= warmup_fraction < 1:
            raise ValueError("warmup_fraction must lie in [0, 1)")
        start = time.process_time()
        warmup_target = int(total_served * warmup_fraction / (1 - warmup_fraction))
        completed = warmup_served = 0
        waits, latencies, batch_sizes = [], [], []
        state_times = {}
        queue = []
        clock = observation_start = 0.0
        next_arrival = float(self.generator.exponential(1 / self.rate))

        while len(waits) < total_served:
            measuring = completed >= warmup_target
            if not queue:
                if measuring:
                    state_times[0] = state_times.get(0, 0.0) + next_arrival - clock
                clock = next_arrival
                queue.append(clock)
                next_arrival = clock + float(self.generator.exponential(1 / self.rate))
            batch, queue = queue, []
            size = len(batch)
            duration = self._draw(size)
            begin, finish = clock, clock + duration
            if not np.isfinite(finish) or finish <= begin:
                raise RuntimeError("service time cannot advance the simulation clock")
            while next_arrival <= finish:
                if measuring:
                    count = size + len(queue)
                    state_times[count] = state_times.get(count, 0.0) + next_arrival - clock
                clock = next_arrival
                queue.append(clock)
                increment = float(self.generator.exponential(1 / self.rate))
                if clock + increment <= clock:
                    raise RuntimeError("arrival time cannot advance the simulation clock")
                next_arrival = clock + increment
            if measuring:
                count = size + len(queue)
                state_times[count] = state_times.get(count, 0.0) + finish - clock
                waits.extend(begin - joined for joined in batch)
                latencies.extend(finish - joined for joined in batch)
                batch_sizes.append(size)
            else:
                observation_start = finish
                warmup_served += size
            completed += size
            clock = finish

        w, v = np.asarray(waits), np.asarray(latencies)
        elapsed = clock - observation_start
        p = np.zeros(max(state_times) + 1)
        for count, duration in state_times.items():
            p[count] = duration / elapsed
        idle = float(p[0])
        return DynamicBatchingSimResults(
            w=[float(np.mean(w**order)) for order in (1, 2, 3)],
            v=[float(np.mean(v**order)) for order in (1, 2, 3)],
            p=p.tolist(),
            utilization=1 - idle,
            waiting_mean=float(w.mean()),
            latency_mean=float(v.mean()),
            service_mean=float((v - w).mean()),
            batch_mean=float(np.mean(batch_sizes)),
            idle_fraction=idle,
            served=w.size,
            batches=len(batch_sizes),
            warmup_served=warmup_served,
            observation_time=elapsed,
            duration=time.process_time() - start,
            waits=w,
            latencies=v,
        )

    def __repr__(self):
        return f"DynamicBatchingSim(kind={self.kind!r}, alpha={self.alpha}, tau0={self.tau0})"
