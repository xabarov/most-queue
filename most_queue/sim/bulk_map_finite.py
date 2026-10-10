"""
Discrete-event simulator for the finite-buffer bulk-service queue with
correlated (MAP) arrivals: MAP/PH^(a,b)/1/N.

Companion to the exact analysis in
:mod:`most_queue.theory.batch.map_ph_finite_buffer`, whose model this follows
exactly: arrivals from a MAP ``(D0, D1)``, a buffer holding at most ``N``
waiting customers (the batch in service does not occupy it), the general
``(a, b)`` bulk rule, and a PH-distributed batch service time that does not
depend on how many customers the batch holds.

Why it exists. The exact method builds the waiting time as a phase-type
distribution on a tagged-customer chain, and the one thing that construction
cannot check about itself is whether it describes the intended system. Playing
the system out event by event and measuring what customers actually wait is the
reference that settles that.

Note that the sojourn time is the queueing time plus one WHOLE batch service:
every customer in a batch leaves when the batch finishes, not when its own
share of the work is done.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from most_queue.random.map_ph import MAP, MAPParams, PHDistribution, PHParams

_EPS = 1e-12


@dataclass
class BulkMapSimResults:
    """Outcome of a simulated run."""

    w: list[float] = field(default_factory=list)  # queueing-time raw moments
    v: list[float] = field(default_factory=list)  # sojourn-time raw moments
    loss_probability: float = 0.0
    served: int = 0
    arrived: int = 0
    blocked: int = 0
    batches: int = 0
    mean_batch_size: float = 0.0
    utilization: float = 0.0
    duration: float = 0.0
    waits: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sojourns: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def cdf(self, t: float, of: str = "w") -> float:
        """Empirical CDF of the queueing (``"w"``) or sojourn (``"v"``) time."""
        sample = self.waits if of == "w" else self.sojourns
        if sample.size == 0:
            raise ValueError("no samples were collected")
        return float((sample <= t).mean())


class BulkServiceMapPhSim:
    """
    MAP/PH^(a,b)/1/N by simulation.

    :param a: the server waits until ``a`` customers have accumulated.
    :param b: at most ``b`` customers go into one batch.
    :param capacity: buffer size ``N``, waiting customers only.
    :param seed: RNG seed.

    Usage::

        sim = BulkServiceMapPhSim(a=3, b=6, capacity=200, seed=1)
        sim.set_poisson_sources(6.7)
        sim.set_exponential_servers(1.7)
        results = sim.run(total_served=500_000)
        results.w[0], results.cdf(2.0)
    """

    def __init__(self, a: int = 1, b: int = 1, capacity: int = 100, seed: int | None = None):
        if a < 1:
            raise ValueError(f"a must be at least 1, got {a}")
        if b < a:
            raise ValueError(f"b must be at least a, got a={a}, b={b}")
        if capacity < a:
            raise ValueError(f"capacity must be at least a, got N={capacity}, a={a}")
        self.a = int(a)
        self.b = int(b)
        self.capacity = int(capacity)
        self.generator = np.random.default_rng(seed)
        self.source: MAP | None = None
        self.service: PHDistribution | None = None

    def set_sources(self, map_params: MAPParams):
        """Arrival MAP ``(D0, D1)``."""
        self.source = MAP(map_params, self.generator)
        return self

    def set_poisson_sources(self, rate: float):
        """Shorthand for Poisson arrivals."""
        if rate <= 0:
            raise ValueError(f"arrival rate must be positive, got {rate}")
        return self.set_sources(MAPParams(D0=np.array([[-rate]]), D1=np.array([[rate]])))

    def set_servers(self, ph_params: PHParams):
        """PH-distributed batch service time."""
        self.service = PHDistribution(ph_params, self.generator)
        return self

    def set_exponential_servers(self, rate: float):
        """Shorthand for exponential batch service."""
        if rate <= 0:
            raise ValueError(f"service rate must be positive, got {rate}")
        return self.set_servers(PHParams(alpha=np.array([1.0]), T=np.array([[-rate]])))

    def run(self, total_served: int = 200_000, warmup_fraction: float = 0.05) -> BulkMapSimResults:
        """
        Run until ``total_served`` customers have departed.

        :param warmup_fraction: leading fraction of departures discarded, so
            the starting state (empty and idle) does not bias the averages.
        """
        if self.source is None or self.service is None:
            raise ValueError("set_sources and set_servers must be called first")
        if total_served < 1:
            raise ValueError(f"total_served must be positive, got {total_served}")
        if not 0.0 <= warmup_fraction < 1.0:
            raise ValueError(f"warmup_fraction must lie in [0, 1), got {warmup_fraction}")
        start = time.process_time()

        a, b, cap = self.a, self.b, self.capacity
        waits: list[float] = []
        sojourns: list[float] = []
        queue: list[float] = []  # arrival times of waiting customers
        in_service: list[float] = []
        clock = 0.0
        busy_until: float | None = None
        arrived = blocked = batches = 0
        batch_total = 0
        busy_time = 0.0
        next_arrival = self.source.generate()

        def start_batch(now: float):
            nonlocal busy_until, batches, batch_total, in_service, queue, busy_time
            take = min(len(queue), b)
            in_service = queue[:take]
            queue = queue[take:]
            for joined in in_service:
                waits.append(now - joined)
            duration = max(self.service.generate(), 0.0)
            busy_until = now + duration
            busy_time += duration
            batches += 1
            batch_total += take

        while len(sojourns) < total_served:
            if busy_until is None or next_arrival < busy_until:
                clock = next_arrival
                next_arrival = clock + self.source.generate()
                arrived += 1
                if len(queue) < cap:
                    queue.append(clock)
                else:
                    blocked += 1
                if busy_until is None and len(queue) >= a:
                    start_batch(clock)
            else:
                clock = busy_until
                for joined in in_service:  # the whole batch leaves together
                    sojourns.append(clock - joined)
                in_service = []
                busy_until = None
                if len(queue) >= a:
                    start_batch(clock)

        cut = int(len(sojourns) * warmup_fraction)
        w = np.array(waits[cut : len(sojourns)])
        v = np.array(sojourns[cut:])
        return BulkMapSimResults(
            w=[float((w**k).mean()) for k in range(1, 5)] if w.size else [],
            v=[float((v**k).mean()) for k in range(1, 5)] if v.size else [],
            loss_probability=blocked / arrived if arrived else 0.0,
            served=len(v),
            arrived=arrived,
            blocked=blocked,
            batches=batches,
            mean_batch_size=batch_total / batches if batches else 0.0,
            utilization=busy_time / clock if clock > _EPS else 0.0,
            duration=time.process_time() - start,
            waits=w,
            sojourns=v,
        )

    def __repr__(self) -> str:
        return f"BulkServiceMapPhSim(a={self.a}, b={self.b}, capacity={self.capacity})"
