"""
Base simulation core with common functionality for all simulators
"""

import numpy as np

from most_queue.sim.utils.stats_update import refresh_moments_stat


class BaseSimulationCore:
    """
    Base class for queueing system simulators.
    Contains common attributes and methods shared across different simulator types.
    """

    def __init__(self, seed: int | None = None):
        """Initialize the base simulation core.

        :param seed: optional RNG seed for reproducible simulation runs
            (None — non-deterministic, as before).
        """
        self.ttek = 0  # current simulation time
        self.generator = np.random.default_rng(seed)
        self.time_spent = 0

        # Statistics for busy periods
        self.busy = [0, 0, 0]
        self.busy_moments = 0

        # Cache for server with minimum time
        self._min_server_time = 1e16
        self._min_server_idx = -1
        self._servers_time_changed = True

        # Placeholder for subclasses - v (sojourn) and w (wait) moments
        self.v: list[float] = [0, 0, 0, 0]
        self.w: list[float] = [0, 0, 0, 0]

        # Optional online deadline-violation counters (SLA cross-validation),
        # see most_queue.theory.utils.sla. Empty until set_deadline_thresholds
        # is called -- zero overhead for simulators that don't use it.
        self.deadline_thresholds: list[float] = []
        self.deadline_hits: dict[float, int] = {}
        self.deadline_n: int = 0

    def set_deadline_thresholds(self, thresholds: list[float]) -> None:
        """
        Start tracking, for each deadline D in `thresholds`, how many served
        tasks had wait_time > D. Call before run(); resets any previous count.
        """
        self.deadline_thresholds = list(thresholds)
        self.deadline_hits = dict.fromkeys(self.deadline_thresholds, 0)
        self.deadline_n = 0

    def _record_deadline_hit(self, wait_time: float) -> None:
        """Update deadline-violation counters for one observed wait time."""
        if not self.deadline_thresholds:
            return
        self.deadline_n += 1
        for d in self.deadline_thresholds:
            if wait_time > d:
                self.deadline_hits[d] += 1

    def get_empirical_violation_prob(self, deadline: float) -> float:
        """Empirical P(wait_time > deadline) accumulated since set_deadline_thresholds()."""
        if deadline not in self.deadline_hits:
            raise KeyError(f"deadline {deadline} is not tracked; pass it to set_deadline_thresholds() before run()")
        if self.deadline_n == 0:
            raise RuntimeError("no samples recorded yet -- call run() first")
        return self.deadline_hits[deadline] / self.deadline_n

    def refresh_busy_stat(self, new_a: float, count: int = None) -> None:
        """
        Update statistics of the busy period.

        Args:
            new_a: New busy period duration to include in statistics.
            count: Number of busy periods (if None, uses self.busy_moments)
        """
        if count is None:
            count = self.busy_moments
        self.busy = refresh_moments_stat(self.busy, new_a, count)

    def refresh_v_stat(self, new_a: float, count: int = None) -> None:
        """
        Update statistics of sojourn times.

        Args:
            new_a: New sojourn time value to include in statistics.
            count: Number of served tasks (if None, should be overridden by subclass)
        """
        # This is a placeholder - subclasses should override with proper count
        if count is not None:
            self.v = refresh_moments_stat(self.v, new_a, count)

    def refresh_w_stat(self, new_a: float, count: int = None) -> None:
        """
        Update statistics of wait times.

        Args:
            new_a: New waiting time value to include in statistics.
            count: Number of taken tasks (if None, should be overridden by subclass)
        """
        # This is a placeholder - subclasses should override with proper count
        if count is not None:
            self.w = refresh_moments_stat(self.w, new_a, count)

    def _get_min_server_time(self):
        """
        Get server with minimum time to end service.
        Uses caching to avoid O(n) search on every call.

        This is a base implementation that should be overridden
        if the simulator uses a different server structure.

        Returns:
            tuple: (server_index, min_time) or (-1, float('inf')) if no servers
        """
        # Base implementation - subclasses should override
        return -1, float("inf")

    def _mark_servers_time_changed(self):
        """Mark that server times have changed and cache needs refresh."""
        self._servers_time_changed = True
