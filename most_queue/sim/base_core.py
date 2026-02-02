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

    def __init__(self):
        """Initialize the base simulation core."""
        self.ttek = 0  # current simulation time
        self.generator = np.random.default_rng()
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
