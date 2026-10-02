"""Prediction-free packing decisions, separated from service-time event data.

Quickswap: Chen et al. (2026), doi:10.1016/j.peva.2025.102525, Section 4.
ServerFilling: Grosof and Harchol-Balter (2023), doi:10.1145/3584684.3597264.
Inputs are (arrival-order ID, resource need), never service or predictions.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

PACKING_POLICIES = ("first_fit", "msf", "msfq", "adaptive_quickswap", "server_filling")


def _greedy(jobs, free):
    selected = []
    for idx, need in jobs:
        if need <= free:
            selected.append(idx)
            free -= need
        if not free:
            break
    return selected


def server_filling_selection(k: int, unfinished: Iterable[tuple[int, int]]) -> list[int]:
    """Pack the minimal arrival-ordered prefix with total need >= k.

    Caller validates that k and all needs are powers of two. Under that
    restriction, a full candidate prefix packs exactly k servers. Ties are FIFO.
    """
    prefix, total = [], 0
    for idx, need in unfinished:
        prefix.append((idx, need))
        total += need
        if total >= k:
            break
    return _greedy(sorted(prefix, key=lambda job: (-job[1], job[0])), k)


@dataclass
class NonpreemptivePacking:
    """Stateful MSFQ/Adaptive decisions; instantiate anew for every replay.

    MSFQ implements the literal four-phase definition (inclusive threshold),
    admitting an initial small batch before testing the threshold on entry.
    A draining phase stays closed even when no wide job is currently waiting.
    Adaptive tests its trigger after the MSF admission batch, as in the authors'
    simulator. Resource classes are defined by need, not external class labels.
    """

    k: int
    discipline: str
    threshold: int = 0
    phase: str = "working"

    def select(self, waiting: Sequence[tuple[int, int]], active: Sequence[tuple[int, int]]) -> list[int]:
        """Select new admissions, without interrupting any active job."""
        free = self.k - sum(need for _, need in active)
        ordered = sorted(waiting, key=lambda job: (-job[1], job[0]))
        if self.discipline == "first_fit":
            return _greedy(waiting, free)
        if self.discipline == "msf" or (self.discipline == "msfq" and self.k == 1):
            return _greedy(ordered, free)
        if self.discipline == "msfq":
            return self._msfq(waiting, active, free)
        if self.phase == "draining":
            if not ordered or ordered[0][1] > free:
                return []
            # The largest waiting job now starts; ordinary MSF resumes.
            self.phase = "working"
        selected = _greedy(ordered, free)
        chosen = set(selected)
        waiting_needs = {need for idx, need in waiting if idx not in chosen}
        running_needs = {need for _, need in active} | {need for idx, need in waiting if idx in chosen}
        if waiting_needs - running_needs and not waiting_needs & running_needs:
            self.phase = "draining"
        return selected

    def _msfq(self, waiting, active, free):
        if self.phase == "draining" and active:
            return []
        if not active:
            wide = [idx for idx, need in waiting if need == self.k]
            if wide:
                self.phase = "wide"
                return wide[:1]
            self.phase = "small"
        if any(need == self.k for _, need in active):
            return []
        small = [(idx, need) for idx, need in waiting if need == 1]
        n_small = len(small) + len(active)
        if active and n_small <= self.threshold:
            self.phase = "draining"
            return []
        selected = _greedy(small, free)
        if n_small <= self.threshold:
            self.phase = "draining"
        return selected
