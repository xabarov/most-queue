"""
The exact clairvoyant optimum for value-based scheduling under overload on one
preemptive machine.

This module is NOT from Buttazzo, Spuri & Sensini (RTSS 1995) -- it is the
yardstick their experiments lack. The paper measures each on-line algorithm by
its Hit Value Ratio, the fraction of the task set's total value that the
algorithm actually banked, and then compares the algorithms with one another.
That normalisation is honest but pessimistic: under overload no scheduler can
collect the whole value, so an HVR of 0.8 may be a poor result or very nearly
the best possible, and the paper's data cannot tell those apart. Indeed the
paper remarks that "the performance of the robust algorithms is close to the
best one achievable by an on-line algorithm" without measuring the best
achievable.

Solving the off-line problem exactly fixes the denominator. An on-line rule can
then be scored against what a scheduler that knew the entire arrival sequence
and every actual execution time in advance would have banked on the very same
realisation, which turns the HVR into a true competitive ratio.

The problem. Choose a subset of the jobs to complete; a chosen job must receive
its full processing time inside its own window, and the machine runs one job at
a time with free preemption. Maximise the total value of the chosen subset.
Rejected jobs simply are not run -- under overload that is the whole point.

Why an integer program, and why its constraints are exactly right. For a FIXED
subset, feasibility on one preemptive machine with release dates and deadlines
is decided by Horn's condition: the subset is schedulable if and only if, for
every pair of times ``t_a < t_b``, the jobs whose windows nest inside
``[t_a, t_b]`` demand no more than ``t_b - t_a`` of machine time. Only pairs
where ``t_a`` is a release and ``t_b`` a deadline can be binding, and there are
finitely many. So with a binary ``y_j`` per job,

    maximise    sum_j value_j * y_j
    subject to  sum_{j: t_a <= r_j, d_j <= t_b} p_j * y_j <= t_b - t_a
                                                   for every such pair
                y_j in {0, 1}

is an exact reformulation and not a relaxation: its feasible points are
precisely the schedulable subsets. The selection is NP-hard (it contains
knapsack), hence the integrality; the scheduling half of the problem is what
Horn's condition dissolves.

Because the constraints are Horn's condition, an actual schedule for the chosen
subset can then be produced by the interval linear program already implemented
in :mod:`most_queue.theory.imprecise.offline`, and re-checked by that module's
independent :func:`~most_queue.theory.imprecise.offline.verify_schedule`. That
is deliberate: the integer program claims a subset is schedulable, and a
separately written solver is made to exhibit the schedule.

Reference for the hardness and for the limits of any on-line rule:
    Baruah S., Koren G., Mao D., Mishra B., Raghunathan A., Rosier L.,
    Shasha D., Wang F., "On the competitiveness of on-line real-time task
    scheduling", Real-Time Systems 4(2):125-144, 1992,
    doi:10.1007/BF00365406 -- no on-line algorithm can guarantee a competitive
    factor better than 1/4 against the clairvoyant optimum once the system is
    overloaded, whatever the value density.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from most_queue.theory.imprecise.offline import ImpreciseComputationScheduler, ImpreciseTask, verify_schedule

_TOL = 1e-7
_MAX_CONSTRAINTS = 400_000


@dataclass
class ValueTask:
    """
    One job as the off-line solver sees it: a window, a processing time and a
    value collected in full if the job completes inside the window, and not at
    all otherwise.

    The processing time here is the ACTUAL one. The clairvoyant optimum is
    allowed to know it; an on-line algorithm is not, and that asymmetry is a
    large part of what the comparison measures.
    """

    release: float
    deadline: float
    processing: float
    value: float
    name: str = ""

    def __post_init__(self):
        if self.deadline <= self.release:
            raise ValueError(f"deadline must exceed release, got {self.release} and {self.deadline}")
        if self.processing < 0.0:
            raise ValueError(f"processing time must be non-negative, got {self.processing}")
        if self.value < 0.0:
            raise ValueError(f"value must be non-negative, got {self.value}")

    @property
    def window(self) -> float:
        """Length of the interval in which the job may run."""
        return self.deadline - self.release

    @property
    def schedulable_alone(self) -> bool:
        """Whether the job could be completed if it were the only one."""
        return self.processing <= self.window + _TOL


@dataclass
class ClairvoyantResult:
    """Outcome of the exact off-line solve."""

    optimal_value: float = 0.0  # best total value any scheduler could bank
    total_value: float = 0.0  # value of the whole task set
    hit_value_ratio: float = 0.0  # optimal_value / total_value
    completed: int = 0  # jobs the optimum runs to completion
    rejected: int = 0  # jobs the optimum does not run at all
    selected: list[bool] = field(default_factory=list)
    schedule: list[tuple[float, float, int]] = field(default_factory=list)
    components: int = 0  # independent sub-problems the instance split into
    constraints: int = 0  # Horn rows actually passed to the solver
    duration: float = 0.0


class ClairvoyantValueScheduler:
    """
    Exact off-line (clairvoyant) value-maximising scheduler for one preemptive
    machine.

    :param tolerance: slack used when comparing times and loads.
    :param max_constraints: refuse instances needing more Horn rows than this,
        rather than silently thrashing. The row count grows roughly as the
        product of the number of distinct releases and deadlines within one
        connected block of overlapping windows, so a few hundred jobs is the
        practical ceiling for a single block.

    Usage::

        scheduler = ClairvoyantValueScheduler().set_tasks(tasks)
        result = scheduler.solve()
        result.hit_value_ratio   # the best any scheduler could have done
    """

    def __init__(self, tolerance: float = _TOL, max_constraints: int = _MAX_CONSTRAINTS):
        self.tolerance = float(tolerance)
        self.max_constraints = int(max_constraints)
        self.tasks: list[ValueTask] = []

    def set_tasks(self, tasks: list[ValueTask]):
        """Set the task set. Returns ``self`` so calls can be chained."""
        if not tasks:
            raise ValueError("task set is empty")
        self.tasks = list(tasks)
        return self

    def _check_ready(self):
        if not self.tasks:
            raise ValueError("set_tasks must be called first")

    def _components(self) -> list[list[int]]:
        """
        Split the instance into blocks that cannot interact.

        Sweep the jobs in release order and cut wherever the next release is at
        or after the furthest deadline seen so far. No Horn row can then be
        binding across a cut: a row spanning two blocks splits at the cut into
        two rows, one per block, whose capacities add up, so it is implied by
        the per-block rows and carries no extra information.
        """
        order = sorted(range(len(self.tasks)), key=lambda i: self.tasks[i].release)
        groups: list[list[int]] = []
        current: list[int] = []
        reach = -np.inf
        for i in order:
            task = self.tasks[i]
            if current and task.release >= reach - self.tolerance:
                groups.append(current)
                current = []
                reach = -np.inf
            current.append(i)
            reach = max(reach, task.deadline)
        if current:
            groups.append(current)
        return groups

    def _horn_rows(self, group: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Build Horn's condition for one block, keeping only rows that can
        actually bite.

        Two reductions, both exact. Several ``(t_a, t_b)`` pairs can select the
        same set of jobs; only the one with the smallest capacity matters, so
        rows are deduplicated by their job set and the tightest capacity kept.
        And a row whose jobs all together fit inside its capacity can never be
        violated by any subset of them, so it is dropped outright.
        """
        processing = np.array([self.tasks[i].processing for i in group])
        releases = np.array([self.tasks[i].release for i in group])
        deadlines = np.array([self.tasks[i].deadline for i in group])

        starts = np.unique(releases)
        ends = np.unique(deadlines)
        if starts.size * ends.size > self.max_constraints:
            raise ValueError(
                f"instance block has {len(group)} jobs with {starts.size} distinct releases and "
                f"{ends.size} distinct deadlines, needing up to {starts.size * ends.size} Horn rows, "
                f"above max_constraints={self.max_constraints}. The exact optimum is NP-hard; solve "
                f"a shorter horizon, or raise max_constraints if the memory is available."
            )

        tightest: dict[bytes, tuple[np.ndarray, float]] = {}
        for lo in starts:
            after = releases >= lo - self.tolerance
            if not after.any():
                continue
            for hi in ends:
                capacity = hi - lo
                if capacity <= self.tolerance:
                    continue
                mask = after & (deadlines <= hi + self.tolerance)
                if not mask.any():
                    continue
                if processing[mask].sum() <= capacity + self.tolerance:
                    continue  # cannot be violated by any subset
                key = mask.tobytes()
                known = tightest.get(key)
                if known is None or capacity < known[1]:
                    tightest[key] = (mask, capacity)

        if not tightest:
            return np.zeros((0, len(group))), np.zeros(0)
        rows = np.array([np.where(mask, processing, 0.0) for mask, _ in tightest.values()])
        rhs = np.array([capacity for _, capacity in tightest.values()])
        return rows, rhs

    def _solve_block(self, group: list[int]) -> tuple[list[int], int]:
        """Return the indices the optimum completes in this block, and the row count."""
        rows, rhs = self._horn_rows(group)
        values = np.array([self.tasks[i].value for i in group])
        if rows.shape[0] == 0:
            # Nothing can bind: every job in the block fits.
            return list(group), 0

        result = milp(
            c=-values,
            constraints=LinearConstraint(rows, -np.inf, rhs),
            integrality=np.ones(len(group)),
            bounds=Bounds(0, 1),
        )
        if not result.success:
            raise RuntimeError(f"integer program failed on a block of {len(group)} jobs: {result.message}")
        chosen = [group[k] for k in range(len(group)) if result.x[k] > 0.5]
        return chosen, rows.shape[0]

    def solve(self, emit_schedule: bool = True) -> ClairvoyantResult:
        """
        Compute the clairvoyant optimum.

        :param emit_schedule: also produce and independently re-check an actual
            schedule for the selected jobs. Costs a linear program on top of
            the integer one; turn it off for large sweeps.
        """
        self._check_ready()
        start = time.process_time()

        selected_flags = [False] * len(self.tasks)
        groups = self._components()
        rows_total = 0
        for group in groups:
            chosen, rows = self._solve_block(group)
            rows_total += rows
            for i in chosen:
                selected_flags[i] = True

        optimal = sum(task.value for task, keep in zip(self.tasks, selected_flags) if keep)
        total = sum(task.value for task in self.tasks)

        schedule: list[tuple[float, float, int]] = []
        if emit_schedule:
            schedule = self._schedule_for(selected_flags)

        return ClairvoyantResult(
            optimal_value=optimal,
            total_value=total,
            hit_value_ratio=optimal / total if total > 0 else 0.0,
            completed=sum(selected_flags),
            rejected=len(self.tasks) - sum(selected_flags),
            selected=selected_flags,
            schedule=schedule,
            components=len(groups),
            constraints=rows_total,
            duration=time.process_time() - start,
        )

    def _schedule_for(self, selected_flags: list[bool]) -> list[tuple[float, float, int]]:
        """
        Exhibit a schedule for the selected jobs, using the interval linear
        program of :mod:`most_queue.theory.imprecise.offline`.

        A selected job becomes a task that is entirely mandatory, so the LP has
        to place all of it inside its window or report infeasibility. Since the
        integer program has just asserted that the subset satisfies Horn's
        condition, infeasibility here would mean the two disagree, and that is
        raised rather than swallowed.
        """
        picked = [i for i, keep in enumerate(selected_flags) if keep]
        if not picked:
            return []
        tasks = [
            ImpreciseTask(
                release=self.tasks[i].release,
                deadline=self.tasks[i].deadline,
                mandatory=self.tasks[i].processing,
                optional=0.0,
            )
            for i in picked
        ]
        scheduler = ImpreciseComputationScheduler().set_tasks(tasks)
        if not scheduler.is_feasible():
            raise RuntimeError(
                "the integer program selected a subset that the interval LP calls infeasible; "
                "the two formulations disagree, which is a bug in one of them"
            )
        inner = scheduler.solve("total")
        if not verify_schedule(tasks, inner.schedule):
            raise RuntimeError("the emitted schedule failed its own verification")
        return [(begin, end, picked[local]) for begin, end, local in inner.schedule]

    def is_feasible(self) -> bool:
        """Whether the whole task set can be completed, leaving nothing behind."""
        self._check_ready()
        result = self.solve(emit_schedule=False)
        return result.rejected == 0

    def __repr__(self) -> str:
        return f"ClairvoyantValueScheduler(tasks={len(self.tasks)})"
