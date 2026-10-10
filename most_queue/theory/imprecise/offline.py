"""
Exact offline scheduling of imprecise computations on one preemptive machine.

Implementation of
    Shih W.-K., Liu J.W.S., Chung J.-Y., Gillies D.W., "Scheduling tasks with
    ready times and deadlines to minimize average error", ACM SIGOPS Operating
    Systems Review 23(3):14-28, 1989, doi:10.1145/71021.71022;
    Shih W.-K., Liu J.W.S., Chung J.-Y., "Algorithms for scheduling tasks to
    minimize total error", SIAM J. Computing 20(3):537-552, 1991;
    Shih W.-K., Liu J.W.S., "Algorithms for scheduling imprecise computations
    with timing constraints to minimize maximum error", IEEE Trans. Computers
    44(3):466-471, 1995;
    unified with the controllable-processing-times literature by
    Shioura A., Shakhlevich N.V., Strusevich V.A., European Journal of
    Operational Research 266(3):795-818, 2018,
    doi:10.1016/j.ejor.2017.08.034.

THE MODEL AND THE SOLVED PROBLEMS ARE THEIRS. What is contributed here is an
implementation, and -- as in the rest of this catch-up series -- by a different
computational route: the literature states the problem as a minimum-cost
maximum-flow network (Shioura et al. spell out the network explicitly), whereas
this module solves the equivalent linear program directly. The two give the same
optimum; the LP is used because the data here are real-valued rather than
integral, and because it extends to the max-error objective without rebuilding
the network. The tests check the LP against a min-cost flow solved by network
simplex, and against exhaustive search on small integer instances.

The model. Each task has a release time, a deadline, a MANDATORY processing
requirement that must be met in full, an OPTIONAL requirement on top of it that
may be truncated, and a weight. The unexecuted part of the optional requirement
is the task's error. One machine, preemption allowed at no cost. Two classical
objectives:

- total weighted error ``sum_j w_j e_j`` (Shih, Liu, Chung 1991);
- maximum weighted error ``max_j w_j e_j`` (Shih, Liu 1995), and the
  lexicographic combination -- minimum total error first, smallest maximum
  error among the schedules achieving it.

Why a linear program is the right shape, and why it is exact. Cut the timeline
at every release and every deadline, giving intervals in which the set of
available tasks does not change. Let ``x[j][k]`` be the time task ``j`` spends
in interval ``k``, allowed only if the whole interval lies inside
``[release_j, deadline_j]``. Then

    sum_k x[j][k] <= mandatory_j + optional_j     (cannot overrun)
    sum_k x[j][k] >= mandatory_j                  (mandatory must complete)
    sum_j x[j][k] <= length of interval k         (one machine)

The last family is exactly Horn's feasibility condition for preemptive
single-machine scheduling with release dates and deadlines, so **every**
LP-feasible ``x`` corresponds to a real schedule: inside an interval all the
tasks assigned to it are available for its whole length, so their shares can be
laid out back to back in any order. :meth:`ImpreciseComputationScheduler.solve`
emits that schedule, and :func:`verify_schedule` checks it independently.
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

_TOL = 1e-9


@dataclass
class ImpreciseTask:
    """
    One imprecise-computation task.

    :param release: earliest time the task may run.
    :param deadline: time by which everything that is going to run must be done.
    :param mandatory: processing requirement that MUST be met in full.
    :param optional: further processing that may be truncated; whatever is left
        unexecuted is the task's error.
    :param weight: relative importance; scales this task's error.
    :param name: optional label, carried through to the result.
    """

    release: float
    deadline: float
    mandatory: float = 0.0
    optional: float = 0.0
    weight: float = 1.0
    name: str | None = None

    def __post_init__(self):
        if self.deadline <= self.release:
            raise ValueError(f"deadline {self.deadline} must exceed release {self.release}")
        if self.mandatory < 0 or self.optional < 0:
            raise ValueError(f"mandatory and optional must be non-negative, got {self.mandatory}, {self.optional}")
        if self.weight <= 0:
            raise ValueError(f"weight must be positive, got {self.weight}")

    @property
    def total(self) -> float:
        """Processing requirement if the optional part is run in full."""
        return self.mandatory + self.optional


@dataclass
class ImpreciseScheduleResult:
    """Outcome of an exact solve."""

    feasible: bool
    total_error: float = 0.0
    max_error: float = 0.0
    executed: list[float] = field(default_factory=list)  # processing time given to each task
    errors: list[float] = field(default_factory=list)  # unexecuted optional part of each task
    schedule: list[tuple[float, float, int]] = field(default_factory=list)  # (start, end, task index)
    duration: float = 0.0


def verify_schedule(tasks: list[ImpreciseTask], schedule: list[tuple[float, float, int]], tol: float = 1e-7):
    """
    Check a schedule against the model, independently of how it was produced.

    Raises ``ValueError`` on the first violation: overlapping pieces, work
    outside a task's window, a task running longer than its total requirement,
    or a mandatory part left unfinished.

    :return: per-task executed processing time.
    """
    ordered = sorted(schedule)
    previous_end = -np.inf
    executed = [0.0] * len(tasks)
    for start, end, index in ordered:
        if end < start - tol:
            raise ValueError(f"piece ({start}, {end}) ends before it starts")
        if start < previous_end - tol:
            raise ValueError(f"piece starting at {start} overlaps the previous one ending at {previous_end}")
        task = tasks[index]
        if start < task.release - tol or end > task.deadline + tol:
            raise ValueError(f"task {index} runs in ({start}, {end}), outside [{task.release}, {task.deadline}]")
        executed[index] += end - start
        previous_end = end
    for index, (task, done) in enumerate(zip(tasks, executed)):
        if done > task.total + tol:
            raise ValueError(f"task {index} runs {done}, more than its total requirement {task.total}")
        if done < task.mandatory - tol:
            raise ValueError(f"task {index} runs {done}, less than its mandatory part {task.mandatory}")
    return executed


class ImpreciseComputationScheduler:
    """
    Exact offline scheduler for imprecise computations on one preemptive machine.

    :param tolerance: slack used when deciding feasibility and when trimming
        numerically negligible schedule pieces.
    """

    def __init__(self, tolerance: float = _TOL):
        self.tolerance = tolerance
        self.tasks: list[ImpreciseTask] = []
        self._intervals: list[tuple[float, float]] | None = None
        self._allowed: np.ndarray | None = None

    def set_tasks(self, tasks: list[ImpreciseTask]):
        """Set the task set. Tasks may be given in any order."""
        if not tasks:
            raise ValueError("at least one task is required")
        self.tasks = list(tasks)
        self._intervals = None
        self._allowed = None
        return self

    # ---------------------------------------------------------------- internals
    def _decompose(self):
        """Cut the timeline at every release and deadline; cache which task may run where."""
        if self._intervals is not None:
            return self._intervals, self._allowed
        points = sorted({t.release for t in self.tasks} | {t.deadline for t in self.tasks})
        self._intervals = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
        self._allowed = np.array(
            [[t.release <= a and b <= t.deadline for (a, b) in self._intervals] for t in self.tasks],
            dtype=bool,
        )
        return self._intervals, self._allowed

    def _base_constraints(self, extra_columns: int = 0):
        """Rows shared by every objective: per-task bounds and one-machine capacity."""
        intervals, allowed = self._decompose()
        n, k_count = len(self.tasks), len(intervals)
        width = n * k_count + extra_columns

        def index(job, slot):
            return job * k_count + slot

        a_ub, b_ub = [], []
        for job, task in enumerate(self.tasks):
            row = np.zeros(width)
            row[[index(job, s) for s in range(k_count) if allowed[job, s]]] = 1.0
            a_ub.append(row)
            b_ub.append(task.total)

            row = np.zeros(width)
            row[[index(job, s) for s in range(k_count) if allowed[job, s]]] = -1.0
            a_ub.append(row)
            b_ub.append(-task.mandatory)

        for slot, (start, end) in enumerate(intervals):
            row = np.zeros(width)
            row[[index(job, slot) for job in range(n) if allowed[job, slot]]] = 1.0
            a_ub.append(row)
            b_ub.append(end - start)

        bounds = [(0.0, None) if allowed[job, slot] else (0.0, 0.0) for job in range(n) for slot in range(k_count)]
        bounds.extend([(0.0, None)] * extra_columns)
        return np.array(a_ub), np.array(b_ub), bounds, n, k_count

    def _build_schedule(self, x: np.ndarray) -> list[tuple[float, float, int]]:
        """
        Lay the LP solution out in time.

        Inside an interval every task assigned to it is available for the whole
        length, so the shares can simply be placed back to back. Consecutive
        pieces of the same task are merged so the output reads as a schedule
        rather than as the LP's internal bookkeeping.
        """
        intervals, _ = self._decompose()
        pieces: list[tuple[float, float, int]] = []
        for slot, (start, _end) in enumerate(intervals):
            cursor = start
            for job in range(len(self.tasks)):
                amount = float(x[job, slot])
                if amount <= self.tolerance:
                    continue
                pieces.append((cursor, cursor + amount, job))
                cursor += amount
        merged: list[tuple[float, float, int]] = []
        for start, end, job in pieces:
            if merged and merged[-1][2] == job and abs(merged[-1][1] - start) <= self.tolerance:
                merged[-1] = (merged[-1][0], end, job)
            else:
                merged.append((start, end, job))
        return merged

    # ----------------------------------------------------------------- results
    def is_feasible(self) -> bool:
        """Can every mandatory part be completed within its window?"""
        a_ub, b_ub, bounds, n, k_count = self._base_constraints()
        res = linprog(np.zeros(n * k_count), A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        return bool(res.success)

    def solve(self, objective: str = "total") -> ImpreciseScheduleResult:
        """
        Solve exactly and return the optimum together with a schedule attaining it.

        :param objective:
            ``"total"`` minimises the total weighted error (Shih, Liu & Chung
            1991); ``"max"`` minimises the maximum weighted error (Shih & Liu
            1995); ``"lexicographic"`` minimises the total error first and then,
            among the schedules achieving it, the maximum weighted error -- the
            "doubly weighted" problem of the same literature.
        """
        if objective not in ("total", "max", "lexicographic"):
            raise ValueError(f"objective must be 'total', 'max' or 'lexicographic', got {objective!r}")
        start_time = time.process_time()

        if objective == "total":
            result = self._solve_total()
        elif objective == "max":
            result = self._solve_max()
        else:
            result = self._solve_lexicographic()
        result.duration = time.process_time() - start_time
        return result

    def _finish(self, x: np.ndarray) -> ImpreciseScheduleResult:
        executed = x.sum(axis=1)
        errors = [max(task.total - done, 0.0) for task, done in zip(self.tasks, executed)]
        weighted = [task.weight * e for task, e in zip(self.tasks, errors)]
        return ImpreciseScheduleResult(
            feasible=True,
            total_error=float(sum(weighted)),
            max_error=float(max(weighted)) if weighted else 0.0,
            executed=[float(v) for v in executed],
            errors=[float(v) for v in errors],
            schedule=self._build_schedule(x),
        )

    def _solve_total(self) -> ImpreciseScheduleResult:
        """Minimising total weighted error is maximising weighted executed work."""
        intervals, allowed = self._decompose()
        a_ub, b_ub, bounds, n, k_count = self._base_constraints()
        cost = np.zeros(n * k_count)
        for job, task in enumerate(self.tasks):
            for slot in range(k_count):
                if allowed[job, slot]:
                    cost[job * k_count + slot] = -task.weight
        res = linprog(cost, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            return ImpreciseScheduleResult(feasible=False)
        _ = intervals
        return self._finish(res.x.reshape(n, k_count))

    def _solve_max(self, total_cap: float | None = None) -> ImpreciseScheduleResult:
        """
        Minimise the largest weighted error, optionally subject to a cap on the
        total one (which is how the lexicographic variant is obtained).
        """
        intervals, allowed = self._decompose()
        a_ub, b_ub, bounds, n, k_count = self._base_constraints(extra_columns=1)
        peak = n * k_count  # the extra column is the epigraph variable

        rows, rhs = list(a_ub), list(b_ub)
        for job, task in enumerate(self.tasks):
            # w_j (total_j - executed_j) <= peak
            row = np.zeros(n * k_count + 1)
            for slot in range(k_count):
                if allowed[job, slot]:
                    row[job * k_count + slot] = -task.weight
            row[peak] = -1.0
            rows.append(row)
            rhs.append(-task.weight * task.total)
        if total_cap is not None:
            row = np.zeros(n * k_count + 1)
            for job, task in enumerate(self.tasks):
                for slot in range(k_count):
                    if allowed[job, slot]:
                        row[job * k_count + slot] = -task.weight
            rows.append(row)
            rhs.append(total_cap - sum(t.weight * t.total for t in self.tasks))

        cost = np.zeros(n * k_count + 1)
        cost[peak] = 1.0
        res = linprog(cost, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds, method="highs")
        if not res.success:
            return ImpreciseScheduleResult(feasible=False)
        _ = intervals
        return self._finish(res.x[:-1].reshape(n, k_count))

    def _solve_lexicographic(self) -> ImpreciseScheduleResult:
        """Minimum total error first; smallest maximum error among those."""
        best_total = self._solve_total()
        if not best_total.feasible:
            return best_total
        refined = self._solve_max(total_cap=best_total.total_error + self.tolerance)
        if not refined.feasible:
            return best_total
        return refined

    def __repr__(self) -> str:
        return f"{type(self).__name__}(tasks={len(self.tasks)})"
