"""
On-line value-based scheduling under overload, and the bridge to the exact
clairvoyant optimum in :mod:`most_queue.theory.value_scheduling.offline`.

Implementation of
    Buttazzo G.C., Spuri M., Sensini F., "Value vs. deadline scheduling in
    overload conditions", Proc. 16th IEEE Real-Time Systems Symposium (RTSS),
    Pisa, 1995, pp. 90-99, doi:10.1109/REAL.1995.495198.

THE MODEL, THE TWELVE ALGORITHMS AND THE EXPERIMENTAL DESIGN ARE THEIRS. What
is contributed here is an implementation, the paper's task-set generator, and
-- as the one genuinely new piece -- a comparison against the exact clairvoyant
optimum, which the paper does not have (see the module it lives in for why that
matters).

The model. Jobs arrive unannounced. Each carries a worst-case computation time,
an actual computation time no larger than it, a deadline and an importance
VALUE. A job that completes by its deadline banks its value in full; a job that
misses banks nothing, so running it past its deadline is pure waste and it is
abandoned the moment the deadline passes. One machine, preemption free. The
figure of merit is the Hit Value Ratio -- cumulative value banked over the
total value of the task set.

The twelve algorithms are a four-by-three grid: a priority assignment crossed
with a guarantee mechanism.

Priority assignments (which ready job runs now):

- ``edf``  earliest deadline first -- ignores value entirely;
- ``hvf``  highest value first -- ignores urgency entirely;
- ``hdf``  highest value density first, value over REMAINING worst-case time,
           so the priority of the running job rises as it executes;
- ``mix``  ``alpha * value - (1 - alpha) * deadline``, a linear blend; the
           paper tunes ``alpha`` and settles on 0.5.

Guarantee mechanisms (what happens when the ready set cannot all make it):

- ``plain``       nothing is rejected; overload is absorbed by missing
                  deadlines, which is where the domino effect lives -- EDF
                  keeps serving the most urgent job even when that job and
                  several behind it are already doomed;
- ``guaranteed``  an acceptance test runs at every arrival and the NEWLY
                  ARRIVED job is rejected if the test predicts an overflow.
                  Simple, and indifferent to how valuable the newcomer is;
- ``robust``      the same test, but the rejected job is the LEAST VALUABLE one
                  whose removal clears the overload, and rejected jobs are
                  parked rather than discarded: whenever a job finishes early
                  -- which it usually does, the actual time being below the
                  worst case -- a parked job is reclaimed if it now fits.

The acceptance test is exact for ``edf`` and only as good as the ordering for
the others. It lays out the ready jobs in the order the algorithm will actually
run them, using worst-case remaining times, and asks whether any of them
finishes late. Since all ready jobs are available immediately and future
arrivals are unknowable to an on-line scheduler, EDF ordering makes this test
necessary and sufficient, by Horn's optimality of EDF. For a value-based
ordering it answers the narrower question "will the schedule I am about to
produce miss a deadline", which is what the paper's guarantee routine does.
"""

import time
from dataclasses import dataclass

import numpy as np

from most_queue.theory.value_scheduling.offline import (
    ClairvoyantValueScheduler,
    ValueTask,
)

PRIORITIES = ("edf", "hvf", "hdf", "mix")
GUARANTEES = ("plain", "guaranteed", "robust")
VALUE_MODES = ("random", "linear")
_EPS = 1e-9


@dataclass
class ValueJob:
    """
    One generated job, as both the simulator and the off-line solver see it.

    ``worst`` is what the on-line algorithms are allowed to reason about;
    ``actual`` is what the machine really spends and what the clairvoyant
    optimum is allowed to know.
    """

    arrival: float
    deadline: float  # absolute
    worst: float
    actual: float
    value: float
    stream: int = -1

    def __post_init__(self):
        if self.actual > self.worst + _EPS:
            raise ValueError(f"actual time {self.actual} exceeds the worst case {self.worst}")


@dataclass
class ValueSchedulingResults:
    """Outcome of one on-line run."""

    hit_value_ratio: float = 0.0  # the paper's HVR
    cumulative_value: float = 0.0  # value actually banked
    total_value: float = 0.0  # value of the whole task set
    completed: int = 0  # finished by their deadline
    missed: int = 0  # admitted, then abandoned at the deadline
    rejected: int = 0  # refused by the acceptance test and never reclaimed
    jobs: int = 0
    busy_time: float = 0.0  # machine time spent on work of any kind
    wasted_time: float = 0.0  # of that, spent on jobs that banked nothing
    reclaimed: int = 0  # parked jobs brought back (robust class only)
    duration: float = 0.0


class ValueSchedulingSim:
    """
    Single preemptive machine, on-line value-based scheduling under overload.

    :param priority: one of :data:`PRIORITIES`.
    :param guarantee: one of :data:`GUARANTEES`.
    :param alpha: the blend weight used by the ``mix`` assignment; the paper's
        tuning experiment picks 0.5.
    :param seed: RNG seed for :meth:`generate_task_set`.

    The naming in the paper is the product of the two choices -- ``edf`` with
    ``guaranteed`` is GEDF, with ``robust`` is REDF, and so on;
    :attr:`algorithm` spells that out.

    Usage::

        sim = ValueSchedulingSim(priority="hdf", guarantee="robust", seed=1)
        jobs = sim.generate_task_set(nominal_load=3.0)
        sim.run_on(jobs).hit_value_ratio
    """

    def __init__(
        self,
        priority: str = "edf",
        guarantee: str = "plain",
        alpha: float = 0.5,
        seed: int | None = None,
    ):
        if priority not in PRIORITIES:
            raise ValueError(f"priority must be one of {PRIORITIES}, got {priority!r}")
        if guarantee not in GUARANTEES:
            raise ValueError(f"guarantee must be one of {GUARANTEES}, got {guarantee!r}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
        self.priority = priority
        self.guarantee = guarantee
        self.alpha = float(alpha)
        self.generator = np.random.default_rng(seed)

    @property
    def algorithm(self) -> str:
        """The paper's name for this combination, e.g. ``"REDF"``."""
        prefix = {"plain": "", "guaranteed": "G", "robust": "R"}[self.guarantee]
        return prefix + self.priority.upper()

    def generate_task_set(
        self,
        nominal_load: float,
        num_streams: int = 100,
        horizon: float = 300_000.0,
        value_mode: str = "random",
        unused_ratio: float | None = None,
    ) -> list[ValueJob]:
        """
        Draw a task set exactly as the paper's experiments do.

        Each of ``num_streams`` streams draws its own worst-case computation
        time ``C`` uniformly from 50 to 350 and then arrives as a Poisson
        process of mean interarrival ``num_streams * C / nominal_load``, so
        every stream contributes the same share of the load and the nominal
        load -- computed on worst-case times -- comes out at ``nominal_load``.
        Laxity is uniform on 150 to 1850 and the relative deadline is
        ``C + laxity``.

        :param nominal_load: load measured with worst-case times; the paper
            sweeps 0.5 to 3.5. The ACTUAL load is about half of it, since the
            actual execution time averages half the worst case, so the system
            only becomes truly overloaded above a nominal load of two.
        :param value_mode: ``"random"`` draws the value uniformly from 150 to
            1850, independent of everything else; ``"linear"`` makes it equal
            to the relative deadline. The paper calls these the random and the
            linear task set, and notes the linear one is the hard case because
            the valuable jobs are also the ones most likely to miss.
        :param unused_ratio: the paper's unused computation time ratio
            ``beta = 1 - actual / worst``. ``None``, the default, draws the
            actual time uniformly between zero and the worst case
            (``beta = 0.5`` on average); a number fixes it at
            ``(1 - beta) * worst``, which is how the paper's last experiment
            sweeps the actual load without touching the nominal one.
        """
        if nominal_load <= 0.0:
            raise ValueError(f"nominal_load must be positive, got {nominal_load}")
        if num_streams < 1:
            raise ValueError(f"num_streams must be at least 1, got {num_streams}")
        if value_mode not in VALUE_MODES:
            raise ValueError(f"value_mode must be one of {VALUE_MODES}, got {value_mode!r}")
        if unused_ratio is not None and not 0.0 <= unused_ratio < 1.0:
            raise ValueError(f"unused_ratio must lie in [0, 1), got {unused_ratio}")

        jobs: list[ValueJob] = []
        for stream in range(num_streams):
            worst = self.generator.uniform(50.0, 350.0)
            mean_gap = num_streams * worst / nominal_load
            clock = self.generator.exponential(mean_gap)
            while clock < horizon:
                laxity = self.generator.uniform(150.0, 1850.0)
                relative = worst + laxity
                if unused_ratio is None:
                    actual = self.generator.uniform(0.0, worst)
                else:
                    actual = (1.0 - unused_ratio) * worst
                jobs.append(
                    ValueJob(
                        arrival=clock,
                        deadline=clock + relative,
                        worst=worst,
                        actual=actual,
                        value=relative if value_mode == "linear" else self.generator.uniform(150.0, 1850.0),
                        stream=stream,
                    )
                )
                clock += self.generator.exponential(mean_gap)
        if not jobs:
            raise ValueError("no jobs were generated; increase horizon or nominal_load")
        return jobs

    def _priority_key(self, index: int, jobs, remaining_worst) -> float:
        """Sort key for the ready set; smaller runs first."""
        job = jobs[index]
        if self.priority == "edf":
            return job.deadline
        if self.priority == "hvf":
            return -job.value
        if self.priority == "hdf":
            return -job.value / max(remaining_worst[index], _EPS)
        return -(self.alpha * job.value - (1.0 - self.alpha) * job.deadline)

    def _overloaded(self, ready, jobs, remaining_worst, clock: float) -> bool:
        """
        The acceptance test: lay the ready set out in the order this algorithm
        will run it, charging worst-case remaining times, and see whether
        anything finishes late.
        """
        order = sorted(ready, key=lambda i: self._priority_key(i, jobs, remaining_worst))
        finish = clock
        for i in order:
            finish += remaining_worst[i]
            if finish > jobs[i].deadline + _EPS:
                return True
        return False

    def _reclaim(self, queues, jobs, remaining_worst, clock: float) -> int:
        """
        Take parked jobs back after a completion, most valuable first, for as
        many as still fit. Only the robust class parks anything.

        A completion frees time -- usually more than the acceptance test
        assumed, since the actual time was below the worst case -- so a job
        rejected earlier may now be feasible.

        :param queues: the ``(ready, parked)`` lists, modified in place.
        """
        ready, parked = queues
        if self.guarantee != "robust" or not parked:
            return 0
        taken = 0
        for candidate in sorted(parked, key=lambda i: -jobs[i].value):
            if jobs[candidate].deadline <= clock + _EPS:
                continue
            if self._overloaded(ready + [candidate], jobs, remaining_worst, clock):
                continue
            ready.append(candidate)
            parked.remove(candidate)
            taken += 1
        return taken

    def run_on(self, jobs: list[ValueJob]) -> ValueSchedulingResults:
        """
        Run the algorithm on a given task set, leaving it untouched.

        The jobs are treated as read-only, so the same set can be handed to
        every algorithm and to the off-line solver and they all see the same
        realisation.
        """
        if not jobs:
            raise ValueError("task set is empty")
        start = time.process_time()

        arrival_order = sorted(range(len(jobs)), key=lambda i: jobs[i].arrival)
        remaining_actual = [job.actual for job in jobs]
        remaining_worst = [job.worst for job in jobs]
        ready: list[int] = []
        parked: list[int] = []  # rejected but still reclaimable (robust class)

        clock = 0.0
        banked = 0.0
        completed = missed = reclaimed = 0
        busy = useful = 0.0
        next_arrival = 0
        count = len(jobs)

        while next_arrival < count or ready:
            if not ready and next_arrival < count:
                clock = max(clock, jobs[arrival_order[next_arrival]].arrival)

            # Admit everything that has arrived by now, testing as we go.
            while next_arrival < count and jobs[arrival_order[next_arrival]].arrival <= clock + _EPS:
                index = arrival_order[next_arrival]
                next_arrival += 1
                ready.append(index)
                if self.guarantee == "plain":
                    continue
                if not self._overloaded(ready, jobs, remaining_worst, clock):
                    continue
                if self.guarantee == "guaranteed":
                    ready.remove(index)
                    parked.append(index)
                else:
                    # Shed the least valuable jobs until the overload clears.
                    while ready and self._overloaded(ready, jobs, remaining_worst, clock):
                        victim = min(ready, key=lambda i: jobs[i].value)
                        ready.remove(victim)
                        parked.append(victim)

            # Abandon anything whose deadline has passed: it can bank nothing.
            expired = [i for i in ready if jobs[i].deadline <= clock + _EPS]
            for i in expired:
                ready.remove(i)
                missed += 1
            parked = [i for i in parked if jobs[i].deadline > clock + _EPS]

            if not ready:
                if next_arrival >= count:
                    break
                clock = max(clock, jobs[arrival_order[next_arrival]].arrival)
                continue

            current = min(ready, key=lambda i: self._priority_key(i, jobs, remaining_worst))
            if remaining_actual[current] <= _EPS:
                # A job needing no time at all banks its value without taking
                # the machine; advancing the clock to the next event here would
                # idle the machine for nothing.
                ready.remove(current)
                banked += jobs[current].value
                completed += 1
                reclaimed += self._reclaim((ready, parked), jobs, remaining_worst, clock)
                continue
            # The running job keeps the machine until the next event, because
            # no priority rule here can be overtaken by a waiting job while it
            # runs: the static rules do not move, and under hdf the running
            # job's own density only rises.
            events = [clock + remaining_actual[current], jobs[current].deadline]
            events += [jobs[i].deadline for i in ready]
            if next_arrival < count:
                events.append(jobs[arrival_order[next_arrival]].arrival)
            step = min(event for event in events if event > clock + _EPS)

            worked = min(step - clock, remaining_actual[current])
            remaining_actual[current] -= worked
            remaining_worst[current] = max(remaining_worst[current] - worked, 0.0)
            busy += worked
            clock = step

            if remaining_actual[current] > _EPS:
                continue
            ready.remove(current)
            if jobs[current].deadline >= clock - _EPS:
                banked += jobs[current].value
                completed += 1
                useful += jobs[current].actual
            else:
                missed += 1

            reclaimed += self._reclaim((ready, parked), jobs, remaining_worst, clock)

        total = sum(job.value for job in jobs)
        return ValueSchedulingResults(
            hit_value_ratio=banked / total if total > 0 else 0.0,
            cumulative_value=banked,
            total_value=total,
            completed=completed,
            missed=missed,
            rejected=count - completed - missed,
            jobs=count,
            busy_time=busy,
            wasted_time=busy - useful,
            reclaimed=reclaimed,
            duration=time.process_time() - start,
        )

    def run(self, nominal_load: float, **kwargs) -> ValueSchedulingResults:
        """Generate a task set at the given nominal load and run on it."""
        return self.run_on(self.generate_task_set(nominal_load, **kwargs))

    def __repr__(self) -> str:
        return f"ValueSchedulingSim({self.algorithm}, alpha={self.alpha})"


def to_offline_tasks(jobs: list[ValueJob]) -> list[ValueTask]:
    """
    Recast a task set for the clairvoyant solver.

    The off-line optimum is charged the ACTUAL execution times -- it is allowed
    to know them, which is exactly the advantage being measured.
    """
    return [
        ValueTask(release=job.arrival, deadline=job.deadline, processing=job.actual, value=job.value) for job in jobs
    ]


def compare_with_clairvoyant(sim: ValueSchedulingSim, jobs: list[ValueJob], **kwargs) -> dict:
    """
    Score an on-line algorithm against the best any scheduler could have done
    on the same realisation.

    :param sim: the configured on-line algorithm.
    :param jobs: the task set, handed to both sides unchanged.
    :param kwargs: forwarded to :meth:`ClairvoyantValueScheduler.solve`.
    :return: the on-line and off-line results, the paper's HVR for each, and
        ``competitive_ratio`` -- banked value over the clairvoyant optimum.
        That ratio cannot exceed one, and a value above one would mean the two
        sides are not solving the same problem.
    """
    online = sim.run_on(jobs)
    offline = ClairvoyantValueScheduler().set_tasks(to_offline_tasks(jobs)).solve(**kwargs)
    ratio = online.cumulative_value / offline.optimal_value if offline.optimal_value > 0 else float("nan")
    return {
        "algorithm": sim.algorithm,
        "online": online,
        "offline": offline,
        "hit_value_ratio": online.hit_value_ratio,
        "clairvoyant_hit_value_ratio": offline.hit_value_ratio,
        "competitive_ratio": ratio,
    }
