"""
Discrete-event simulator for ONLINE imprecise-computation scheduling, and the
bridge to the exact offline optimum in
``most_queue.theory.imprecise.offline``.

Each arriving job carries a mandatory requirement that must be completed by its
deadline, an optional requirement on top of it that may be truncated, and a
weight. Whatever of the optional part is left unexecuted when the deadline
arrives is the job's error. The server is single and preemption is free, as in
the model of Shih, Liu, Chung & Gillies (1989).

Why both halves are here. The offline problem is solved exactly, so the online
policies can be scored against the best any scheduler could have done on the
very same realisation -- not against each other and not against an asymptotic
bound. :func:`compare_with_offline` does exactly that, and the ratio it returns
is the thing worth reporting about an online rule.

The two policies implemented are deliberately simple and are NOT the
literature's optimal online algorithms (Shih & Liu, RTSS 1992, give those). They
are the two obvious operating points -- finish everything mandatory before
polishing anything, or polish as you go.

``full_edf`` is included as a COUNTEREXAMPLE, not as a recommendation: because
it does not give mandatory work priority, it can and does miss mandatory
deadlines, which the model treats as a hard failure. Its optional-error figure
then looks better than the offline optimum's, for the simple reason that it is
not solving the same problem -- it bought the lower error by skipping work it
was not allowed to skip. :func:`compare_with_offline` refuses to report a ratio
in that case rather than printing a number that flatters a broken policy. This
is also, concretely, why every policy in the literature serves mandatory parts
first.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from most_queue.random.utils.create import create_distribution
from most_queue.sim.base_core import BaseSimulationCore
from most_queue.theory.imprecise.offline import ImpreciseComputationScheduler, ImpreciseTask

POLICIES = ("mandatory_first", "full_edf")
_EPS = 1e-12


@dataclass
class ImpreciseJob:
    """One generated job, as the simulator and the offline solver both see it."""

    arrival: float
    deadline: float
    mandatory: float
    optional: float
    weight: float = 1.0


@dataclass
class ImpreciseSimResults:
    """Outcome of an online run."""

    total_error: float = 0.0  # sum of weighted unexecuted optional work
    mean_error: float = 0.0  # per job
    error_fraction: float = 0.0  # jobs that lost ANY optional work
    mandatory_miss_fraction: float = 0.0  # jobs whose mandatory part missed its deadline
    utilization: float = 0.0
    jobs: int = 0
    duration: float = 0.0
    per_job_error: list[float] = field(default_factory=list)


class ImpreciseComputationSim(BaseSimulationCore):
    """
    Single server, Poisson arrivals, online imprecise-computation scheduling.

    :param policy: ``"mandatory_first"`` runs every pending mandatory part
        (earliest deadline first) before touching any optional work;
        ``"full_edf"`` serves whole jobs in earliest-deadline-first order,
        truncating the optional part when the deadline arrives.
    :param seed: RNG seed.
    """

    def __init__(self, policy: str = "mandatory_first", seed: int | None = None):
        super().__init__(seed=seed)
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        self.policy = policy
        self.source = None
        self.mandatory_dist = None
        self.optional_dist = None
        self.deadline_dist = None
        self.weight = 1.0

    def set_sources(self, params, kendall_notation: str = "M"):
        """Interarrival distribution (see ``create_distribution``)."""
        self.source = create_distribution(params, kendall_notation, self.generator)

    def set_servers(self, mandatory, optional, kendall_notation: str = "M"):
        """
        :param mandatory: spec of the mandatory processing requirement.
        :param optional: spec of the optional requirement on top of it.
        :param kendall_notation: distribution family for both.
        """
        self.mandatory_dist = create_distribution(mandatory, kendall_notation, self.generator)
        self.optional_dist = create_distribution(optional, kendall_notation, self.generator)

    def set_deadlines(self, params, kendall_notation: str = "M", weight: float = 1.0):
        """Relative-deadline distribution and the (constant) job weight."""
        self.deadline_dist = create_distribution(params, kendall_notation, self.generator)
        self.weight = float(weight)

    def generate_jobs(self, count: int) -> list[ImpreciseJob]:
        """Draw a batch of jobs. The same batch can be fed to the offline solver."""
        if self.source is None or self.mandatory_dist is None or self.deadline_dist is None:
            raise ValueError("set_sources, set_servers and set_deadlines must be called first")
        jobs, clock = [], 0.0
        for _ in range(count):
            clock += self.source.generate()
            mandatory = max(self.mandatory_dist.generate(), 0.0)
            optional = max(self.optional_dist.generate(), 0.0)
            relative = max(self.deadline_dist.generate(), mandatory + _EPS)
            jobs.append(ImpreciseJob(clock, clock + relative, mandatory, optional, self.weight))
        return jobs

    def run_on(self, jobs: list[ImpreciseJob]) -> ImpreciseSimResults:
        """
        Run the online policy on a given batch -- the form used when the same
        batch is also solved offline.
        """
        start = time.process_time()
        pending = sorted(range(len(jobs)), key=lambda i: jobs[i].arrival)
        mandatory_left = [j.mandatory for j in jobs]
        optional_left = [j.optional for j in jobs]
        active: list[int] = []
        finished = [False] * len(jobs)
        clock = 0.0
        busy = 0.0
        next_index = 0

        while next_index < len(pending) or active:
            if not active:
                clock = max(clock, jobs[pending[next_index]].arrival)

            while next_index < len(pending) and jobs[pending[next_index]].arrival <= clock + _EPS:
                active.append(pending[next_index])
                next_index += 1

            chosen = self._choose(active, jobs, mandatory_left, optional_left)
            horizon = [jobs[i].deadline for i in active]
            if next_index < len(pending):
                horizon.append(jobs[pending[next_index]].arrival)
            if chosen is not None:
                remaining = mandatory_left[chosen] if mandatory_left[chosen] > _EPS else optional_left[chosen]
                horizon.append(clock + remaining)
            step = min(h for h in horizon if h > clock + _EPS) if any(h > clock + _EPS for h in horizon) else None
            if step is None:
                break

            if chosen is not None:
                worked = step - clock
                busy += worked
                if mandatory_left[chosen] > _EPS:
                    mandatory_left[chosen] = max(mandatory_left[chosen] - worked, 0.0)
                else:
                    optional_left[chosen] = max(optional_left[chosen] - worked, 0.0)
            clock = step

            for i in list(active):
                done = mandatory_left[i] <= _EPS and optional_left[i] <= _EPS
                if done or jobs[i].deadline <= clock + _EPS:
                    finished[i] = True
                    active.remove(i)

        errors = [jobs[i].weight * optional_left[i] for i in range(len(jobs))]
        misses = sum(1 for i in range(len(jobs)) if mandatory_left[i] > 1e-7)
        total_span = max((j.deadline for j in jobs), default=1.0)
        _ = finished
        result = ImpreciseSimResults(
            total_error=float(sum(errors)),
            mean_error=float(np.mean(errors)) if errors else 0.0,
            error_fraction=float(np.mean([e > 1e-9 for e in errors])) if errors else 0.0,
            mandatory_miss_fraction=misses / len(jobs) if jobs else 0.0,
            utilization=busy / total_span if total_span > 0 else 0.0,
            jobs=len(jobs),
            per_job_error=[float(e) for e in errors],
        )
        result.duration = time.process_time() - start
        return result

    def _choose(self, active, jobs, mandatory_left, optional_left):
        """Which job the policy runs now, or ``None`` if there is nothing to do."""
        if not active:
            return None
        if self.policy == "mandatory_first":
            needy = [i for i in active if mandatory_left[i] > _EPS]
            pool = needy or [i for i in active if optional_left[i] > _EPS]
        else:  # full_edf: whole jobs in deadline order, mandatory part of each first
            pool = [i for i in active if mandatory_left[i] > _EPS or optional_left[i] > _EPS]
        if not pool:
            return None
        return min(pool, key=lambda i: (jobs[i].deadline, jobs[i].arrival))

    def run(self, total_jobs: int) -> ImpreciseSimResults:
        """Generate ``total_jobs`` jobs and run the online policy on them."""
        return self.run_on(self.generate_jobs(total_jobs))


def to_offline_tasks(jobs: list[ImpreciseJob]) -> list[ImpreciseTask]:
    """Convert a generated batch into the offline solver's task list."""
    return [
        ImpreciseTask(
            release=j.arrival,
            deadline=j.deadline,
            mandatory=j.mandatory,
            optional=j.optional,
            weight=j.weight,
        )
        for j in jobs
    ]


def compare_with_offline(sim: ImpreciseComputationSim, jobs: list[ImpreciseJob]) -> dict:
    """
    Score an online policy against the exact offline optimum on the SAME jobs.

    The offline problem has no horizon effects to worry about -- every deadline
    is attached to its own job -- so the comparison is clean: the two numbers
    describe the same finite instance.

    A ratio is reported only when it means something. Two situations make it
    meaningless, and both are flagged rather than papered over:

    - the mandatory set is not schedulable at all, so there is no optimum to
      compare against (``feasible`` is False);
    - the ONLINE run missed mandatory deadlines. The offline optimum is
      constrained to complete every mandatory part, so a policy that skips some
      is not solving the same problem and can report a lower optional error
      while being strictly worse. ``online_mandatory_misses`` says how many.

    :return: ``{"online", "offline", "ratio", "feasible",
        "online_mandatory_misses", "comparable"}``.
    """
    online = sim.run_on(jobs)
    misses = int(round(online.mandatory_miss_fraction * len(jobs)))
    scheduler = ImpreciseComputationScheduler().set_tasks(to_offline_tasks(jobs))
    offline = scheduler.solve("total")
    comparable = offline.feasible and misses == 0
    ratio = None
    if comparable and offline.total_error > 1e-12:
        ratio = online.total_error / offline.total_error
    return {
        "online": online.total_error,
        "offline": offline.total_error if offline.feasible else None,
        "ratio": ratio,
        "feasible": offline.feasible,
        "online_mandatory_misses": misses,
        "comparable": comparable,
    }
