"""
Discrete-event simulator for a batch of jobs whose value decays during
service, and the bridge to the exact optimal control in
:mod:`most_queue.theory.decaying_value.service_rate_control`.

The model is that of Master N. & Bambos N., "Service rate control for jobs with
decaying value", Proc. 2015 American Control Conference, pp. 3255-3260,
doi:10.1109/ACC.2015.7171834. **The model is theirs**; this module simply plays
it out slot by slot.

Why it is here. The dynamic program returns an expected total cost, and a
recursion that produces a number is easy to get subtly wrong in a way that no
amount of internal consistency will reveal. Running the system under the same
policy and averaging what it actually costs is an independent check of both the
cost-to-go and the dynamics, so :func:`compare_with_optimal` is the test that
matters about the solver.

It also makes the practical question answerable: the optimal policy is worth
having only to the extent it beats the obvious alternatives, so constant-rate
and myopic baselines are scored against it on the same model.
"""

import time
from dataclasses import dataclass

import numpy as np

from most_queue.theory.decaying_value.service_rate_control import DecayingValueRateControl

_EPS = 1e-12


@dataclass
class DecayingValueSimResults:
    """Outcome of a batch of simulated runs."""

    mean_cost: float = 0.0  # realised total cost per run
    cost_error: float = 0.0  # standard error of that mean
    mean_reward: float = 0.0  # reward collected per run
    mean_slots: float = 0.0  # slots until the batch is cleared
    completed_fraction: float = 0.0  # jobs finished before their value ran out
    ejected_fraction: float = 0.0  # jobs whose value hit zero
    replications: int = 0
    duration: float = 0.0


class DecayingValueSim:
    """
    Slot-by-slot simulation of the decaying-value batch under a given policy.

    :param control: the configured problem, which supplies the batch size, the
        decay horizon and all three cost functions. Taking it whole rather than
        re-specifying the model keeps the simulator and the solver provably on
        the same instance.
    :param policy: an array of completion probabilities indexed by
        ``[backlog, value]``. Defaults to the optimal policy.
    :param seed: RNG seed.

    Usage::

        sim = DecayingValueSim(control, seed=0)
        sim.run(replications=20000).mean_cost     # compare with result.total_cost
    """

    def __init__(
        self,
        control: DecayingValueRateControl,
        policy: np.ndarray | None = None,
        seed: int | None = None,
    ):
        self.control = control
        self.policy = control.solve().policy if policy is None else np.asarray(policy, dtype=float)
        expected = (control.num_jobs + 1, control.initial_value + 1)
        if self.policy.shape != expected:
            raise ValueError(f"policy must have shape {expected}, got {self.policy.shape}")
        if np.any(self.policy[1:, 1:] < -_EPS) or np.any(self.policy[1:, 1:] > 1.0 + _EPS):
            raise ValueError("every policy entry must be a probability in [0, 1]")
        self.generator = np.random.default_rng(seed)
        self._service_cost = {rate: float(control.service_cost(rate)) for rate in np.unique(self.policy[1:, 1:])}

    def _run_once(self) -> tuple[float, float, int, int, int]:
        """One realisation: total cost, reward, slots, completions, ejections."""
        control = self.control
        top = control.initial_value
        backlog, value = control.num_jobs, top
        cost = reward = 0.0
        slots = completed = ejected = 0

        while backlog > 0:
            rate = float(self.policy[backlog, value])
            cost += float(control.holding_cost(backlog)) + self._service_cost[rate]
            slots += 1
            if self.generator.random() < rate:
                gained = float(control.reward(value))
                reward += gained
                cost -= gained
                completed += 1
                backlog -= 1
                value = top
            elif value > 1:
                value -= 1
            else:
                ejected += 1  # value ran out: the job leaves for nothing
                backlog -= 1
                value = top
        return cost, reward, slots, completed, ejected

    def run(self, replications: int = 10_000) -> DecayingValueSimResults:
        """Average over independent runs of the whole batch."""
        if replications < 1:
            raise ValueError(f"replications must be at least 1, got {replications}")
        start = time.process_time()
        costs = np.empty(replications)
        rewards = np.empty(replications)
        slots = np.empty(replications)
        completed = np.empty(replications)
        ejected = np.empty(replications)
        for i in range(replications):
            costs[i], rewards[i], slots[i], completed[i], ejected[i] = self._run_once()

        jobs = float(self.control.num_jobs)
        return DecayingValueSimResults(
            mean_cost=float(costs.mean()),
            cost_error=float(costs.std(ddof=1) / np.sqrt(replications)) if replications > 1 else 0.0,
            mean_reward=float(rewards.mean()),
            mean_slots=float(slots.mean()),
            completed_fraction=float(completed.mean() / jobs),
            ejected_fraction=float(ejected.mean() / jobs),
            replications=replications,
            duration=time.process_time() - start,
        )

    def __repr__(self) -> str:
        return f"DecayingValueSim({self.control!r})"


def constant_rate_policy(control: DecayingValueRateControl, rate: float) -> np.ndarray:
    """
    The simplest possible rule: the same service rate in every state.

    :param rate: must be one of the admissible rates.
    """
    if not np.any(np.isclose(control.rates, rate)):
        raise ValueError(f"rate {rate} is not among the admissible rates {control.rates.tolist()}")
    policy = np.zeros((control.num_jobs + 1, control.initial_value + 1))
    policy[1:, 1:] = rate
    return policy


def compare_with_optimal(
    control: DecayingValueRateControl,
    policy: np.ndarray | None = None,
    replications: int = 20_000,
    seed: int | None = 0,
) -> dict:
    """
    Score a policy against the optimum, both exactly and by simulation.

    The exact figures come from the dynamic program -- the optimum from
    :meth:`~most_queue.theory.decaying_value.service_rate_control.DecayingValueRateControl.solve`
    and the policy's own cost from its
    :meth:`~most_queue.theory.decaying_value.service_rate_control.DecayingValueRateControl.evaluate`,
    which is the same acyclic sweep without the minimisation. The simulated
    figure is there to confirm the exact one rather than to replace it.

    :param policy: the rule to score; defaults to the myopic one.
    :return: the optimal and achieved expected costs, the simulated cost with
        its standard error, the excess cost of the policy, and how many
        standard errors separate simulation from theory.
    """
    optimal = control.solve()
    if policy is None:
        policy = control.myopic_policy()
    achieved = float(control.evaluate(policy)[control.num_jobs, control.initial_value])

    simulated = DecayingValueSim(control, policy=policy, seed=seed).run(replications)
    gap = simulated.mean_cost - achieved
    sigmas = abs(gap) / simulated.cost_error if simulated.cost_error > 0 else 0.0

    return {
        "optimal_cost": optimal.total_cost,
        "policy_cost": achieved,
        "excess_cost": achieved - optimal.total_cost,
        "simulated_cost": simulated.mean_cost,
        "simulated_error": simulated.cost_error,
        "sigmas": sigmas,
        "simulation": simulated,
    }
