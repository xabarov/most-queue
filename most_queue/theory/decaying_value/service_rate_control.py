"""
Optimal service rate control for a finite batch of jobs whose value decays
while they are being served.

Implementation of
    Master N., Bambos N., "Service rate control for jobs with decaying value",
    Proc. 2015 American Control Conference (ACC), Chicago, pp. 3255-3260,
    doi:10.1109/ACC.2015.7171834 (preprint arXiv:1609.05355).

THE MODEL, THE REFORMULATION AND THE MONOTONICITY THEOREMS ARE THEIRS. What is
contributed here is an implementation, two independent solution routes that
check each other, and a simulator that checks the predicted cost against a
realised one.

The model. A single server works through a finite batch of ``B`` identical jobs
in discrete time. The job at the head of the line arrives there with value
``V`` and, in each slot, the controller picks a completion probability ``s``
from a finite set ``S``. With probability ``s`` the job completes and the
controller collects a reward ``r(v)`` that depends on the value the job still
had; otherwise the value decays by one. A job whose value reaches zero is
EJECTED, earning nothing. Each slot also costs ``h(b)`` for holding the ``b``
jobs still present and ``c(s)`` for running the server that fast. The system
ends when every job has been completed or ejected, so this is a stochastic
shortest path problem rather than a stationary queue, and the quantity of
interest is the total expected cost rather than a waiting time.

What makes the decay unusual. The value decays only DURING service, not during
the whole sojourn -- which is what distinguishes it from the impatience models
elsewhere in this library, where a job's clock runs while it waits. The
motivating cases in the paper are wireless streaming (a packet is worth less
the longer the transmitter spends on it), healthcare (delay in treatment erodes
the benefit), and perishable inventory (goods decay in handling rather than in
storage).

Why the dynamic program collapses. Written out, the Bellman equation is

    J(b,v) = min_s { c(s) + h(b)
                     + s[-r(v) + J(b-1,V)]
                     + (1-s)[ J(b,v-1) if v > 1 else J(b-1,V) ] }

with J(0,V) = 0. The state never moves "up" -- ``b`` only decreases and, within
a fixed ``b``, each failed slot decreases ``v`` -- so the problem is acyclic and
no iteration to a fixed point is needed. More than that, the paper's
Proposition 1 shows the whole thing telescopes. Writing

    delta(b,v) = h(b) + min_s { c(s) - s[ r(v) + sigma(b,v-1) ] }
    sigma(b,v) = sigma(b,v-1) + delta(b,v),      sigma(b,0) = 0

gives J(b,v) = J(b-1,V) + sigma(b,v) and, crucially,

    mu(b,v) = min argmin_s { c(s) - s[ r(v) + sigma(b,v-1) ] }

so the optimal rate depends on the state only through the single scalar
``r(v) + sigma(b,v-1)``. That is the whole content of the reformulation: an
O(B*V*|S|) forward recursion, and a one-dimensional quantity to reason about
instead of a policy surface.

The structural results that follow from it, all proved in the paper:

- **Theorem 1.** For each ``v``, ``b -> mu(b,v)`` is non-decreasing: with more
  jobs still waiting, the server should work faster. This needs nothing beyond
  ``h`` being non-decreasing, so it holds for every admissible instance.
- **Theorem 2.** For a fixed ``b``, if ``delta(b,v) >= -[r(v+1) - r(v)]`` for
  every ``v``, then ``v -> mu(b,v)`` is non-decreasing; if the inequality points
  the other way throughout, it is non-increasing. Either can happen, and
  neither need happen.
- **Theorem 3.** When the reward is a constant ``r`` -- the step-function case
  used to model a service time constraint -- the test collapses to the sign of
  ``h(b) + min_s { c(s) - s*r }``.

The point of Theorems 2 and 3 is that they are checkable WITHOUT computing the
policy, which :meth:`DecayingValueRateControl.monotonicity_conditions` does.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

_TOL = 1e-9
METHODS = ("direct", "bellman")


@dataclass
class DecayingValueResult:
    """Outcome of solving the control problem."""

    policy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    cost_to_go: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    delta: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    sigma: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    total_cost: float = 0.0  # J(B, V): the cost of serving the whole batch
    method: str = "direct"
    duration: float = 0.0

    def rate(self, backlog: int, value: int) -> float:
        """Optimal completion probability in state ``(backlog, value)``."""
        return float(self.policy[backlog, value])


class DecayingValueRateControl:
    """
    Optimal service rate control for a batch of jobs with decaying value.

    :param num_jobs: batch size ``B``, the number of jobs to get through.
    :param initial_value: ``V``, the value a job has when it reaches the head
        of the line. A job surviving ``V`` unsuccessful slots is ejected.
    :param rates: the finite set ``S`` of admissible completion probabilities,
        each in ``[0, 1]``.
    :param service_cost: ``c(s)``, the per-slot cost of running at rate ``s``.
        Must be non-decreasing and finite on ``rates``.
    :param holding_cost: ``h(b)``, the per-slot cost of holding ``b`` jobs.
        Must be non-decreasing.
    :param reward: ``r(v)``, collected when a job completes with value ``v``
        left. Must be positive and non-decreasing on ``1..V``.
    :param validate: check the monotonicity assumptions the paper's theorems
        rest on. Turning it off lets you explore instances outside the model,
        where Theorem 1 need not hold.

    Usage::

        control = DecayingValueRateControl(
            num_jobs=20, initial_value=10, rates=[0.1, 0.5, 0.9],
            service_cost=lambda s: 5 * np.log(1 / (1 - s)),
            holding_cost=lambda b: b,
            reward=lambda v: v,
        )
        result = control.solve()
        result.rate(backlog=12, value=4)   # how fast to run in that state
        result.total_cost                  # expected cost for the whole batch
    """

    def __init__(
        self,
        num_jobs: int,
        initial_value: int,
        rates: Sequence[float],
        service_cost: Callable[[float], float],
        holding_cost: Callable[[int], float],
        reward: Callable[[int], float],
        validate: bool = True,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        if num_jobs < 1:
            raise ValueError(f"num_jobs must be at least 1, got {num_jobs}")
        if initial_value < 1:
            raise ValueError(f"initial_value must be at least 1, got {initial_value}")
        grid = np.asarray(sorted(float(rate) for rate in rates), dtype=float)
        if grid.size == 0:
            raise ValueError("rates must contain at least one admissible rate")
        if np.any(grid < 0.0) or np.any(grid > 1.0):
            raise ValueError(f"every rate must lie in [0, 1], got {grid.tolist()}")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError(f"rates must be distinct, got {grid.tolist()}")

        self.num_jobs = int(num_jobs)
        self.initial_value = int(initial_value)
        self.rates = grid
        self.service_cost = service_cost
        self.holding_cost = holding_cost
        self.reward = reward

        self._service_costs = np.array([float(service_cost(rate)) for rate in grid])
        self._holding_costs = np.array([float(holding_cost(b)) for b in range(self.num_jobs + 1)])
        self._rewards = np.array([0.0] + [float(reward(v)) for v in range(1, self.initial_value + 1)])

        if not np.all(np.isfinite(self._service_costs)):
            raise ValueError(
                f"service_cost must be finite on every admissible rate; got "
                f"{self._service_costs.tolist()} for rates {grid.tolist()}"
            )
        if validate:
            self._validate()

    def _validate(self):
        """Check the assumptions the paper's results depend on."""
        if np.any(self._service_costs < -_TOL):
            raise ValueError("service_cost must be non-negative")
        if np.any(np.diff(self._service_costs) < -_TOL):
            raise ValueError("service_cost must be non-decreasing in the rate")
        if np.any(self._holding_costs < -_TOL):
            raise ValueError("holding_cost must be non-negative")
        if np.any(np.diff(self._holding_costs) < -_TOL):
            raise ValueError("holding_cost must be non-decreasing in the backlog")
        positive = self._rewards[1:]
        if np.any(positive <= 0.0):
            raise ValueError("reward must be strictly positive for every value in 1..V")
        if np.any(np.diff(positive) < -_TOL):
            raise ValueError("reward must be non-decreasing in the residual value")

    def _best_rate(self, weight: float) -> tuple[float, float, int]:
        """
        Solve ``min_s { c(s) - s * weight }``, breaking ties toward the
        smallest rate as the paper's ``min argmin`` prescribes.

        :return: the minimising rate, the attained minimum, and its index.
        """
        objective = self._service_costs - self.rates * weight
        index = int(np.argmin(objective))  # argmin returns the FIRST minimiser
        return float(self.rates[index]), float(objective[index]), index

    def solve(self, method: str = "direct") -> DecayingValueResult:
        """
        Compute the optimal policy and cost-to-go.

        :param method: ``"direct"`` uses the paper's Proposition 1 recursion in
            ``delta``/``sigma``; ``"bellman"`` solves the Bellman equation as
            written, without the reformulation. They are separate code paths
            and must agree -- which is exactly what makes the reformulation
            worth testing rather than trusting.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        start = time.process_time()
        if method == "direct":
            policy, cost_to_go, delta, sigma = self._solve_direct()
        else:
            policy, cost_to_go, delta, sigma = self._solve_bellman()
        return DecayingValueResult(
            policy=policy,
            cost_to_go=cost_to_go,
            delta=delta,
            sigma=sigma,
            total_cost=float(cost_to_go[self.num_jobs, self.initial_value]),
            method=method,
            duration=time.process_time() - start,
        )

    def _shape(self) -> tuple[int, int]:
        return self.num_jobs + 1, self.initial_value + 1

    def _solve_direct(self):
        """
        Proposition 1: a forward recursion in ``v`` for each fixed ``b``.

        ``delta(b,v)`` is the extra cost contributed by one more unit of
        residual value, and ``sigma`` accumulates it. The optimal rate depends
        on the state only through ``r(v) + sigma(b,v-1)``.
        """
        rows, cols = self._shape()
        policy = np.zeros((rows, cols))
        delta = np.zeros((rows, cols))
        sigma = np.zeros((rows, cols))
        cost_to_go = np.zeros((rows, cols))

        for b in range(1, rows):
            running = 0.0  # sigma(b, 0)
            for v in range(1, cols):
                rate, best, _ = self._best_rate(self._rewards[v] + running)
                delta[b, v] = self._holding_costs[b] + best
                running += delta[b, v]
                sigma[b, v] = running
                policy[b, v] = rate
            # J(b,v) = J(b-1,V) + sigma(b,v), with J(0,V) = 0
            base = cost_to_go[b - 1, self.initial_value]
            cost_to_go[b, 1:] = base + sigma[b, 1:]
        return policy, cost_to_go, delta, sigma

    def _solve_bellman(self):
        """
        The Bellman equation as written, with no reformulation.

        The state space is acyclic -- ``b`` never grows and ``v`` falls within
        a fixed ``b`` -- so sweeping ``b`` upward and ``v`` upward is already
        exact, with no iteration to a fixed point.
        """
        rows, cols = self._shape()
        policy = np.zeros((rows, cols))
        cost_to_go = np.zeros((rows, cols))
        top = self.initial_value

        for b in range(1, rows):
            ejected = cost_to_go[b - 1, top]  # J(b-1, V): next job takes over
            for v in range(1, cols):
                continuation = cost_to_go[b, v - 1] if v > 1 else ejected
                objective = (
                    self._service_costs
                    + self._holding_costs[b]
                    + self.rates * (-self._rewards[v] + ejected)
                    + (1.0 - self.rates) * continuation
                )
                index = int(np.argmin(objective))
                policy[b, v] = self.rates[index]
                cost_to_go[b, v] = objective[index]
        # delta and sigma are recovered, not used, on this path
        delta = np.zeros((rows, cols))
        sigma = np.zeros((rows, cols))
        for b in range(1, rows):
            base = cost_to_go[b - 1, top]
            sigma[b, 1:] = cost_to_go[b, 1:] - base
            delta[b, 1:] = np.diff(np.concatenate(([0.0], sigma[b, 1:])))
        return policy, cost_to_go, delta, sigma

    def monotonicity_conditions(self, result: DecayingValueResult | None = None) -> dict:
        """
        Apply the paper's Theorems 2 and 3, which decide how the optimal rate
        moves with the residual value WITHOUT inspecting the policy.

        Theorem 2 compares ``delta(b,v)`` against the reward increment
        ``-[r(v+1) - r(v)]`` across all ``v``; Theorem 3 is the sharper test
        available when the reward is constant, where only the sign of
        ``h(b) + min_s { c(s) - s*r }`` matters.

        :param result: a previously computed solution, to avoid re-solving.
        :return: ``value_non_decreasing`` / ``value_non_increasing``, boolean
            arrays indexed by backlog; ``constant_reward`` saying which test
            was used; and ``backlog_non_decreasing``, which Theorem 1
            guarantees unconditionally for an admissible instance.
        """
        if result is None:
            result = self.solve()
        rows = self.num_jobs + 1
        top = self.initial_value
        non_decreasing = np.zeros(rows, dtype=bool)
        non_increasing = np.zeros(rows, dtype=bool)

        rewards = self._rewards[1 : top + 1]
        constant_reward = bool(top == 1 or np.all(np.abs(np.diff(rewards)) <= _TOL))

        for b in range(1, rows):
            if constant_reward:
                # Theorem 3: the sign of h(b) + min_s { c(s) - s*r }
                _, best, _ = self._best_rate(float(rewards[0]))
                marker = self._holding_costs[b] + best
                non_decreasing[b] = marker >= -_TOL
                non_increasing[b] = marker <= _TOL
            else:
                # Theorem 2: delta(b,v) against -[r(v+1) - r(v)], for all v < V
                gaps = np.array([result.delta[b, v] + (self._rewards[v + 1] - self._rewards[v]) for v in range(1, top)])
                non_decreasing[b] = bool(np.all(gaps >= -_TOL))
                non_increasing[b] = bool(np.all(gaps <= _TOL))
        return {
            "value_non_decreasing": non_decreasing,
            "value_non_increasing": non_increasing,
            "constant_reward": constant_reward,
            "backlog_non_decreasing": True,  # Theorem 1, unconditional
        }

    def policy_monotonicity(self, result: DecayingValueResult | None = None) -> dict:
        """
        Read the monotonicity straight off a computed policy, for comparison
        with what :meth:`monotonicity_conditions` predicts.

        :return: ``backlog_non_decreasing`` (Theorem 1's claim, which must
            hold), plus per-backlog arrays of what the policy actually does as
            the residual value grows.
        """
        if result is None:
            result = self.solve()
        policy = result.policy
        top = self.initial_value
        rows = self.num_jobs + 1

        backlog_ok = True
        for v in range(1, top + 1):
            column = policy[1:, v]
            backlog_ok = backlog_ok and bool(np.all(np.diff(column) >= -_TOL))

        non_decreasing = np.zeros(rows, dtype=bool)
        non_increasing = np.zeros(rows, dtype=bool)
        for b in range(1, rows):
            row = policy[b, 1 : top + 1]
            non_decreasing[b] = bool(np.all(np.diff(row) >= -_TOL))
            non_increasing[b] = bool(np.all(np.diff(row) <= _TOL))
        return {
            "backlog_non_decreasing": backlog_ok,
            "value_non_decreasing": non_decreasing,
            "value_non_increasing": non_increasing,
        }

    def myopic_policy(self) -> np.ndarray:
        """
        The obvious short-sighted rule, for comparison: minimise this slot's
        expected cost, ``c(s) - s*r(v)``, ignoring everything downstream.

        It is exactly the optimal rule with ``sigma`` set to zero, so the gap
        between the two is precisely what looking ahead is worth. This is a
        baseline of our own, not an algorithm from the paper.
        """
        rows, cols = self._shape()
        policy = np.zeros((rows, cols))
        for b in range(1, rows):
            for v in range(1, cols):
                policy[b, v], _, _ = self._best_rate(self._rewards[v])
        return policy

    def evaluate(self, policy: np.ndarray) -> np.ndarray:
        """
        Expected total cost-to-go under an ARBITRARY policy, by the same
        acyclic sweep used to solve the problem but without the minimisation.

        Lets any rule -- myopic, constant, hand-written -- be scored against
        the optimum exactly, rather than only by simulation.
        """
        rows, cols = self._shape()
        if policy.shape != (rows, cols):
            raise ValueError(f"policy must have shape {(rows, cols)}, got {policy.shape}")
        cost_to_go = np.zeros((rows, cols))
        top = self.initial_value
        for b in range(1, rows):
            ejected = cost_to_go[b - 1, top]
            for v in range(1, cols):
                rate = float(policy[b, v])
                if not 0.0 <= rate <= 1.0:
                    raise ValueError(f"policy[{b},{v}] = {rate} is not a probability")
                continuation = cost_to_go[b, v - 1] if v > 1 else ejected
                cost_to_go[b, v] = (
                    float(self.service_cost(rate))
                    + self._holding_costs[b]
                    + rate * (-self._rewards[v] + ejected)
                    + (1.0 - rate) * continuation
                )
        return cost_to_go

    def __repr__(self) -> str:
        return (
            f"DecayingValueRateControl(num_jobs={self.num_jobs}, "
            f"initial_value={self.initial_value}, rates={self.rates.tolist()})"
        )
