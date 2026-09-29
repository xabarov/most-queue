"""
Exact M/M/2 with two preemptive-resume priority classes and heterogeneous
servers (mu_a != mu_b) -- EPIC-025.

Every prior multi-server priority model in the library (MMkPriorityExact,
RDRAPriorityCalc, MPhPhK2Class) assumes servers are interchangeable: state is
just the per-class counts (n_1,...,n_m), because it never matters *which*
server is busy. With two servers at different rates that stops being true:
progress depends on which physical server a job landed on. This mirrors the
EPIC-023 machine-repair extension (Krishnamoorthi's 1963 technique for two
heterogeneous exponential servers), applied here on top of a priority
discipline instead of a repair-crew discipline.

Literature: Analysis of a Pre-Emptive Two-Priority Queuing System with
Impatient Customers and Heterogeneous Servers, Mathematics, 2023,
doi:10.3390/math11183878 (base setting, without impatience here). Krishnamoorthi
B., On Poisson Queue with Two Heterogeneous Servers, Operations Research,
1963, doi:10.1287/opre.11.3.321 (the state-splitting technique itself).

Discipline: class 0 (high priority) preempts class 1; an idle server is
always preferred over preempting an occupied one; a job already in service
never migrates to a different (even faster) server once started ("sticky").

State: (n0, n1) -- per-class counts -- plus a `config` bit that disambiguates
which physical server is occupied, needed only in the boundary cases n0 == 1
(which server holds the class-0 job) and n0 == 0, n1 == 1 (which server holds
the class-1 job); everywhere else the server assignment is unambiguous
(n0 >= 2: both servers serve class 0; n0 == 0, n1 >= 2: both serve class 1;
n0 == n1 == 0: both idle).

The full transition list is built mechanically from four small pure
functions (`_canonicalize`, `_class0_arrival`, `_class1_arrival`,
`_departure`) operating on the "raw" per-server assignment (a_class, b_class)
-- see docs/roadmaps/priority_heterogeneous_servers_roadmap.md sec. 2 -- to
avoid hand-deriving each of the many branching transitions by hand (real
risk of a transcription error at this branching factor). Reduces exactly to
MMkPriorityExact(n=2, ...) when mu_a == mu_b -- the primary regression test.
"""

import numpy as np

from most_queue.structs import PriorityResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.reliability.utils import ctmc_stationary

Config = str | None  # "A", "B" or None (unambiguous)


def _servers_from_state(n0: int, n1: int, config: Config):
    """(a_class, b_class) in {0, 1, None} implied by a canonical (n0, n1, config)."""
    if n0 >= 2:
        return 0, 0
    if n0 == 1:
        if config == "A":
            return 0, (1 if n1 >= 1 else None)
        return (1 if n1 >= 1 else None), 0
    # n0 == 0
    if n1 == 0:
        return None, None
    if n1 == 1:
        return (1, None) if config == "A" else (None, 1)
    return 1, 1


def _config_from_servers(n0: int, n1: int, a_class, _b_class) -> Config:
    if n0 == 1:
        return "A" if a_class == 0 else "B"
    if n0 == 0 and n1 == 1:
        return "A" if a_class == 1 else "B"
    return None


def _canonicalize(n0: int, n1: int, a_class, b_class):
    """Fill any idle slot from the backlog (class 0 first), return the canonical state."""
    backlog0 = n0 - (a_class == 0) - (b_class == 0)
    backlog1 = n1 - (a_class == 1) - (b_class == 1)
    for slot in ("a", "b"):
        cur = a_class if slot == "a" else b_class
        if cur is not None:
            continue
        if backlog0 > 0:
            new_val, backlog0 = 0, backlog0 - 1
        elif backlog1 > 0:
            new_val, backlog1 = 1, backlog1 - 1
        else:
            continue
        if slot == "a":
            a_class = new_val
        else:
            b_class = new_val
    return n0, n1, _config_from_servers(n0, n1, a_class, b_class)


def _class0_arrival(n0: int, n1: int, a_class, b_class):
    n0 += 1
    if a_class is None:
        a_class = 0
    elif b_class is None:
        b_class = 0
    elif a_class == 1:
        a_class = 0  # preempt the fast server first (best resource for the new job)
    elif b_class == 1:
        b_class = 0
    # else: both already class 0 -- just grows the class-0 backlog
    return _canonicalize(n0, n1, a_class, b_class)


def _class1_arrival(n0: int, n1: int, a_class, b_class):
    n1 += 1
    if a_class is None:
        a_class = 1
    elif b_class is None:
        b_class = 1
    # class 1 never preempts -- otherwise just grows the class-1 backlog
    return _canonicalize(n0, n1, a_class, b_class)


def _departure(n0: int, n1: int, a_class, b_class, server: str):
    served = a_class if server == "a" else b_class
    if server == "a":
        a_class = None
    else:
        b_class = None
    if served == 0:
        n0 -= 1
    else:
        n1 -= 1
    return _canonicalize(n0, n1, a_class, b_class)


class MM2PriorityHeterogeneousCalc(BaseQueue):
    """
    Exact truncated-CTMC solver for M/M/2 with two preemptive-resume priority
    classes and heterogeneous servers.

    :param truncation: per-class state-space cap (n0, n1 each capped at this
        value). None auto-sizes from the load.
    """

    MAX_STATES = 400_000

    def __init__(self, truncation: int | None = None):
        super().__init__(n=2)
        self.truncation = truncation
        self.l0 = None
        self.l1 = None
        self.mu_a = None
        self.mu_b = None

    def set_sources(self, l: list[float]):  # pylint: disable=arguments-differ
        """:param l: [lambda_0, lambda_1], class 0 = high priority."""
        if len(l) != 2:
            raise ValueError("exactly two classes are supported")
        self.l0, self.l1 = float(l[0]), float(l[1])
        self.is_sources_set = True

    def set_servers(self, mu_a: float, mu_b: float):  # pylint: disable=arguments-differ
        """:param mu_a, mu_b: server rates, any order -- normalised so mu_a >= mu_b."""
        self.mu_a, self.mu_b = max(mu_a, mu_b), min(mu_a, mu_b)
        self.is_servers_set = True

    def _auto_truncation(self) -> int:
        total_rho = (self.l0 + self.l1) / (self.mu_a + self.mu_b)
        total_rho = min(total_rho, 0.97)
        return int(min(100, max(15, 6.0 / (1.0 - total_rho))))

    def _states(self, cap: int):
        """Enumerate all valid (n0, n1, config) states, 0 <= n0, n1 <= cap."""
        states = []
        for n0 in range(cap + 1):
            for n1 in range(cap + 1):
                if n0 == 1 or (n0 == 0 and n1 == 1):
                    states.extend([(n0, n1, "A"), (n0, n1, "B")])
                else:
                    states.append((n0, n1, None))
        return states

    def run(self) -> PriorityResults:
        """Build and solve the truncated CTMC; per-class moments via Little's law."""
        start = self._measure_time()
        with self._validate_state():
            cap = self.truncation if self.truncation is not None else self._auto_truncation()
            states = self._states(cap)
            if len(states) > self.MAX_STATES:
                raise ValueError(f"truncated state space is {len(states)} > {self.MAX_STATES}; reduce truncation")
            index = {s: i for i, s in enumerate(states)}
            n_states = len(states)

            transitions = []
            for n0, n1, config in states:
                a_class, b_class = _servers_from_state(n0, n1, config)
                src = index[(n0, n1, config)]

                if n0 < cap:
                    dst = index[_class0_arrival(n0, n1, a_class, b_class)]
                    transitions.append((src, dst, self.l0))
                if n1 < cap:
                    dst = index[_class1_arrival(n0, n1, a_class, b_class)]
                    transitions.append((src, dst, self.l1))
                if a_class is not None:
                    dst = index[_departure(n0, n1, a_class, b_class, "a")]
                    transitions.append((src, dst, self.mu_a))
                if b_class is not None:
                    dst = index[_departure(n0, n1, a_class, b_class, "b")]
                    transitions.append((src, dst, self.mu_b))

            pi = ctmc_stationary(transitions, n_states)

            n0_arr = np.array([s[0] for s in states])
            n1_arr = np.array([s[1] for s in states])
            e_n0 = float(np.dot(n0_arr, pi))
            e_n1 = float(np.dot(n1_arr, pi))

            e_v0 = e_n0 / self.l0
            e_v1 = e_n1 / self.l1
            # V = W + S for FCFS-within-class. A class-i job's own service
            # duration is 1/mu_a or 1/mu_b depending on which server it lands
            # on, not a single constant -- so recover E[S_i] via Little's law
            # applied to "class-i jobs currently in service" instead of a
            # hand-picked rate: throughput of class i entering service is l_i
            # (stable system), so E[in_service_i] = l_i * E[S_i].
            in_service_0 = np.array([1.0 * (_servers_from_state(*s).count(0)) for s in states])
            in_service_1 = np.array([1.0 * (_servers_from_state(*s).count(1)) for s in states])
            e_in_service_0 = float(np.dot(in_service_0, pi))
            e_in_service_1 = float(np.dot(in_service_1, pi))
            e_s0 = e_in_service_0 / self.l0
            e_s1 = e_in_service_1 / self.l1
            e_w0 = e_v0 - e_s0
            e_w1 = e_v1 - e_s1

            utilization = (self.l0 + self.l1) / (self.mu_a + self.mu_b)

        result = PriorityResults(
            v=[[e_v0, 0, 0, 0], [e_v1, 0, 0, 0]],
            w=[[e_w0, 0, 0, 0], [e_w1, 0, 0, 0]],
            p=[],
            utilization=utilization,
        )
        self._set_duration(result, start)
        return result
