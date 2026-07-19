"""
Two-class M/G/1 with preemptive-repeat priority (Gaver, JRSS-B, 1962).

A high-priority arrival interrupts the low-priority job in service. Under
**repeat with resampling (RS)** the interrupted job later restarts with a
fresh service draw; under **repeat without resampling / repeat-identical
(RW)** it restarts with the same total duration.

What is exact here:

- `MG1PreemptiveRepeatCalc(kind="RS")` — the RS queueing model is solved
  EXACTLY as a CTMC (`MapPh1PriorityCalc(discipline="RS")`) with both service
  distributions fitted by Coxian-2 from their first three moments; the same
  discipline as `PriorityQueueSimulator(prty_type="RS")`, giving the first
  analytical benchmark for it in the library.
- `completion_time_mean(...)` — Gaver's closed forms for the mean completion
  time C (first service start to completion) of a low job, both kinds:

      E[C_RS] = (1/E[e^{-aS}] - 1) (1/a + E[B]),
      E[C_RW] = (E[e^{aS}] - 1) (1/a + E[B]),

  where a is the high-class rate and B its M/G/1 busy period
  (E[B] = b_H/(1 - rho_H)). E[e^{+-aS}] is evaluated through a Gamma fit of
  the low service distribution. RW requires a light service tail
  (E[e^{aS}] < infinity).

The RW *queueing* model has no finite Markov representation (the repeated
identical duration must be remembered), and delay-cycle candidates
calibrated against the exact RS CTMC show O(1%) systematic error — so RW
waiting times are deliberately NOT provided (reserved; see EPIC-020).
"""

import time

import numpy as np

from most_queue.random.distributions import CoxDistribution, GammaDistribution
from most_queue.structs import MulticlassResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.priority.map_ph_priority import MapPh1PriorityCalc
from most_queue.theory.utils.transforms import lst_gamma


def completion_time_mean(b_low: list[float], l_high: float, b_high: list[float], kind: str = "RS") -> float:
    """
    Mean completion time of a low-priority job under preemptive repeat.

    :param b_low: raw moments of the low-class service time.
    :param l_high: arrival rate of the interrupting (high) class.
    :param b_high: raw moments of the high-class service time.
    :param kind: "RS" (resampling) or "RW" (repeat-identical).
    """
    a = float(l_high)
    if a <= 0:
        return float(b_low[0])
    rho_h = a * b_high[0]
    if rho_h >= 1.0:
        raise ValueError(f"High class alone is unstable: rho_H = {rho_h:.4f} >= 1")
    mean_busy = b_high[0] / (1.0 - rho_h)
    gamma_params = GammaDistribution.get_params(b_low[:2])
    if kind.upper() == "RS":
        q = lst_gamma(gamma_params, a)
        return (1.0 / q - 1.0) * (1.0 / a + mean_busy)
    if kind.upper() == "RW":
        if gamma_params.mu <= a:
            raise ValueError(
                "E[e^{aS}] diverges: the low service tail is too heavy for "
                f"repeat-identical (Gamma rate {gamma_params.mu:.4f} <= a = {a:.4f})"
            )
        mgf = lst_gamma(gamma_params, -a)
        return (mgf - 1.0) * (1.0 / a + mean_busy)
    raise ValueError("kind must be 'RS' or 'RW'")


class MG1PreemptiveRepeatCalc(BaseQueue):
    """
    Exact two-class M/G/1 preemptive repeat-with-resampling solver (Cox-2
    service fit + RS CTMC).

    :param kind: only "RS" is solvable; "RW" raises on run() (see module
        docstring) but its completion-time mean is available via
        `completion_time_mean`.
    """

    def __init__(self, kind: str = "RS"):
        super().__init__(n=1)
        self.kind = kind.upper()
        if self.kind not in ("RS", "RW"):
            raise ValueError("kind must be 'RS' or 'RW'")
        self.l = None
        self.b = None
        self.completion_means = None

    def set_sources(self, l: list[float]):  # pylint: disable=arguments-differ
        """:param l: arrival rates [high, low]."""
        if len(l) != 2:
            raise ValueError("Exactly two classes are supported")
        self.l = [float(x) for x in l]
        self.is_sources_set = True

    def set_servers(self, b: list[list[float]]):  # pylint: disable=arguments-differ
        """:param b: raw service moments per class, [b_high, b_low] (3 moments each)."""
        self.b = [list(x) for x in b]
        self.is_servers_set = True

    def run(self, q_start: int = 60, q_max: int = 240) -> MulticlassResults:
        """
        Solve the RS model exactly via the CTMC; also computes the mean
        completion times for both RS and RW.
        """
        start = time.process_time()
        self._check_if_servers_and_sources_set()

        self.completion_means = {
            "RS": completion_time_mean(self.b[1], self.l[0], self.b[0], "RS"),
        }
        try:
            self.completion_means["RW"] = completion_time_mean(self.b[1], self.l[0], self.b[0], "RW")
        except ValueError:
            self.completion_means["RW"] = None

        if self.kind == "RW":
            raise NotImplementedError(
                "The RW (repeat-identical) queueing model has no finite Markov "
                "representation; only completion_time_mean is available (see "
                "module docstring / EPIC-020 reserve)."
            )

        def fit_ph(moments):
            b1, b2 = moments[0], moments[1]
            if abs(b2 - 2.0 * b1 * b1) < 1e-8 * b1 * b1:
                # exponential moments: the Cox fit degenerates, use PH(1)
                return ([1.0], [[-1.0 / b1]])
            c = CoxDistribution.get_params(moments)
            # the Cox fit may return complex params with negligible imaginary
            # parts (library convention); the CTMC needs real rates
            p1, mu1, mu2 = (float(np.real(x)) for x in (c.p1, c.mu1, c.mu2))
            return ([1.0, 0.0], [[-mu1, p1 * mu1], [0.0, -mu2]])

        ctmc = MapPh1PriorityCalc(discipline="RS")
        ctmc.set_sources(
            D0=[[-(self.l[0] + self.l[1])]],
            D1_high=[[self.l[0]]],
            D1_low=[[self.l[1]]],
        )
        ctmc.set_servers(ph_high=fit_ph(self.b[0]), ph_low=fit_ph(self.b[1]))
        res = ctmc.run(q_start=q_start, q_max=q_max)

        self.results = MulticlassResults(
            w=res.w,
            v=res.v,
            utilization=res.utilization,
            duration=time.process_time() - start,
        )
        return self.results
