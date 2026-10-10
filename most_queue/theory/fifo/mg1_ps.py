"""
M/G/1 with egalitarian Processor Sharing (PS).

The stationary number of jobs is geometric (1 - rho) * rho^k, insensitive to
the service distribution beyond its mean (BCMP). The conditional mean sojourn
time of a job of size x is exactly x / (1 - rho), so the slowdown is uniform:
every job is stretched by the same factor 1 / (1 - rho).

The mean is where the insensitivity stops. Higher moments of the conditional
sojourn time DO depend on the service-time distribution, and they are available
here via ``get_conditional_sojourn_moments`` / ``get_conditional_sojourn_var``
once the service time is described by more than its mean (``set_servers_from_moments``
or ``set_servers_params``). That is Yashkov's result, implemented in
:mod:`most_queue.theory.fifo._ps_sojourn`; everything on this class that only
needs ``b[0]`` keeps working with ``set_servers`` as before.

References:
    Kleinrock L. Time-shared Systems: A Theoretical Treatment. JACM, 14(2),
        1967. doi:10.1145/321386.321388.
    Baskett F., Chandy K.M., Muntz R.R., Palacios F.G. Open, Closed, and Mixed
        Networks of Queues with Different Classes of Customers. JACM, 22(2),
        1975. doi:10.1145/321879.321887 (insensitivity).
    Yashkov S.F. Processor-Sharing Queues: Some Progress in Analysis. Queueing
        Systems, 2, 1987. doi:10.1007/bf01182931.
    Yashkov S.F. Explicit formulas for the moments of the sojourn time in the
        M/G/1 processor sharing queue with permanent jobs. arXiv:math/0512281,
        2005 (the moment recursion implemented here).
"""

import math

import numpy as np
from scipy.integrate import simpson
from scipy.linalg import expm

from most_queue.random.distributions import ErlangDistribution, H2Distribution
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams
from most_queue.theory.fifo._ps_sojourn import (
    conditional_sojourn_moments,
    equilibrium_ph,
    ph_mean,
    ph_representation,
    waiting_time_ph,
)


class MG1PSCalc(BaseQueue):
    """
    M/G/1 Processor Sharing: the server is shared equally by all jobs present
    (each of k jobs is served at rate 1/k). No queue and no waiting in the
    classic sense; "waiting" below is the delay E[V] - b1 caused by sharing.
    """

    def __init__(self, calc_params: CalcParams | None = None):
        super().__init__(n=1, calc_params=calc_params)

        self.l = None  # arrival intensity
        self.b = None  # service time raw moments
        self.service_params = None  # phase-type params, if the shape was supplied
        self._wait_ph = None  # cached (pi, S) of the FCFS waiting time

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """
        Set sources
        :param l: arrival rate
        """
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = l
        self.is_sources_set = True

    def set_servers(self, b: list[float]):  # pylint: disable=arguments-differ
        """
        Set servers
        :param b: raw moments of service time distribution (only b[0] is used —
            PS characteristics are insensitive to the rest)
        """
        if not b or b[0] <= 0:
            raise ValueError("Service time moments must be non-empty with positive mean")
        self.b = list(b)
        self.service_params = None
        self._wait_ph = None
        self.is_servers_set = True

    def set_servers_params(self, params):
        """
        Set the service time by its phase-type PARAMETERS rather than by moments.

        Needed only for the quantities that are not insensitive -- the higher
        conditional sojourn-time moments. ``params`` is :class:`H2Params`
        (CV >= 1), :class:`ErlangParams` (CV <= 1) or an integer-shape
        :class:`GammaParams`.
        """
        alpha, t_mat = ph_representation(params)
        mean = ph_mean(alpha, t_mat)
        if mean <= 0:
            raise ValueError(f"service-time params give a non-positive mean: {mean}")
        self.service_params = params
        self.b = [mean]
        self._wait_ph = None
        self.is_servers_set = True

    def set_servers_from_moments(self, b: list[float], family: str = "auto"):
        """
        Set the service time by fitting a phase-type distribution to its raw moments.

        Same CV-based convention as the rest of the library: ``cv <= 1`` fits an
        Erlang, ``cv > 1`` fits an H2 (Aliev's real-valued method). Use this when
        you want the higher conditional sojourn moments but only have moments.

        Be aware of what the fit does and does not promise: ``E[V(x)]`` and the
        queue-length distribution are insensitive and so are exact whatever is
        fitted, but the VARIANCE and higher moments genuinely depend on the shape
        of the service-time distribution, so they are exact for the FITTED law,
        not for an arbitrary one matching the same moments.

        :param b: raw moments; two are enough for Erlang, three are needed for H2.
        :param family: ``"auto"``, ``"erlang"`` or ``"h2"``.
        """
        if len(b) < 2:
            raise ValueError("at least two raw moments (mean, second moment) are required")
        variance = b[1] - b[0] ** 2
        if variance < 0:
            raise ValueError(f"moments are not valid: variance {variance} is negative")
        cv = math.sqrt(variance) / b[0]
        resolved = ("erlang" if cv <= 1.0 else "h2") if family == "auto" else family

        if resolved == "erlang":
            if cv > 1.0:
                raise ValueError(f"Erlang cannot represent cv > 1 (got cv={cv:.4f}); use family='h2' or 'auto'")
            params = ErlangDistribution.get_params(list(b[:2]))
        elif resolved == "h2":
            if cv < 1.0:
                raise ValueError(f"H2 cannot represent cv < 1 (got cv={cv:.4f}); use family='erlang' or 'auto'")
            if len(b) < 3:
                raise ValueError("family='h2' requires at least three raw moments")
            params = H2Distribution.get_params(list(b[:3]))
        else:
            raise ValueError(f"unknown family: {family!r}; expected 'auto', 'erlang' or 'h2'")

        self.set_servers_params(params)
        self.b = list(b)  # keep the user's moments for the insensitive quantities

    def _utilization(self) -> float:
        self._check_if_servers_and_sources_set()
        ro = self.l * self.b[0]
        if ro >= 1:
            raise ValueError(f"System is unstable: utilization rho={ro} must be < 1")
        return ro

    def get_p(self) -> list[float]:
        """
        Get probabilities of states: geometric, p[k] = (1 - rho) * rho^k
        (insensitive to the service distribution).
        """
        ro = self._utilization()
        num_probs = self.calc_params.p_num
        self.p = [(1.0 - ro) * ro**k for k in range(num_probs)]
        return self.p

    def get_conditional_sojourn_mean(self, x: float) -> float:
        """
        Exact conditional mean sojourn time of a job of size x: x / (1 - rho).
        """
        return x / (1.0 - self._utilization())

    def get_mean_slowdown(self) -> float:
        """
        Mean slowdown (sojourn / size), uniform over sizes: 1 / (1 - rho).
        """
        return 1.0 / (1.0 - self._utilization())

    # -------------------- higher conditional sojourn moments (Yashkov) ---------
    def _waiting_phase_type(self):
        """Cached ``(pi, S)`` of the FCFS waiting time that the recursion runs on."""
        if self._wait_ph is None:
            if self.service_params is None:
                raise ValueError(
                    "higher sojourn moments need the SHAPE of the service time, not just its mean. "
                    "Use set_servers_from_moments(b) to fit one, or set_servers_params(...) to give "
                    "it explicitly. (The mean sojourn time and the queue length are insensitive and "
                    "remain available from set_servers alone.)"
                )
            alpha, t_mat = ph_representation(self.service_params)
            pi_eq = equilibrium_ph(alpha, t_mat)
            self._wait_ph = (pi_eq, waiting_time_ph(pi_eq, t_mat, self._utilization()))
        return self._wait_ph

    def get_conditional_sojourn_moments(self, x: float, num: int = 2, permanent_jobs: int = 0) -> list[float]:
        """
        Exact raw moments of the sojourn time of a job of size ``x``.

        The first one is the insensitive ``x / (1 - rho)``; the rest are not
        insensitive and are where the Yashkov machinery is needed. See
        :mod:`most_queue.theory.fifo._ps_sojourn` for the construction and the
        attribution.

        :param x: job size (service requirement).
        :param num: how many raw moments.
        :param permanent_jobs: ``K`` permanent jobs of infinite size sharing the
            processor alongside the ordinary ones; the sojourn time is then a sum
            of ``K+1`` independent copies of the ordinary one.
        """
        rho = self._utilization()
        pi_eq, s_mat = self._waiting_phase_type()
        return conditional_sojourn_moments(x, num, pi_eq, s_mat, rho, permanent_jobs=permanent_jobs)

    def get_conditional_sojourn_var(self, x: float, permanent_jobs: int = 0) -> float:
        """
        Exact variance of the sojourn time of a job of size ``x``.

        Equation (3.10) of Yashkov's note, ``2/(1-rho)^2 int_0^x (x-y)(1-W(y))dy``
        with ``W`` the M/G/1-FCFS waiting-time distribution, obtained here through
        the general recursion rather than by quadrature. Unlike the mean, this
        DOES depend on the service-time distribution.
        """
        moments = self.get_conditional_sojourn_moments(x, 2, permanent_jobs=permanent_jobs)
        return moments[1] - moments[0] ** 2

    def get_conditional_sojourn_cv(self, x: float) -> float:
        """Coefficient of variation of the sojourn time of a job of size ``x``."""
        moments = self.get_conditional_sojourn_moments(x, 2)
        return math.sqrt(max(moments[1] - moments[0] ** 2, 0.0)) / moments[0]

    def get_v_moments(self, num: int = 2, quad_points: int = 2001, tail_quantile: float = 1e-10) -> list[float]:
        """
        Raw moments of the UNCONDITIONAL sojourn time, ``E[V^n] = int v_n(u) dB(u)``.

        The conditional moments are exact; this outer integration over the
        service-time distribution is the one numerical step, done by Simpson's
        rule on a grid that extends until the fitted service-time tail falls
        below ``tail_quantile``. The first moment must come back as the exact
        ``b1/(1-rho)``, which is a usable accuracy check on the quadrature.
        """
        rho = self._utilization()
        pi_eq, s_mat = self._waiting_phase_type()
        alpha, t_mat = ph_representation(self.service_params)

        # Grid out to where the service-time tail is negligible.
        decay = float(np.min(np.abs(np.linalg.eigvals(t_mat).real)))
        upper = -math.log(tail_quantile) / decay
        grid = np.linspace(0.0, upper, quad_points)
        exit_rates = -t_mat @ np.ones(len(alpha))
        density = np.array([float(alpha @ expm(t_mat * u) @ exit_rates) for u in grid])

        out = []
        for n in range(1, num + 1):
            vals = np.array([conditional_sojourn_moments(float(u), n, pi_eq, s_mat, rho)[-1] for u in grid])
            out.append(float(simpson(vals * density, x=grid)))
        self.v = out
        return out

    def get_v(self, num: int = 1) -> list[float]:
        """
        Raw moments of the sojourn time.

        The first one is the exact insensitive ``b1 / (1 - rho)``. Asking for
        more requires the service-time shape (see
        :meth:`set_servers_from_moments`) and goes through
        :meth:`get_v_moments`; the first entry is still reported from the closed
        form rather than from the quadrature.
        """
        exact_first = self.b[0] / (1.0 - self._utilization())
        if num <= 1:
            self.v = [exact_first]
        else:
            self.v = [exact_first] + self.get_v_moments(num)[1:]
        return self.v

    def get_w(self, num: int = 1) -> list[float]:  # pylint: disable=unused-argument
        """
        Mean sharing delay (first moment only): E[V] - b1 = rho * b1 / (1 - rho).

        Higher UNCONDITIONAL delay moments are deliberately absent: the delay
        ``V - S`` and the job's own size ``S`` are dependent, so they do not
        follow from the moments of ``V`` and ``S`` separately. Conditioned on the
        size they do -- ``V(x) - x`` is a pure shift of
        :meth:`get_conditional_sojourn_moments`.
        """
        ro = self._utilization()
        self.w = [ro * self.b[0] / (1.0 - ro)]
        return self.w

    def run(self, num_of_moments: int = 1) -> QueueResults:
        """
        Run calculation. ``num_of_moments > 1`` needs the service-time shape.
        """
        start = self._measure_time()
        with self._validate_state():
            utilization = self._utilization()
            p = self.get_p()
            w = self.get_w(num_of_moments)
            v = self.get_v(num_of_moments)
            self.ro = utilization

        result = QueueResults(p=p, w=w, v=v, utilization=utilization)
        self._set_duration(result, start)
        return result
