"""
M/M/c in which a customer's service rate depends on the queueing delay that
customer experienced: ``mu1`` if the delay was at most a threshold ``k``,
``mu2`` otherwise.

Implementation of
    D'Auria B., Adan I.J.B.F., Bekker R., Kulkarni V.,
    "An M/M/c queue with queueing-time dependent service rates",
    European Journal of Operational Research 299(2):566-579, 2022,
    doi:10.1016/j.ejor.2021.12.023 (preprint arXiv:2107.04557).

THE MODEL AND THE SOLUTION ARE THEIRS, not this library's. This module is a
from-the-paper implementation; the authors also published their own code, which
was not consulted or copied. Equation numbers in the comments below refer to the
arXiv version.

Why the model matters. Classical queueing assumes service times are independent
of the delay already suffered, which is analytically convenient and empirically
false in several domains: in health care a delayed admission lengthens the stay
("slowdown"), in retail customers who waited longer consume more, and a server
may deliberately speed up or slow down under congestion. The paper's point is
blunt and worth repeating: ignoring the dependence can give "wholly inadequate"
performance figures -- their numerics show E[W] losing even its convexity in
lambda, which no classical M/M/c can do.

The construction, in outline. Let ``W(t)`` be the virtual queueing time (VQT):
a customer arriving at ``t`` starts service at ``t + W(t)``. ``W`` alone is not
Markov here, because the rate at which it decreases depends on which classes the
currently-committed services belong to. The paper closes the state with the
SERVER STATE ``S(t) = (S1(t), S2(t))`` -- how many servers will be serving class
1 and class 2 customers at time ``t + W(t)``, just before the new service
starts. Whenever ``W(t) > 0`` all servers are committed, so ``S1 + S2 = c - 1``
and only ``c`` phases survive; ``F_i(x) = P(0 < W <= x, S = (i, c-1-i))``.

The balance equations give integro-differential equations for ``F``, which turn
into second-order linear ODEs with constant coefficients (Theorem 2) -- one
system below the threshold and one above it, coupled through the continuity
conditions at ``k``. Their solution is a MIXTURE OF MATRIX EXPONENTIALS
(Theorem 3), and all the unknown constants collapse, via a recursion over the
boundary states, onto a single scalar ``b_c`` fixed by normalisation
(Theorems 5 and 6).

What this module computes exactly: ``P(W = 0)``, ``P(W <= x)``, the density,
``E[W]``, the boundary probabilities ``pi(i, j)``, and -- a small addition, not
in the paper -- ``E[V] = E[W] + E[S]``, since PASTA makes a random arrival class
1 exactly with probability ``P(W <= k)``.

Degenerate check built into the tests: ``mu1 == mu2`` is an ordinary M/M/c, and
the VQT must reduce to the Erlang-C tail ``C(c, a) exp(-c mu (1 - rho) x)``.
"""

import numpy as np
import scipy.linalg as sla

from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue
from most_queue.theory.calc_params import CalcParams

# Below this gap two eigenvalues count as coincident, which breaks the
# mixture-of-exponentials representation (Remark 2 of the paper).
_EIGENVALUE_GAP_TOL = 1e-9
# Above this, a complex part in an eigenvalue is a real failure, not round-off.
_IMAG_TOL = 1e-8
# Relative closeness to the resonance surface mu2 == mu1 + lam/c at which the
# mixture-of-exponentials representation stops being usable (see _solve).
_RESONANCE_TOL = 1e-7


def _integral_exp(d_mat: np.ndarray, t: float) -> np.ndarray:
    """
    ``int_0^t exp(D s) ds``, via the standard augmented-matrix identity

        expm([[D, I], [0, 0]] t) = [[exp(Dt), int_0^t exp(Ds) ds], [0, I]].

    Written this way on purpose: the closed form ``D^{-1}(exp(Dt) - I)`` breaks
    down exactly where this model needs it, since ``U_1^-`` or ``U_1^+`` carries a
    zero eigenvalue whenever ``lam != c*mu1`` (equations (20)-(21) put
    ``min(0, lam - c*mu1)`` in one group and ``max(0, lam - c*mu1)`` in the other).
    The integral itself is perfectly finite there -- it is only the factored form
    that is singular, which is what the paper's "defined by continuity" remark
    under equation (85) means.
    """
    n = d_mat.shape[0]
    aug = np.zeros((2 * n, 2 * n))
    aug[:n, :n] = d_mat
    aug[:n, n:] = np.eye(n)
    return sla.expm(aug * t)[:n, n:]


def _integral_x_exp(a: float, b: float, d_mat: np.ndarray) -> np.ndarray:
    """``I(a, b; D) = int_a^b D x exp(Dx) dx`` -- equation (85), singular-safe."""
    boundary = b * sla.expm(d_mat * b) - a * sla.expm(d_mat * a)
    return boundary - (_integral_exp(d_mat, b) - _integral_exp(d_mat, a))


class MMcDelayDependentServiceCalc(BaseQueue):
    """
    M/M/c with queueing-time dependent service rates (D'Auria et al., 2022).

    A customer whose queueing delay is at most ``k`` is served at rate ``mu1``;
    one that waited longer is served at rate ``mu2``. ``mu2 < mu1`` is the
    slowdown case, ``mu2 > mu1`` the speedup case, and ``mu1 == mu2`` is an
    ordinary M/M/c.

    :param c: number of identical servers.
    :param k: delay threshold separating the two service rates, ``k >= 0``.
    :param calc_params: standard calculation parameters (unused here beyond the
        base class; the solution is closed-form, not iterative).
    """

    def __init__(self, c: int, k: float, calc_params: CalcParams | None = None):
        super().__init__(n=c, calc_params=calc_params)
        if k < 0:
            raise ValueError(f"threshold k must be non-negative, got {k}")
        self.c = c
        self.k = float(k)
        self.l: float | None = None
        self.mu1: float | None = None
        self.mu2: float | None = None
        self._sol: dict | None = None

    def set_sources(self, l: float):  # pylint: disable=arguments-differ
        """:param l: arrival rate (Poisson)."""
        if l <= 0:
            raise ValueError(f"Arrival rate must be positive, got {l}")
        self.l = float(l)
        self.is_sources_set = True

    def set_servers(self, mu1: float, mu2: float):  # pylint: disable=arguments-differ
        """
        :param mu1: service rate of a customer whose delay was ``<= k``.
        :param mu2: service rate of a customer whose delay was ``> k``.
        """
        if mu1 <= 0 or mu2 <= 0:
            raise ValueError(f"service rates must be positive, got mu1={mu1}, mu2={mu2}")
        self.mu1 = float(mu1)
        self.mu2 = float(mu2)
        self.is_servers_set = True

    # ----------------------------------------------------------- building blocks
    def _blocks(self):
        """B1, B2 and the Delta_n family -- the paper's Section 4 notation."""
        c, mu1, mu2 = self.c, self.mu1, self.mu2
        b1 = np.zeros((c, c))
        b2 = np.zeros((c, c))
        for i in range(c):
            b1[i, i] = (i + 1) * mu1
            if i < c - 1:
                b1[i, i + 1] = (c - 1 - i) * mu2
            b2[i, i] = (c - i) * mu2
            if i >= 1:
                b2[i, i - 1] = i * mu1
        deltas = [np.diag([j * mu1 + (n - j) * mu2 for j in range(n + 1)]) for n in range(c)]
        return b1, b2, deltas

    def _solve_u(self, b_mat: np.ndarray, dt_mat: np.ndarray):
        """
        The two solutions ``U^-``, ``U^+`` of ``U^2 - U(lam I - Dt) + lam(B - Dt) = 0``
        (Section 4.2), one with non-positive and one with non-negative eigenvalues.

        The paper builds them from the closed-form roots (20)/(21) and (24)/(25)
        of a scalar quadratic, which is possible because the matrices happen to be
        triangular. Here the equivalent quadratic eigenvalue problem is linearised
        into an ordinary one instead -- the same spectrum, obtained without relying
        on triangularity, and the closed forms are kept as a cross-check (see
        :meth:`_closed_form_eigenvalues`).
        """
        c, lam = self.c, self.l
        c0 = lam * (b_mat - dt_mat).T
        c1 = -(lam * np.eye(c) - dt_mat).T
        companion = np.block([[np.zeros((c, c)), np.eye(c)], [-c0, -c1]])
        vals, vecs = sla.eig(companion)

        if np.max(np.abs(vals.imag)) > _IMAG_TOL:
            raise ValueError(
                "complex eigenvalues in the quadratic matrix equation; the "
                "mixture-of-exponentials representation does not apply to these parameters"
            )
        vals = vals.real
        # phi is the top half of each companion eigenvector, as a ROW vector.
        phis = vecs[:c, :].real.T

        order = np.argsort(vals)
        vals, phis = vals[order], phis[order]
        if vals[c] - vals[c - 1] < _EIGENVALUE_GAP_TOL:
            raise ValueError(
                "the negative and non-negative eigenvalue groups touch; this is the "
                "degenerate case of Remark 2 (lam == c*mu1, or lam == c*(mu1-mu2), or "
                "lam == c*(mu2-mu1)), where a different representation is needed"
            )

        def build(idx):
            # U = Phi^{-1} Theta Phi makes each phi_i a LEFT eigenvector of U,
            # which is the side the row-vector function F is multiplied on.
            phi = phis[idx]
            return np.linalg.solve(phi, np.diag(vals[idx]) @ phi)

        return build(slice(0, c)), build(slice(c, 2 * c)), vals

    def _closed_form_eigenvalues(self, which: int) -> np.ndarray:
        """
        The paper's explicit roots -- (20)/(21) for ``which == 1`` and (24)/(25)
        for ``which == 2``. Used only to cross-check :meth:`_solve_u`, which
        obtains the same numbers by a completely different route.
        """
        c, lam, mu1, mu2 = self.c, self.l, self.mu1, self.mu2
        lower, upper = [], []
        for i in range(c):
            if which == 1:
                s = lam - (i + 1) * mu1 - (c - 1 - i) * mu2
                disc = np.sqrt(s * s + 4 * (c - 1 - i) * lam * mu2)
            else:
                s = lam - i * mu1 - (c - i) * mu2
                disc = np.sqrt(s * s + 4 * i * lam * mu1)
            lower.append(0.5 * (s - disc))
            upper.append(0.5 * (s + disc))
        return np.sort(np.array(lower + upper))

    # ------------------------------------------------------------------- solution
    def _solve(self) -> dict:  # pylint: disable=too-many-locals,too-many-statements
        """Solve the paper's matching/boundary equations in a bounded basis."""
        if self._sol is not None:
            return self._sol
        self._check_if_servers_and_sources_set()
        c, lam, mu1, mu2, k = self.c, self.l, self.mu1, self.mu2, self.k
        if lam >= c * mu2:
            raise ValueError(
                f"System is unstable: lam={lam} must be < c*mu2={c * mu2}. "
                "Stability is governed by mu2 alone -- however fast mu1 is, a long "
                "enough queue puts every customer past the threshold."
            )

        eye = np.eye(c)
        b1, b2, deltas = self._blocks()
        delta_last = deltas[c - 1]
        dt1 = mu1 * eye + np.linalg.solve(b1, delta_last @ b1)
        dt2 = mu2 * eye + np.linalg.solve(b2, delta_last @ b2)

        u1m, u1p, vals1 = self._solve_u(b1, dt1)
        u2m, _u2p, vals2 = self._solve_u(b2, dt2)
        for vals, which in ((vals1, 1), (vals2, 2)):
            ref = self._closed_form_eigenvalues(which)
            if not np.allclose(vals, ref, rtol=1e-7, atol=1e-9):
                raise RuntimeError(
                    f"eigenvalues of the quadratic matrix equation (kappa={which}) disagree "
                    f"with the paper's closed form: {vals} vs {ref}"
                )

        phi_star = np.zeros(c)
        phi_star[-1] = 1.0  # (22): left null vector of lam(B1 - Dt1)
        psi_c = np.zeros(c)
        psi_c[0] = 1.0  # (26): left null vector of lam(B2 - Dt2)

        if abs(mu2 - mu1 - lam / c) < _RESONANCE_TOL * max(mu1, mu2):
            # Above the threshold the particular solution decays at rate c*mu1
            # while the homogeneous modes decay at the rates -beta_i. Those
            # coincide exactly on this surface, and the representation of
            # Theorem 3 -- a particular solution plus a separate homogeneous
            # mixture -- degenerates (M2 of equation (34) is then singular).
            # The paper's Remark 2 lists the eigenvalue coincidences but not
            # this one; it is visible in their own c=1 formula of Section 5.1,
            # where mu2 - mu1 - lam sits in a denominator.
            raise ValueError(
                f"degenerate case mu2 == mu1 + lam/c (mu1={mu1}, mu2={mu2}, lam={lam}, c={c}): "
                "the particular solution above the threshold resonates with a homogeneous mode, "
                "so the mixture-of-exponentials representation breaks down. Perturb one rate "
                "slightly -- the model itself is perfectly well behaved there, only this "
                "representation of it is not."
            )

        m0 = np.linalg.inv(lam * (b1 - dt1) + np.diag(phi_star))  # (31)
        m1 = np.linalg.inv(lam * (b2 - dt2) + np.diag(psi_c))  # (33)
        m2 = np.linalg.inv((c * mu1 + lam) * (c * mu1 * eye - dt2) + lam * b2)  # (34)

        # Anchor the non-negative modes at k, not zero. Forward exp(U1p*k)
        # amplifies round-off exponentially for large thresholds. This is the
        # same Theorem 3 solution with rescaled coefficients:
        # F(x) = a exp(U1m*x) + b exp(U1p*(x-k)) + alpha0 M0.
        exp_1m = sla.expm(u1m * k)
        exp_back_1p = sla.expm(-u1p * k)
        boundary_maps, boundary_rate = self._boundary_maps(deltas)
        cross = b1 @ np.linalg.solve(dt1, dt2) - b2
        dt_gap_m2 = (dt1 - dt2) @ m2
        b2_dt2_inv = b2 @ np.linalg.inv(dt2)

        def components(unknown):
            a, b, delta = np.split(unknown[: 3 * c], 3)
            bc = unknown[-1]
            fp0 = delta @ boundary_rate
            al0 = fp0 @ dt1 - lam * delta @ b1
            fk = a @ exp_1m + b + al0 @ m0
            fpk = a @ u1m @ exp_1m + b @ u1p
            al1 = al0 @ np.linalg.solve(dt1, dt2) - lam * fk @ cross
            al2 = al1 @ np.linalg.inv(dt2) - fpk + lam * fk @ (eye - b2_dt2_inv)
            finf = bc * psi_c + al1 @ m1
            return a, b, delta, bc, fp0, al0, fk, fpk, al1, al2, finf

        def equations(unknown):
            a, b, delta, _bc, fp0, al0, fk, fpk, _al1, al2, finf = components(unknown)
            # F(0)=0; boundary flux F'(0); continuity of F' at k;
            # total probability. F itself is continuous by construction.
            at_zero = a + b @ exp_back_1p + al0 @ m0
            flux = a @ u1m + b @ u1p @ exp_back_1p - fp0
            matching = (fk - finf - al2 @ dt_gap_m2) @ u2m - al2 @ dt1 @ dt_gap_m2 - fpk
            mass = sum(float((delta @ mapping).sum()) for mapping in boundary_maps) + float(finf.sum())
            return np.concatenate((at_zero, flux, matching, [mass]))

        basis = np.eye(3 * c + 1)
        matrix = np.column_stack([equations(column) for column in basis])
        target = np.zeros(3 * c + 1)
        target[-1] = 1.0
        unknown = np.linalg.solve(matrix, target)
        lower_a, lower_b, delta_cm1, b_c, f_prime0, alpha0, f_k, _f_prime_k, alpha1, alpha2, f_inf = components(unknown)
        deltas_prob = [delta_cm1 @ mapping for mapping in boundary_maps]

        self._sol = {
            "u1m": u1m,
            "u1p": u1p,
            "u2m": u2m,
            "m0": m0,
            "m1": m1,
            "m2": m2,
            "dt1": dt1,
            "dt2": dt2,
            "b_c": b_c,
            "psi_c": psi_c,
            "deltas_prob": deltas_prob,
            "f_prime0": f_prime0,
            "f_k": f_k,
            "f_inf": f_inf,
            "alpha0": alpha0,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "lower_a": lower_a,
            "lower_b": lower_b,
        }
        return self._sol

    def _boundary_maps(self, deltas):
        """Boundary balance maps delta_n = delta_(c-1) H_n, equations (78)-(79).

        The last balance equation gives F'(0) = delta_(c-1) R. Keeping this
        flux relation explicit avoids the exponentially growing H matrices in
        the eliminated form of Theorem 6.
        """
        c, lam = self.c, self.l
        c_hats = []
        prev_shifted = np.zeros((c, c))
        for n in range(c - 1):
            b_hat = np.zeros((n + 2, n + 1))
            for i in range(n + 1):
                b_hat[i, i] = (n - i + 1) * self.mu2
            for i in range(1, n + 2):
                b_hat[i, i - 1] = i * self.mu1
            if n == 0:
                current = b_hat / lam
            else:
                current = b_hat @ np.linalg.inv(lam * (np.eye(n + 1) - prev_shifted) + deltas[n])
            c_hats.append(current)
            prev_shifted = current @ np.hstack([np.zeros((n + 1, 1)), np.eye(n + 1)])
        maps = [None] * c
        maps[-1] = np.eye(c)
        for n in range(c - 2, -1, -1):
            maps[n] = maps[n + 1] @ c_hats[n]
        return maps, lam * np.eye(c) + deltas[c - 1] - lam * prev_shifted

    # -------------------------------------------------------------------- results
    def _f_vector(self, x: float) -> np.ndarray:
        """``F(x)`` -- the row vector of ``P(0 < W <= x, S = (i, c-1-i))``, (41)/(42)."""
        s = self._solve()
        if x <= 0:
            return np.zeros(self.c)
        if x <= self.k:
            return (
                s["lower_a"] @ sla.expm(s["u1m"] * x)
                + s["lower_b"] @ sla.expm(s["u1p"] * (x - self.k))
                + s["alpha0"] @ s["m0"]
            )
        y = x - self.k
        exp_2m = sla.expm(s["u2m"] * y)
        const = s["b_c"] * s["psi_c"] + s["alpha1"] @ s["m1"]
        gap_m2 = (s["dt1"] - s["dt2"]) @ s["m2"]
        return (
            s["f_k"] @ exp_2m
            + const @ (np.eye(self.c) - exp_2m)
            - s["alpha2"] @ gap_m2 @ exp_2m
            + s["alpha2"] @ sla.expm(-s["dt1"] * y) @ gap_m2
        )

    def _f_prime_vector(self, x: float) -> np.ndarray:
        """``F'(x)`` -- the VQT density split by server state, (83)/(84)."""
        s = self._solve()
        if x < 0:
            raise ValueError(f"x must be non-negative, got {x}")
        if x <= self.k:
            return s["lower_a"] @ s["u1m"] @ sla.expm(s["u1m"] * x) + s["lower_b"] @ s["u1p"] @ sla.expm(
                s["u1p"] * (x - self.k)
            )
        y = x - self.k
        gap_m2 = (s["dt1"] - s["dt2"]) @ s["m2"]
        const = s["f_k"] - s["b_c"] * s["psi_c"] - s["alpha1"] @ s["m1"] - s["alpha2"] @ gap_m2
        return const @ s["u2m"] @ sla.expm(s["u2m"] * y) - s["alpha2"] @ s["dt1"] @ sla.expm(-s["dt1"] * y) @ gap_m2

    def get_p0_wait(self) -> float:
        """``P(W = 0)`` -- the probability of finding a free server."""
        s = self._solve()
        return float(sum(d.sum() for d in s["deltas_prob"]))

    def get_cdf(self, x: float) -> float:
        """``P(W <= x)``, exact."""
        if x < 0:
            return 0.0
        return self.get_p0_wait() + float(self._f_vector(x).sum())

    def get_tail(self, x: float) -> float:
        """``P(W > x)`` -- the probability of missing a deadline ``x``."""
        return 1.0 - self.get_cdf(x)

    def get_pdf(self, x: float) -> float:
        """Density of the waiting time on ``x > 0`` (there is an atom at ``x = 0``)."""
        return float(self._f_prime_vector(x).sum())

    def get_server_state_probs(self) -> dict[tuple[int, int], float]:
        """
        ``pi(i, j)`` -- probability of zero delay with ``i`` servers committed to
        class-1 customers and ``j`` to class-2 ones, ``0 <= i + j <= c - 1``.
        """
        s = self._solve()
        return {(j, n - j): float(s["deltas_prob"][n][j]) for n in range(self.c) for j in range(n + 1)}

    def get_class1_prob(self) -> float:
        """Fraction of customers served at the fast/slow rate ``mu1``, i.e. ``P(W <= k)``."""
        return self.get_cdf(self.k)

    def get_w(self) -> list[float]:
        """Mean waiting time ``E[W]``, exact -- Lemma 2 of the paper, equation (47)."""
        s = self._solve()
        k, ones = self.k, np.ones(self.c)
        eye = np.eye(self.c)

        below = s["lower_a"] @ _integral_x_exp(0.0, k, s["u1m"]) @ ones
        # Integration by parts with the positive modes anchored at k:
        # int_0^k x U exp(U(x-k)) dx = k I + int_0^(-k) exp(Uy) dy.
        below += s["lower_b"] @ (k * eye + _integral_exp(s["u1p"], -k)) @ ones

        gap_m2 = (s["dt1"] - s["dt2"]) @ s["m2"]
        const = s["f_k"] - s["b_c"] * s["psi_c"] - s["alpha1"] @ s["m1"] - s["alpha2"] @ gap_m2
        above = const @ (np.linalg.inv(s["u2m"]) - k * eye) @ ones
        above -= s["alpha2"] @ (np.linalg.inv(s["dt1"]) + k * eye) @ gap_m2 @ ones

        self.w = [float(below + above)]
        return self.w

    def get_service_time_mean(self) -> float:
        """
        ``E[S]``, the mean service time actually realised. By PASTA an arrival is
        class 1 exactly when the VQT it observes is at most ``k``, so
        ``E[S] = P(W <= k)/mu1 + P(W > k)/mu2``.
        """
        p1 = self.get_class1_prob()
        return p1 / self.mu1 + (1.0 - p1) / self.mu2

    def get_v(self) -> list[float]:
        """Mean sojourn time ``E[V] = E[W] + E[S]`` (not in the paper; see above)."""
        w = self.w if self.w is not None else self.get_w()
        self.v = [w[0] + self.get_service_time_mean()]
        return self.v

    def run(self) -> QueueResults:
        """Solve and report the standard metrics."""
        start = self._measure_time()
        with self._validate_state():
            w = self.get_w()
            v = self.get_v()
            # Fraction of servers busy: by Little's law on the service facility.
            utilization = self.l * self.get_service_time_mean() / self.c
        result = QueueResults(v=v, w=w, p=None, utilization=utilization)
        self._set_duration(result, start)
        return result
