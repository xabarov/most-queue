"""
Quasi-birth-death (QBD) process solver.

A QBD is a CTMC on levels 0, 1, 2, ... with phase structure: transitions go
one level up (blocks A0), stay (A1) or one level down (A2), the same for all
levels >= 1; level 0 is a boundary with its own blocks. The stationary
distribution is matrix-geometric: pi_k = pi_1 R^{k-1} for k >= 1.

The G matrix is computed by the logarithmic reduction algorithm of
Latouche & Ramaswami (quadratic convergence), then R = A0 (-A1 - A0 G)^{-1}.

References:
    Latouche G., Ramaswami V. A logarithmic reduction algorithm for
        quasi-birth-death processes. J. Appl. Probab., 30(3), 1993.
        doi:10.2307/3214773.
    Latouche G., Ramaswami V. Introduction to Matrix Analytic Methods in
        Stochastic Modeling. SIAM, 1999. doi:10.1137/1.9780898719734.
"""

import numpy as np


def logarithmic_reduction_g(
    a0: np.ndarray, a1: np.ndarray, a2: np.ndarray, tol: float = 1e-12, max_iter: int = 100
) -> np.ndarray:
    """
    G matrix of the QBD with repeating blocks (a0 up, a1 local, a2 down):
    the minimal non-negative solution of A2 + A1 G + A0 G^2 = 0.

    :return: stochastic matrix G (row sums 1 for a positive recurrent QBD)
    """
    m = a1.shape[0]
    inv_a1 = np.linalg.inv(-a1)
    b0 = inv_a1 @ a0
    b2 = inv_a1 @ a2

    g = b2.copy()
    t = b0.copy()
    for _ in range(max_iter):
        u = b0 @ b2 + b2 @ b0
        h = np.linalg.inv(np.eye(m) - u)
        b0 = h @ b0 @ b0
        b2 = h @ b2 @ b2
        g += t @ b2
        t = t @ b0
        if np.max(np.abs(1.0 - g.sum(axis=1))) < tol:
            break
    return g


class QBDSolver:
    """
    Stationary distribution of a QBD with a boundary level of (possibly)
    different dimension.

    Generator layout::

        [ B00  B01   0    0  ... ]
        [ B10  A1    A0   0  ... ]
        [ 0    A2    A1   A0 ... ]
        [ 0    0     A2   A1 ... ]

    Balance equations solved for (pi0, pi1); pi_k = pi_1 R^{k-1} for k >= 1,
    normalization pi0 @ 1 + pi1 (I - R)^{-1} @ 1 = 1.
    """

    def __init__(
        self,
        a0: np.ndarray,
        a1: np.ndarray,
        a2: np.ndarray,
        b00: np.ndarray,
        b01: np.ndarray,
        b10: np.ndarray,
        tol: float = 1e-12,
    ):  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.a0 = np.asarray(a0, dtype=float)
        self.a1 = np.asarray(a1, dtype=float)
        self.a2 = np.asarray(a2, dtype=float)
        self.b00 = np.asarray(b00, dtype=float)
        self.b01 = np.asarray(b01, dtype=float)
        self.b10 = np.asarray(b10, dtype=float)
        self.tol = tol

        self.g = None
        self.r = None
        self.pi0 = None
        self.pi1 = None

    def solve(self) -> None:
        """
        Compute G, R and the boundary vectors (pi0, pi1).

        Raises:
            ValueError: if the QBD is not positive recurrent (sp(R) >= 1).
        """
        self.g = logarithmic_reduction_g(self.a0, self.a1, self.a2, tol=self.tol)
        self.r = self.a0 @ np.linalg.inv(-(self.a1 + self.a0 @ self.g))

        spectral_radius = max(abs(np.linalg.eigvals(self.r)))
        if spectral_radius >= 1.0 - 1e-10:
            raise ValueError(f"QBD is not positive recurrent: sp(R) = {spectral_radius:.6f} must be < 1")

        m0 = self.b00.shape[0]
        m = self.a1.shape[0]
        # unknowns x = [pi0, pi1]; equations:
        # pi0 B00 + pi1 B10 = 0
        # pi0 B01 + pi1 (A1 + R A2) = 0
        a_sys = np.zeros((m0 + m, m0 + m))
        a_sys[:m0, :m0] = self.b00
        a_sys[m0:, :m0] = self.b10
        a_sys[:m0, m0:] = self.b01
        a_sys[m0:, m0:] = self.a1 + self.r @ self.a2
        # replace the last equation with normalization
        norm_col = np.concatenate([np.ones(m0), np.linalg.inv(np.eye(m) - self.r) @ np.ones(m)])
        a_full = a_sys.T.copy()
        a_full[-1, :] = norm_col
        rhs = np.zeros(m0 + m)
        rhs[-1] = 1.0
        x = np.linalg.solve(a_full, rhs)

        self.pi0 = x[:m0]
        self.pi1 = x[m0:]

    def level_probs(self, num_levels: int) -> list[np.ndarray]:
        """
        Stationary phase vectors of levels 0..num_levels-1
        (pi_k = pi_1 R^{k-1} for k >= 1).
        """
        if self.pi1 is None:
            self.solve()
        levels = [self.pi0]
        vec = self.pi1.copy()
        for _ in range(1, num_levels):
            levels.append(vec)
            vec = vec @ self.r
        return levels

    def marginal_level_probs(self, num_levels: int) -> list[float]:
        """Total probability of each level (sum over phases)."""
        return [float(np.sum(v)) for v in self.level_probs(num_levels)]

    def residual(self) -> float:
        """Max abs residual of A2 + A1 G + A0 G^2 = 0 (quality check)."""
        if self.g is None:
            self.solve()
        return float(np.max(np.abs(self.a2 + self.a1 @ self.g + self.a0 @ self.g @ self.g)))
