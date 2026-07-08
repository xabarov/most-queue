"""
Numerical n-th derivative via central finite differences.

Drop-in replacement for ``scipy.misc.derivative``, which was deprecated in
SciPy 1.10 and removed in later releases. Reproduces the same algorithm
(central difference weights from a Vandermonde system), including support
for complex-valued functions (used by Takahashi-Takami LST differentiation).
"""

import math
from fractions import Fraction

import numpy as np


def central_diff_weights(num_points: int, ndiv: int = 1) -> np.ndarray:
    """
    Weights of a central finite-difference approximation of the ndiv-th
    derivative on num_points equally spaced points.

    The Vandermonde system is solved exactly in rational arithmetic, so the
    weights carry no inversion noise (which the removed scipy implementation
    amplified by dx**(-n) for higher derivatives).

    :param num_points: number of stencil points (odd, > ndiv)
    :param ndiv: derivative order
    """
    if num_points < ndiv + 1:
        raise ValueError("Number of points must be at least the derivative order + 1.")
    if num_points % 2 == 0:
        raise ValueError("The number of points must be odd.")
    ho = num_points >> 1
    # exact Gauss-Jordan inversion of the Vandermonde matrix X[i][j] = x_i^j
    n = num_points
    aug = [
        [Fraction(x**j) for j in range(n)] + [Fraction(int(i == k)) for k in range(n)]
        for i, x in enumerate(range(-ho, ho + 1))
    ]
    for col in range(n):
        pivot_row = next(r for r in range(col, n) if aug[r][col] != 0)
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        aug[col] = [v / pivot for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [v - factor * p for v, p in zip(aug[r], aug[col])]
    inv_row = aug[ndiv][n:]  # row ndiv of X^{-1}, as in scipy's central_diff_weights
    return np.array([float(math.factorial(ndiv) * w) for w in inv_row])


def derivative(
    func, x0, dx=1.0, n=1, args=(), order=3
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    n-th derivative of func at x0 using a central difference formula with
    ``order`` points and spacing ``dx``. Same signature and behaviour as the
    removed ``scipy.misc.derivative``.
    """
    if order < n + 1:
        raise ValueError("'order' (number of stencil points) must be at least the derivative order 'n' + 1.")
    if order % 2 == 0:
        raise ValueError("'order' (the number of points used to compute the derivative) must be odd.")
    # exact rational weights for the common cases (same table as scipy.misc.derivative)
    if n == 1:
        if order == 3:
            weights = np.array([-1, 0, 1]) / 2.0
        elif order == 5:
            weights = np.array([1, -8, 0, 8, -1]) / 12.0
        elif order == 7:
            weights = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
        elif order == 9:
            weights = np.array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0
        else:
            weights = central_diff_weights(order, 1)
    elif n == 2:
        if order == 3:
            weights = np.array([1, -2.0, 1])
        elif order == 5:
            weights = np.array([-1, 16, -30, 16, -1]) / 12.0
        elif order == 7:
            weights = np.array([2, -27, 270, -490, 270, -27, 2]) / 180.0
        elif order == 9:
            weights = np.array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9]) / 5040.0
        else:
            weights = central_diff_weights(order, 2)
    else:
        weights = central_diff_weights(order, n)
    val = 0.0
    ho = order >> 1
    for k in range(order):
        val += weights[k] * func(x0 + (k - ho) * dx, *args)
    return val / np.prod((dx,) * n, axis=0)
