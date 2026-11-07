"""Positive phase representations for the original Gamma service distribution.

For shape m+r, prepend m exponential stages to a positive quadrature for
Gamma(r,theta). The latter is an exponential mixture with X~Beta(r,1-r)
and conditional rate 1/(theta*X). Integer shapes give an exact Erlang law.
"""

import numpy as np
from scipy.special import roots_jacobi

from works.negative_queues.revision_v7_20261107.ph_queue import Service


def gamma_ph(mean, cv, quadrature=4):
    shape = 1 / cv**2
    theta = mean / shape
    integer = int(np.floor(shape + 1e-12))
    fractional = shape - integer
    if abs(fractional) < 1e-10:
        size = integer
        alpha = np.zeros(size)
        alpha[0] = 1
        generator = -np.eye(size) / theta
        for i in range(size - 1):
            generator[i, i + 1] = 1 / theta
        return Service(alpha, generator, np.tile(alpha, (size, 1)), [list(range(size))], f"exact Erlang-{size}")
    # scipy evaluates the unused k=1 branch as 0/0 when alpha+beta=-1.
    # The selected recurrence coefficient is finite; validate all outputs below.
    with np.errstate(invalid="ignore", divide="ignore"):
        roots, weights = roots_jacobi(quadrature, -fractional, fractional - 1)
    if not np.isfinite(np.r_[roots, weights]).all() or min(weights) <= 0:
        raise FloatingPointError("Invalid positive quadrature")
    nodes = (roots + 1) / 2
    weights = weights / weights.sum()
    size = integer + quadrature
    alpha = np.zeros(size)
    generator = np.zeros((size, size))
    if integer:
        alpha[0] = 1
        for i in range(integer):
            generator[i, i] = -1 / theta
            if i + 1 < integer:
                generator[i, i + 1] = 1 / theta
            else:
                generator[i, integer:] = weights / theta
    else:
        alpha[:] = weights
    for j, x in enumerate(nodes):
        generator[integer + j, integer + j] = -1 / (theta * x)
    return Service(
        alpha,
        generator,
        np.tile(alpha, (size, 1)),
        [list(range(size))],
        f"positive Gamma PH, J={quadrature}, phases={size}",
    )


def positive_h2_equivalent(y, mu):
    y, mu = np.asarray(y, complex), np.asarray(mu, complex)
    if max(np.max(np.abs(y.imag)), np.max(np.abs(mu.imag))) > 1e-10:
        raise ValueError("A real two-phase representation cannot have these complex poles")
    y, mu = y.real, mu.real
    if min(mu) <= 0 or abs(sum(y) - 1) > 1e-10:
        raise ValueError("Invalid transform parameters")
    if min(y) >= 0:
        return Service(y, -np.diag(mu), np.tile(y, (len(y), 1)), [list(range(len(y)))], "positive H2")
    if len(y) != 2:
        raise ValueError("Coxian equivalence here is only for two poles")
    numerator = float(y @ mu)
    for a, b in [tuple(mu), tuple(mu[::-1])]:
        p = 1 - numerator / a
        if 0 <= p <= 1:
            alpha = np.array([1.0, 0.0])
            generator = np.array([[-a, a * p], [0.0, -b]])
            return Service(alpha, generator, np.tile(alpha, (2, 1)), [[0, 1]], "Coxian-2 equivalent to algebraic H2")
    raise ValueError("No positive Coxian-2 form for this transform")


def transform(service, s):
    return service.alpha @ np.linalg.solve(s * np.eye(service.size) - service.generator, service.exit)
