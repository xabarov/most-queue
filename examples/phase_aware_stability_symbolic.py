"""Optional SymPy reproduction of the preregistered EPIC-096 identities.

Run python -m examples.phase_aware_stability_symbolic (requires SymPy).
This is a symbolic check, not a replacement for the full drift theorem.
"""

import sympy as sp


def run():
    """Check periodic corrector, Poisson second moment and area potential."""
    rate, d0, d1, v0, v1 = sp.symbols("lambda d0 d1 v0 v1")
    g0, g1 = rate * d0 - v0, rate * d1 - v1
    delta = (g0 + g1) / 2
    b1 = (g1 - g0) / 2
    assert sp.simplify(g0 + b1 - delta) == 0
    assert sp.simplify(g1 - b1 - delta) == 0
    y, mu, shift = sp.symbols("Y mu r")
    # For J~Poisson(mu), E[J]=mu and E[J^2]=mu^2+mu.
    increment = 2 * y * (mu + shift) + mu + (mu + shift) ** 2
    direct = 2 * y * (mu + shift) + mu**2 + mu + 2 * shift * mu + shift**2
    assert sp.simplify(increment - direct) == 0
    eps, m2 = sp.symbols("epsilon M2", positive=True)
    potential_drift = ((-2 * eps * y + m2) + m2 / eps * (-eps)) / (2 * eps)
    assert sp.simplify(potential_drift + y) == 0
    print("Periodic corrector, Poisson quadratic increment and exterior area identity: PASS")


if __name__ == "__main__":
    run()
