"""Prespecified parameter variations for step 3."""


def cases():
    specifications = [
        ("remove", "rcs", 2, 0.45, 0.8, 0.3),
        ("remove", "rcs", 3, 0.85, 1.2, 0.6),
        ("remove", "disaster", 2, 0.9, 0.8, 0.15),
        ("remove", "disaster", 4, 1.2, 1.2, 0.3),
        ("restart", "rcs", 2, 0.5, 0.8, 0.6),
        ("restart", "rcs", 4, 0.85, 1.2, 0.6),
        ("restart", "disaster", 2, 0.4, 0.8, 0.6),
        ("restart", "disaster", 3, 0.85, 1.2, 0.6),
        ("fixed", "rcs", 2, 0.5, 0.8, 0.04),
        ("fixed", "rcs", 3, 0.85, 1.2, 0.04),
        ("fixed", "disaster", 2, 0.5, 0.8, 0.04),
        ("fixed", "disaster", 3, 0.8, 1.2, 0.04),
    ]
    return [
        dict(id=f"case{i:02d}", action=a, mode=m, n=n, rho=rho, cv=cv, delta=d, mean=2.1, lam=rho * n / 2.1)
        for i, (a, m, n, rho, cv, d) in enumerate(specifications)
    ]
