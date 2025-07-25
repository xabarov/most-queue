"""
Converts raw moments to central moments.
"""


def convert_raw_to_central(raw_moments: list[float] | float):
    """
    Converts raw moments to central moments.
    """

    if isinstance(raw_moments, float):
        return [raw_moments]

    n = len(raw_moments)

    if n == 0:
        return []

    central_moments = [0.0] * n

    # The first central moment is always 0 (mean - mean),
    # so we return mean as the first moment.
    if n > 0:
        # Variance
        central_moments[0] = raw_moments[0]

    if n > 1:
        mu1 = raw_moments[0]  # E[X]
        mu2 = raw_moments[1]  # E[X^2]
        central_moments[1] = mu2 - mu1**2  # Var(X) = E[X^2] - (E[X])^2

        if n > 2:
            # Skewness
            mu3 = raw_moments[2]  # E[X^3]
            central_moments[2] = mu3 - 3 * mu1 * mu2 + 2 * mu1**3

            if n > 3:
                # Kurtosis
                mu4 = raw_moments[3]  # E[X^4]
                central_moments[3] = mu4 - 4 * mu1 * mu3 + 6 * mu1**2 * mu2 - 3 * mu1**4

    return central_moments
