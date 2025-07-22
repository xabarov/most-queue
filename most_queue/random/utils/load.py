"""
Calculates the utilization (load factor) of the QS
"""


def calc_qs_load(source_kendall_notation: str, source_params, server_kendall_notation: str, server_params, n) -> float:
    """Calculates the utilization (load factor) of the QS

    Args:
        source_kendall_notation: str,  Kendall notation of source
        server_kendall_notation: str, Kendall notation of source
        n (int): number of QS channels

    Returns:
        float: utilization (load factor) of the QS
    """

    l = 0
    if source_kendall_notation == "M":
        l = source_params
    elif source_kendall_notation == "D":
        l = 1.00 / source_params
    elif source_kendall_notation == "Uniform":
        l = 1.00 / source_params.mean
    elif source_kendall_notation == "H":
        y1 = source_params.p1
        y2 = 1.0 - y1
        mu1 = source_params.mu1
        mu2 = source_params.mu2

        f1 = y1 / mu1 + y2 / mu2
        l = 1.0 / f1

    elif source_kendall_notation == "E":
        r = source_params.r
        mu = source_params.mu
        l = mu / r

    elif source_kendall_notation == "Gamma":
        mu = source_params.mu
        alpha = source_params.alpha
        l = mu / alpha

    elif source_kendall_notation == "C":
        y1 = source_params.p1
        y2 = 1.0 - y1
        mu1 = source_params.mu1
        mu2 = source_params.mu2

        f1 = y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
        l = 1.0 / f1
    elif source_kendall_notation == "Pa":
        if source_params[0] < 1:
            return None

        a = source_params.alpha
        k = source_params.K
        f1 = a * k / (a - 1)
        l = 1.0 / f1

    b1 = 0
    if server_kendall_notation == "M":
        mu = server_params
        b1 = 1.0 / mu
    elif server_kendall_notation == "D":
        b1 = server_params
    elif server_kendall_notation == "Uniform":
        b1 = server_params.mean

    elif server_kendall_notation == "H":
        y1 = server_params.p1
        y2 = 1.0 - y1
        mu1 = server_params.mu1
        mu2 = server_params.mu2

        b1 = y1 / mu1 + y2 / mu2

    elif server_kendall_notation == "Gamma":
        mu = server_params.mu
        alpha = server_params.alpha
        b1 = alpha / mu

    elif server_kendall_notation == "E":
        r = server_params.r
        mu = server_params.mu
        b1 = r / mu

    elif server_kendall_notation == "C":
        y1 = server_params.p1
        y2 = 1.0 - y1
        mu1 = server_params.mu1
        mu2 = server_params.mu2

        b1 = y2 / mu1 + y1 * (1.0 / mu1 + 1.0 / mu2)
    elif server_kendall_notation == "Pa":
        a = server_params.alpha
        k = server_params.K
        b1 = a * k / (a - 1)

    return l * b1 / n
