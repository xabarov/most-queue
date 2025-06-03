import numpy as np

from most_queue.theory.fifo.mmnr import MMnrCalc


def v1_on_utilization_approx(channels: int, arrival_rate: float, deg: int = 3):
    """
    Find cubic approximation, where xs are number of utilizations and ys are values of V1. 
    """

    utilizations = np.linspace(0.1, 0.95, 20)
    arrival_rate = 1

    v1_array = []

    for u in utilizations:
        # Calculate arrival rate based on utilization
        mu = arrival_rate / (u*channels)
        mmnr_calc = MMnrCalc(l=arrival_rate, mu=mu, n=channels, r=100)
        v1_array.append(mmnr_calc.get_v()[0])

    xs = np.array(utilizations)
    ys = np.array(v1_array)

    # Fit a cubic model to the data
    coefficients = np.polyfit(xs, ys, deg=deg)

    # Create a polynomial function with the coefficients
    poly = np.poly1d(coefficients)

    return poly


def find_delta_utilization(poly1: callable, poly2: callable,
                           load1: float, load2: float) -> float:
    """
    Find the amount of load that needs to be removed from load1 to minimize
    the sum of V1 and V2
    """
    v1 = poly1(load1)
    v2 = poly2(load2)

    delta_load = min(1-load1, 1-load2)
    min_u = 0
    minv1v2 = 1e10
    utilaztions = np.linspace(delta_load, 0.01, 100)

    for u in utilaztions:
        if v1 > v2:
            v1 = poly1(load1-u)
            v2 = poly2(load2+u)
        else:
            v1 = poly1(load1+u)
            v2 = poly2(load2-u)
        if (v1 + v2) < minv1v2:
            min_u = u
            minv1v2 = (v1 + v2)

    if v1 <= v2:
        min_u = -min_u

    return min_u