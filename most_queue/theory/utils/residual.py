import numpy as np


def lst_residual(lst_original: callable, b1: float, s: float) -> float:
    """
    Calculate the Laplace-Stieltjes transform of a residual distribution.
    """
    return (1-lst_original(s))/(b1*s)


def get_b_residual(b_original: list[float]) -> list[float]:
    """
    Calculate the initial moments E[X^k] for k=0,1,...,n-1 of a residual distribution.
    b_res1 = E[X^2]/(2*E[X])
    """
    b_res = np.zeros(len(b_original)-1)
    for i in range(len(b_original)-1):
        b_res[i] = b_original[i+1]/((i+2)*b_original[0])
    return b_res
