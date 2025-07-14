"""
Parameters for queueing system calculation
"""

from dataclasses import dataclass


@dataclass
class CalcParams:
    """
    Parameters for queueing system calculation
    """

    e: float = 1e-10  # tolerance for convergence
    approx_distr: str = "Gamma"  # distribution approximation method
    p_num: int = 100  # number of probabilities to calculate


@dataclass
class TakahashiTakamiParams:
    """Parameters for the Takahashi-Takami method."""

    N: int = 150
    accuracy: float = 1e-8
    dtype: str = "c16"
    verbose: bool = False
    max_iter: int = 300
    is_cox: bool = True
    approx_ee: float = 0.1
    approx_e: float = 0.5
    is_fitting: bool = True
    stable_w_pls: bool = False
    w_pls_dt: float = 1e-3
