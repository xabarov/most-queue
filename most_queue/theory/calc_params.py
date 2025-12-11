"""
Parameters for queueing system calculation
"""

from dataclasses import dataclass

from most_queue.constants import DEFAULT_P_NUM


@dataclass
class CalcParams:
    """
    Parameters for queueing system calculation
    """

    tolerance: float = 1e-10  # tolerance for convergence
    approx_distr: str = "gamma"  # distribution approximation method
    p_num: int = DEFAULT_P_NUM  # number of probabilities to calculate


@dataclass
class TakahashiTakamiParams(CalcParams):
    """Parameters for the Takahashi-Takami method."""

    N: int = 200
    dtype: str = "c16"
    verbose: bool = False
    max_iter: int = 300
    is_cox: bool = True
    approx_ee: float = 0.1
    approx_e: float = 0.5
    is_fitting: bool = True
    stable_w_pls: bool = False
    w_pls_dt: float = 1e-3
