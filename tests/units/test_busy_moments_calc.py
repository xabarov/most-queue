"""
Test different variants of the busy period calculation for M/G/1 queue.
"""

import math
import time

from most_queue.io.tables import print_raw_moments
from most_queue.random.utils.fit import gamma_moments_by_mean_and_cv
from most_queue.theory.utils.busy_periods import busy_calc, busy_calc_lst


def test_busy_calc_variants():
    """
    Test different variants of the busy period calculation for M/G/1 queue.
    """

    arrival_rate = 0.7  # arrival rate of positive jobs
    ro = 0.7
    b1 = 1 * ro / arrival_rate  # average service time
    b_cv = 0.57

    b = gamma_moments_by_mean_and_cv(b1, b_cv)  # gamma distribution parameters

    start = time.time()
    busy_1 = busy_calc_lst(arrival_rate, b)
    busy1_duration = time.time() - start
    print(f"busy_calc_lst duration: {busy1_duration:.4g}")

    start = time.time()
    busy_2 = busy_calc(arrival_rate, b, num=3)
    busy2_duration = time.time() - start
    print(f"busy_calc duration: {busy2_duration:.4g}")

    print_raw_moments(
        sim_moments=busy_1,
        calc_moments=busy_2,
        header="Busy periods",
        calc_header="Lst",
        sim_header="Num",
    )

    assert math.isclose(busy_1[0], busy_2[0], rel_tol=1e-2), "First moment is not close"


if __name__ == "__main__":
    test_busy_calc_variants()
