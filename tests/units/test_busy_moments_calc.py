"""
Test different variants of the busy period calculation for M/G/1 queue.
"""
import math
import time

from most_queue.general.tables import times_print
from most_queue.theory.utils.busy_periods import busy_calc, busy_calc_lst


def test_busy_calc_variants():
    """
    Test different variants of the busy period calculation for M/G/1 queue.
    """

    arrival_rate = 0.7  # arrival rate of positive jobs
    ro = 0.7
    b1 = 1 * ro / arrival_rate  # average service time
    b_coev = 0.57

    b = [0.0] * 3
    alpha = 1 / (b_coev ** 2)
    b[0] = b1
    b[1] = math.pow(b[0], 2) * (math.pow(b_coev, 2) + 1)
    b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

    start = time.time()
    busy_1 = busy_calc_lst(arrival_rate, b)
    busy1_duration = time.time() - start
    print(f'busy_calc_lst duration: {busy1_duration:.4g}')
    
    start = time.time()
    busy_2 = busy_calc(arrival_rate, b, num=3)
    busy2_duration = time.time() - start
    print(f'busy_calc duration: {busy2_duration:.4g}')
    
    times_print(sim_moments=busy_1, calc_moments=busy_2, header="Busy periods", calc_header='Lst', sim_header='Num')
    
    assert math.isclose(busy_1[0], busy_2[0], rel_tol=1e-2), "First moment is not close"
    
if __name__ == "__main__":
    test_busy_calc_variants()
