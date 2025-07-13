"""
Test for ForkJoin queue with delta.
"""
import os

import numpy as np
import yaml

from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.fork_join_delta import ForkJoinSimDelta
from most_queue.theory.fork_join.split_join import SplitJoinCalc

cur_dir = os.getcwd()
params_path = os.path.join(cur_dir, 'tests', 'default_params.yaml')

with open(params_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)

# Import constants from params file
NUM_OF_CHANNELS = int(params['num_of_channels'])

ARRIVAL_RATE = float(params['arrival']['rate'])

SERVICE_TIME_CV = float(params['service']['cv'])

NUM_OF_JOBS = int(params['num_of_jobs'])
UTILIZATION_FACTOR = float(params['utilization_factor'])
ERROR_MSG = params['error_msg']

PROBS_ATOL = float(params['probs_atol'])
PROBS_RTOL = float(params['probs_rtol'])

MOMENTS_ATOL = float(params['moments_atol'])
MOMENTS_RTOL = float(params['moments_rtol'])

SERVICE_TIME_AVERAGE = 0.35
SERVICE_TIME_DELTA_AVERAGE = 0.1


def test_fj_delta():
    """
    Test for ForkJoin queue with delta.
    """

    b_params = GammaDistribution.get_params_by_mean_and_coev(
        SERVICE_TIME_AVERAGE, SERVICE_TIME_CV)

    delta_params = GammaDistribution.get_params_by_mean_and_coev(
        SERVICE_TIME_DELTA_AVERAGE, SERVICE_TIME_CV)
    b_delta = GammaDistribution.calc_theory_moments(delta_params)
    b = GammaDistribution.calc_theory_moments(b_params)

    sj_delta = SplitJoinCalc(
        ARRIVAL_RATE, NUM_OF_CHANNELS, b, approximation='gamma')

    v_num = sj_delta.get_v_delta(b_delta)
    ro = sj_delta.get_ro()

    qs = ForkJoinSimDelta(NUM_OF_CHANNELS, NUM_OF_CHANNELS, b_delta, True)

    qs.set_sources(ARRIVAL_RATE, 'M')
    qs.set_servers(b_params, 'Gamma')
    qs.run(NUM_OF_JOBS)
    v_sim = qs.v

    print("-" * 60)
    print(f"{'Split-Join QS with service start delay':^60s}")
    print("-" * 60)
    print(f"Coefficient of variation of service time: {SERVICE_TIME_CV}")
    print(
        f"Average delay before service start: {SERVICE_TIME_DELTA_AVERAGE:.3f}")
    print(f"Coefficient of variation of delay: {SERVICE_TIME_CV:.3f}")
    print(f"Utilization coefficient: {ro:.3f}")

    times_print(v_sim, v_num, is_w=False)

    assert len(v_sim) == len(v_num)
    assert np.allclose(np.array(v_sim), np.array(
        v_num), rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL), ERROR_MSG


if __name__ == "__main__":

    test_fj_delta()
