"""
Test for ForkJoin queue with delta.
"""
import numpy as np

from most_queue.general.tables import times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.fork_join_delta import ForkJoinSimDelta
from most_queue.theory.fork_join.split_join import SplitJoinCalc

NUMBER_OF_JOBS = 100000
NUM_OF_CHANNELS = 3
SERVICE_TIME_AVERAGE = 0.35
SERVICE_TIME_DELTA_AVERAGE = 0.1
SERVICE_TIME_CV = [0.73, 1.2]  # Coefficient of variation for service time

ARRIVAL_RATE = 1.0


def test_fj_delta():
    """
    Test for ForkJoin queue with delta.
    """
    for coev in SERVICE_TIME_CV:

        b_params = GammaDistribution.get_params_by_mean_and_coev(
            SERVICE_TIME_AVERAGE, coev)

        delta_params = GammaDistribution.get_params_by_mean_and_coev(
            SERVICE_TIME_DELTA_AVERAGE, coev)
        b_delta = GammaDistribution.calc_theory_moments(delta_params)
        b = GammaDistribution.calc_theory_moments(b_params)

        sj_delta = SplitJoinCalc(ARRIVAL_RATE, NUM_OF_CHANNELS, b, approximation='gamma')

        v_num = sj_delta.get_v_delta(b_delta)
        ro = sj_delta.get_ro()

        qs = ForkJoinSimDelta(NUM_OF_CHANNELS, NUM_OF_CHANNELS, b_delta, True)

        qs.set_sources(ARRIVAL_RATE, 'M')
        qs.set_servers(b_params, 'Gamma')
        qs.run(NUMBER_OF_JOBS)
        v_im = qs.v

        

        print("\n")
        print("-" * 60)
        print(f"{'Split-Join QS with service start delay':^60s}")
        print("-" * 60)
        print(f"Coefficient of variation of service time: {coev}")
        print(
            f"Average delay before service start: {SERVICE_TIME_DELTA_AVERAGE:.3f}")
        print(f"Coefficient of variation of delay: {coev:.3f}")
        print(f"Utilization coefficient: {ro:.3f}")

        times_print(v_im, v_num, is_w=False)

        assert len(v_im) == len(v_num)
        assert np.allclose(np.array(v_im[:1]), np.array(v_num[:1]), rtol=3)


if __name__ == "__main__":

    test_fj_delta()
