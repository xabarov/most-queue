"""
Testing the Takahasi-Takami method for calculating an M/H2/n queue

When the coefficient of variation of service time is less than 1, 
the parameters of the approximating H2 distribution
are complex, which does not prevent obtaining meaningful results.

For verification, simulation is used.

"""
import math
import time

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.queueing_systems.fifo import QueueingSystemSimulator
from most_queue.theory.queueing_systems.fifo.mgn_takahasi import MGnCalc


def test_mgn_tt():
    """
    Testing the Takahasi-Takami method for calculating an M/H2/n queue
    """

    n = 3  # number of channels
    l = 1.0  # arrival rate
    ro = 0.7  # utilization factor
    b1 = n * ro / l  # average service time
    num_of_jobs = 1000000  # number of jobs for simulation
    # two variants of the coefficient of variation of service time, run calculation and simulation for each
    b_coev_mass = [0.8, 1.2]

    for b_coev in b_coev_mass:
        # calculate initial moments of service time based on the given average and coefficient of variation
        b = [0.0] * 3
        alpha = 1 / (b_coev ** 2)
        b[0] = b1
        b[1] = math.pow(b[0], 2) * (math.pow(b_coev, 2) + 1)
        b[2] = b[1] * b[0] * (1.0 + 2 / alpha)

        tt_start = time.process_time()
        # run Takahasi-Takami method
        tt = MGnCalc(n, l, b)
        tt.run()
        # get numerical calculation results
        p_tt = tt.get_p()
        v_tt = tt.get_v()
        tt_time = time.process_time() - tt_start
        # also can find out how many iterations were required
        num_of_iter = tt.num_of_iter_

        # run simulation for verification of the results
        im_start = time.process_time()

        qs = QueueingSystemSimulator(n)

        # set arrival process. M - exponential with rate l
        qs.set_sources(l, 'M')

        # set server parameters as Gamma distribution.
        # Distribution parameters are selected using the method from the random_distribution library
        gamma_params = GammaDistribution.get_params([b[0], b[1]])
        qs.set_servers(gamma_params, 'Gamma')

        # Run simulation
        qs.run(num_of_jobs)

        # Get results
        p = qs.get_p()
        v_sim = qs.v
        im_time = time.process_time() - im_start

        # print results

        print("\nComparison of calculation results by the Takahasi-Takami method and simulation.")
        print(
            f"Simulation - M/Gamma/{n:^2d}\nTakahasi-Takami - M/H2/{n:^2d} with complex parameters")
        print(f"Utilization factor: {ro:^1.2f}")
        print(f"Coefficient of variation of service time: {b_coev:^1.2f}")
        print(
            f"Number of iterations of the Takahasi-Takami algorithm: {num_of_iter:^4d}")
        print(f"Takahasi-Takami algorithm execution time: {tt_time:^5.3f} s")
        print(f"Simulation execution time: {im_time:^5.3f} s")
        probs_print(p, p_tt, 10)

        times_print(v_sim, v_tt, False)

        assert 100*abs(v_tt[0] - v_sim[0])/max(v_tt[0], v_sim[0]) < 10


if __name__ == "__main__":
    test_mgn_tt()
