"""
Testing the Takahasi-Takami method for calculating an M/H2/n queue

When the coefficient of variation of service time is less than 1,
the parameters of the approximating H2 distribution
are complex, which does not prevent obtaining meaningful results.

For verification, simulation is used.

"""

from most_queue.general.distribution_fitting import \
    gamma_moments_by_mean_and_coev
from most_queue.general.tables import print_mrx
from most_queue.theory.fifo.mgn_takahasi import MGnCalc

ARRIVAL_RATE = 1.0
NUM_OF_JOBS = 300000
NUM_OF_CHANNELS = 3
UTILIZATION = 0.7
SERVICE_TIME_CV = 1.2


def testcalc_up_probs():
    """
    Testing the Takahasi-Takami method calculation of up-probabilities
    """

    # calculate initial moments of service time based
    # on the given average and coefficient of variation
    b1 = NUM_OF_CHANNELS * UTILIZATION / ARRIVAL_RATE  # average service time
    b = gamma_moments_by_mean_and_coev(b1, SERVICE_TIME_CV)

    # run Takahasi-Takami method
    tt = MGnCalc(NUM_OF_CHANNELS, ARRIVAL_RATE, b)
    tt.run()

    for i in range(1, NUM_OF_CHANNELS + 2):
        probs_mrx = tt.calc_up_probs(i)

        print_mrx(probs_mrx)


if __name__ == "__main__":
    testcalc_up_probs()
