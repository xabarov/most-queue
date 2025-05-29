"""
Testing the numerical calculation of the multichannel Ek/D/n system
with deterministic service

For calling - use the EkDn class of the most_queue.theory.ek_d_n_calc package
For verification, we use simulation modeling (sim).
"""
from most_queue.general.tables import probs_print
from most_queue.rand_distribution import ErlangDistribution
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.ek_d_n import EkDn

UTILIZATION_FACTOR = 0.7
NUM_OF_JOBS = 300000
CHANNELS_NUM = 2
ARRIVAL_TIME_AVERAGE = 1.0
ARRIVAL_TIME_CV = 0.56  # coefficient of variation (CV) for arrival time

ERROR_MSG = "The difference between theoretical and simulation results is too large"


def test_ek_d_n():
    """
    Testing the numerical calculation of the multichannel Ek/D/n system
    with deterministic service
    """

    # When creating an instance of the EkDn class, you need to pass 3 parameters:
    # - erlang_params: dataclass ErlangParams - Erlang distribution parameters of the input stream
    #     ErlangParams has two fields, r and mu:
    # - b - service time (deterministic)
    # - n - number of service channels

    # Let us select the ErlangParams
    # based on the mean value and the coefficient of variation
    # using the ErlangDistribution.get_params_by_mean_and_coev() method

    erl_params = ErlangDistribution.get_params_by_mean_and_coev(
        ARRIVAL_TIME_AVERAGE, ARRIVAL_TIME_CV)

    # service time will be determined based on the specified utilization factor

    b = ARRIVAL_TIME_AVERAGE * CHANNELS_NUM * UTILIZATION_FACTOR

    # create an instance of the class for numerical calculation
    ekdn = EkDn(erl_params, b, CHANNELS_NUM)

    # start calculating the probabilities of the QS states
    p_num = ekdn.calc_p()

    # for verification we use simulation.
    # create an instance of the QsSim class, pass the number of service channels
    qs = QsSim(CHANNELS_NUM)

    # we set the input stream. The method needs to be passed
    # the distribution parameters as a list and the distribution type. E - Erlang
    qs.set_sources(erl_params, "E")
    # we set the service channels. The input is the service time and the distribution type - D.
    qs.set_servers(b, "D")

    # run simulation
    qs.run(NUM_OF_JOBS)

    # obtain parameters - initial moments (3) of sojourn time
    # and probability distribution of the system state
    v_sim = qs.v

    print(f'v_sim: {v_sim}')
    p_sim = qs.get_p()

    probs_print(p_sim, p_num, 10)

    # probs of zero jobs in queue are 0.084411 | 0.084...

    assert abs(p_sim[0] - p_num[0]) < 0.01, ERROR_MSG


if __name__ == "__main__":
    test_ek_d_n()
