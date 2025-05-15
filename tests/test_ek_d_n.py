"""
Testing the numerical calculation of the multichannel Ek/D/n system
with deterministic service

For calling - use the EkDn class of the most_queue.theory.ek_d_n_calc package
For verification, we use simulation modeling (sim).
"""
from most_queue.general.tables import probs_print
from most_queue.rand_distribution import ErlangDistribution
from most_queue.sim.queueing_systems.fifo import QueueingSystemSimulator
from most_queue.theory.queueing_systems.fifo.ek_d_n import EkDn


def test_ek_d_n():
    """
    Testing the numerical calculation of the multichannel Ek/D/n system
    with deterministic service

    For calling - use the EkDn class of the most_queue.theory.ek_d_n_calc package
    For verification, we use simulation modeling (sim).
    """

    ro = 0.7  # utilization factor
    num_of_jobs = 300000  # for sim
    channels_num = 2

    # When creating an instance of the EkDn class, you need to pass 4 parameters:
    # - l , k - Erlang distribution parameters of the input stream
    # - b - service time (deterministic)
    # - n - number of service channels

    # Let us select the parameters k and l of the approximating distribution
    # based on the mean value and the coefficient of variation
    # using the ErlangDistribution.get_params_by_mean_and_coev() method

    a1 = 1  # average time between requests in the input stream
    coev_a = 0.56  # coefficient of variation of input flow
    erl_params = ErlangDistribution.get_params_by_mean_and_coev(a1, coev_a)

    # service time will be determined based on the specified utilization factor
    # In your case, the parameters l, k, b and n can be specified directly.
    # Be sure to check that the load factor of the QS does not exceed 1.0

    b = a1 * channels_num * ro

    # create an instance of the class for numerical calculation
    ekdn = EkDn(erl_params, b, channels_num)

    # start calculating the probabilities of the QS states
    p_ch = ekdn.calc_p()

    # for verification we use IM.
    # create an instance of the IM class, pass the number of service channels
    qs = QueueingSystemSimulator(channels_num)

    # we set the input stream. The method needs to be passed the distribution parameters as a list and the distribution type. E - Erlang
    qs.set_sources(erl_params, "E")
    # we set the service channels. The input is the service time and the distribution type - D.
    qs.set_servers(b, "D")

    # run simulation
    qs.run(num_of_jobs)

    # obtain parameters - initial moments (3) of sojourn time and probability distribution of the system state
    v_sim = qs.v

    print(f'v_sim: {v_sim}')
    p_sim = qs.get_p()

    probs_print(p_sim, p_ch, 10)

    # probs of zero jobs in queue are 0.084411 | 0.084...

    assert abs(p_sim[0] - p_ch[0]) < 0.01


if __name__ == "__main__":
    test_ek_d_n()
