"""
Testing the simulation model of an M/G/c queue with priorities.
"""
from most_queue.general.tables import times_print_with_classes
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.priority import PriorityQueueSimulator
from most_queue.theory.priority.mgn_invar_approx import MGnInvarApproximation

NUM_OF_CHANNELS = 5
NUM_OF_CLASSES = 3
ARRIVAL_RATES = [0.1, 0.2, 0.3]
SERVICE_TIMES_AVE = [2.25, 4.5, 6.75]
SERVICE_TIME_CV = 1.2
NUM_OF_JOBS = 300_000


def test_sim():
    """
    Testing the simulation model of an M/G/c queue with priorities.
    For verification, comparing results with those calculated using the method of invariant relations:
        Ryzhikov Yu.I., Khomonenko A.D. Calculation of multi-channel service systems with absolute and
        relative priorities based on invariant relations // Intelligent technologies
        for transportation. 2015. №3
    """
    lsum = sum(ARRIVAL_RATES)

    # second initial moments
    b2 = [0] * NUM_OF_CLASSES
    for i in range(NUM_OF_CLASSES):
        b2[i] = (SERVICE_TIMES_AVE[i] ** 2) * (1 + SERVICE_TIME_CV ** 2)

    b_sr = sum(SERVICE_TIMES_AVE) / NUM_OF_CLASSES

    # get the coefficient of load
    ro = lsum * b_sr / NUM_OF_CHANNELS

    # now, given the two initial moments, select parameters for the approximating Gamma distribution
    # and add them to the list of parameters params
    params = []
    for i in range(NUM_OF_CLASSES):
        params.append(GammaDistribution.get_params(
            [SERVICE_TIMES_AVE[i], b2[i]]))

    b = []
    for j in range(NUM_OF_CLASSES):
        b.append(GammaDistribution.calc_theory_moments(
            params[j], 4))

    print("\nComparison of data from the simulation and results calculated using the method of invariant relations (R) \n"
          "time spent in a multi-channel queue with priorities")
    print(f"Number of servers: {NUM_OF_CHANNELS}")
    print(f"Number of classes: {NUM_OF_CLASSES}")
    print(f"Coefficient of load: {ro:<1.2f}")
    print(f"Coefficient of variation of service time: {SERVICE_TIME_CV:<1.2f}")
    print("PR (Preamptive) priority")

    # when creating the simulation, pass the number of servers, number of classes and type of priority.
    # PR - absolute with re-service of requests
    qs = PriorityQueueSimulator(NUM_OF_CHANNELS, NUM_OF_CLASSES, "PR")

    # to set up sources of requests and service servers, we need to specify a set of dictionaries with fields
    # type - distribution type,
    # params - its parameters.
    # The number of such dictionaries in the lists sources and servers_params corresponds to the number of classes

    sources = []
    servers_params = []
    for j in range(NUM_OF_CLASSES):
        sources.append({'type': 'M', 'params': ARRIVAL_RATES[j]})
        servers_params.append({'type': 'Gamma', 'params': params[j]})

    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # start the simulation
    qs.run(NUM_OF_JOBS)

    # get the initial moments of time spent

    v_sim = qs.v

    # calculate them as well using the method of invariant relations (for comparison)
    invar_calc = MGnInvarApproximation(ARRIVAL_RATES, b, n=NUM_OF_CHANNELS)
    v_teor = invar_calc.get_v('PR')

    assert abs(v_sim[0][0] - v_teor[0][0]) < 0.3

    times_print_with_classes(v_sim, v_teor, False)

    print("NP (Non-preamptive) priority")

    # The same for relative priority (NP)
    qs = PriorityQueueSimulator(NUM_OF_CHANNELS, NUM_OF_CLASSES, "NP")
    sources = []
    servers_params = []
    for j in range(NUM_OF_CLASSES):
        sources.append({'type': 'M', 'params': ARRIVAL_RATES[j]})
        servers_params.append({'type': 'Gamma', 'params': params[j]})

    qs.set_sources(sources)
    qs.set_servers(servers_params)

    qs.run(NUM_OF_JOBS)

    v_sim = qs.v

    v_teor = invar_calc.get_v('NP')

    times_print_with_classes(v_sim, v_teor, False)

    assert abs(v_sim[0][0] - v_teor[0][0]) < 0.3


if __name__ == "__main__":
    test_sim()
