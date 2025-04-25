from most_queue.general.tables import times_print_with_classes
from most_queue.rand_distribution import GammaDistribution
from most_queue.sim.queueing_systems.priority import PriorityQueueSimulator
from most_queue.theory.queueing_systems.priority.mgn_invar_approx import MGnInvarApproximation



def test_sim():
    """
    Testing the simulation model of an M/M/c queue with priorities.
    For verification, comparing results with those calculated using the method of invariant relations:
        Ryzhikov Yu.I., Khomonenko A.D. Calculation of multi-channel service systems with absolute and
        relative priorities based on invariant relations // Intelligent technologies
        for transportation. 2015. №3
    """
    n = 5  # number of servers
    k = 3  # number of classes of requests
    l = [0.2, 0.3, 0.4]  # service intensities by request classes
    lsum = sum(l)
    num_of_jobs = 300000  # number of jobs for the simulation

    # Set up the parameters for service times at initial moments.
    # Set average service times by class
    b1 = [0.45 * n, 0.9 * n, 1.35 * n]

    # Coefficient of variation of service time let's be the same for all classes
    coev = 0.577

    # second initial moments
    b2 = [0] * k
    for i in range(k):
        b2[i] = (b1[i] ** 2) * (1 + coev ** 2)

    b_sr = sum(b1) / k

    # get the coefficient of load
    ro = lsum * b_sr / n

    # now, given the two initial moments, select parameters for the approximating Gamma distribution
    # and add them to the list of parameters params
    params = []
    for i in range(k):
        params.append(GammaDistribution.get_params([b1[i], b2[i]]))

    b = []
    for j in range(k):
        b.append(GammaDistribution.calc_theory_moments(
            params[j], 4))

    print("\nComparison of data from the simulation and results calculated using the method of invariant relations (R) \n"
          "time spent in a multi-channel queue with priorities")
    print("Number of servers: " + str(n) + "\nNumber of classes: " + str(k) + "\nCoefficient of load: {0:<1.2f}".format(ro) +
          "\nCoefficient of variation of service time: " + str(coev) + "\n")
    print("PR (Preamptive) priority")

    # when creating the simulation, pass the number of servers, number of classes and type of priority.
    # PR - absolute with re-service of requests
    qs = PriorityQueueSimulator(n, k, "PR")

    # to set up sources of requests and service servers, we need to specify a set of dictionaries with fields
    # type - distribution type,
    # params - its parameters.
    # The number of such dictionaries in the lists sources and servers_params corresponds to the number of classes

    sources = []
    servers_params = []
    for j in range(k):
        sources.append({'type': 'M', 'params': l[j]})
        servers_params.append({'type': 'Gamma', 'params': params[j]})

    qs.set_sources(sources)
    qs.set_servers(servers_params)

    # start the simulation
    qs.run(num_of_jobs)

    # get the initial moments of time spent

    v_sim = qs.v

    # calculate them as well using the method of invariant relations (for comparison)
    invar_calc = MGnInvarApproximation(l, b, n=n)
    v_teor = invar_calc.get_v('PR')

    assert abs(v_sim[0][0] - v_teor[0][0]) < 0.3

    times_print_with_classes(v_sim, v_teor, False)

    print("NP (Non-preamptive) priority")

    # The same for relative priority (NP)
    qs = PriorityQueueSimulator(n, k, "NP")
    sources = []
    servers_params = []
    for j in range(k):
        sources.append({'type': 'M', 'params': l[j]})
        servers_params.append({'type': 'Gamma', 'params': params[j]})

    qs.set_sources(sources)
    qs.set_servers(servers_params)

    qs.run(num_of_jobs)

    v_sim = qs.v

    v_teor = invar_calc.get_v('NP')

    times_print_with_classes(v_sim, v_teor, False)

    assert abs(v_sim[0][0] - v_teor[0][0]) < 0.3


if __name__ == "__main__":
    test_sim()
