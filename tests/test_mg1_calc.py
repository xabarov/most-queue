import numpy as np

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import (
    H2Distribution,
    ParetoDistribution,
    UniformDistribution,
)
from most_queue.sim.queueing_systems.fifo import QueueingSystemSimulator
from most_queue.theory.queueing_systems.fifo.mg1 import MG1Calculation


def test_mg1():
    """
    Testing the M/G/1 queueing system calculation
    For verification, we use simulation modeling (IM).
    """
    l = 1  # input flow intensity
    b1 = 0.7  # average service time
    coev = 1.2  # coefficient of variation of service time
    num_of_jobs = 300000  # number of jobs for IM

    # selecting parameters of the approximating H2-distribution for service time H2Params [p1, mu1, mu2]:
    params = H2Distribution.get_params_by_mean_and_coev(b1, coev)
    print(params)
    b = H2Distribution.calc_theory_moments(params, 4)

    # calculation using numerical methods
    mg1_num = MG1Calculation(l, b)
    w_ch = mg1_num.get_w()
    p_ch = mg1_num.get_p()
    v_ch = mg1_num.get_v()

    # running IM for verification of results
    qs = QueueingSystemSimulator(1)
    qs.set_servers(params, "H")
    qs.set_sources(l, "M")
    qs.run(num_of_jobs)
    w_sim = qs.w
    p_sim = qs.get_p()
    v_sim = qs.v

    # outputting the results
    print("M/H2/1")

    times_print(w_sim, w_ch, True)
    times_print(v_sim, v_ch, False)
    probs_print(p_sim, p_ch, 10)

    # The same for other distributions of service time
    print("Uniform")
    params = UniformDistribution.get_params_by_mean_and_coev(b1, coev)
    b = UniformDistribution.calc_theory_moments(params, 4)
    mg1_num = MG1Calculation(l, b)
    w_ch = mg1_num.get_w()
    p_ch = mg1_num.get_p(dist_type='Uniform')
    v_ch = mg1_num.get_v()

    qs = QueueingSystemSimulator(1)
    qs.set_servers(params, "Uniform")
    qs.set_sources(l, "M")
    qs.run(num_of_jobs)
    w_sim = qs.w
    p_sim = qs.get_p()
    v_sim = qs.v

    times_print(w_sim, w_ch, True)
    times_print(v_sim, v_ch, False)
    probs_print(p_sim, p_ch, 10)

    print("Pareto")

    pareto_params = ParetoDistribution.get_params_by_mean_and_coev(b1, coev)
    print(pareto_params)
    b = ParetoDistribution.calc_theory_moments(pareto_params, 4)
    mg1_num = MG1Calculation(l, b)
    w_ch = mg1_num.get_w()
    p_ch = mg1_num.get_p(dist_type='Pa')
    v_ch = mg1_num.get_v()

    qs = QueueingSystemSimulator(1)
    qs.set_servers(pareto_params, "Pa")
    qs.set_sources(l, "M")
    qs.run(num_of_jobs)
    w_sim = qs.w
    p_sim = qs.get_p()
    v_sim = qs.v

    assert np.allclose(np.array(p_sim[:10]), np.array(p_ch[:10]), atol=1e-2)

    times_print(w_sim, w_ch, True)
    times_print(v_sim, v_ch, False)
    probs_print(p_sim, p_ch, 10)


if __name__ == "__main__":
    test_mg1()
