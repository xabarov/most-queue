"""
Testing the M/G/1 queueing system calculation
For verification, we use simulation modeling 
"""
import numpy as np

from most_queue.general.tables import probs_print, times_print
from most_queue.rand_distribution import (
    H2Distribution,
    ParetoDistribution,
    UniformDistribution,
)
from most_queue.sim.base import QsSim
from most_queue.theory.fifo.mg1 import MG1Calculation

NUM_OF_SERVERS = 1
ARRIVAL_RATE = 1.0
UTILIZATION = 0.7
NUM_OF_JOBS = 300000
SERVICE_TIME_CV = 1.5

def test_mg1():
    """
    Testing the M/G/1 queueing system calculation
    For verification, we use simulation modeling
    """
    b1 = UTILIZATION*NUM_OF_SERVERS/ARRIVAL_RATE

    # selecting parameters of the approximating H2-distribution
    # for service time H2Params [p1, mu1, mu2]:
    params = H2Distribution.get_params_by_mean_and_coev(b1, SERVICE_TIME_CV)
    print(params)
    b = H2Distribution.calc_theory_moments(params, 4)

    # calculation using numerical methods
    mg1_num = MG1Calculation(ARRIVAL_RATE, b)
    w_num = mg1_num.get_w()
    p_num = mg1_num.get_p()
    v_num = mg1_num.get_v()

    # running IM for verification of results
    qs = QsSim(1)
    qs.set_servers(params, "H")
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.run(NUM_OF_JOBS)
    w_sim = qs.w
    p_sim = qs.get_p()
    v_sim = qs.v

    # outputting the results
    print("M/H2/1")

    times_print(w_sim, w_num, True)
    times_print(v_sim, v_num, False)
    probs_print(p_sim, p_num, 10)

    # The same for other distributions of service time
    print("Uniform")
    params = UniformDistribution.get_params_by_mean_and_coev(b1, SERVICE_TIME_CV)
    b = UniformDistribution.calc_theory_moments(params, 4)
    mg1_num = MG1Calculation(ARRIVAL_RATE, b)
    w_num = mg1_num.get_w()
    p_num = mg1_num.get_p(dist_type='Uniform')
    v_num = mg1_num.get_v()

    qs = QsSim(1)
    qs.set_servers(params, "Uniform")
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.run(NUM_OF_JOBS)
    w_sim = qs.w
    p_sim = qs.get_p()
    v_sim = qs.v

    times_print(w_sim, w_num, True)
    times_print(v_sim, v_num, False)
    probs_print(p_sim, p_num, 10)

    print("Pareto")

    pareto_params = ParetoDistribution.get_params_by_mean_and_coev(b1, SERVICE_TIME_CV)
    print(pareto_params)
    b = ParetoDistribution.calc_theory_moments(pareto_params, 4)
    mg1_num = MG1Calculation(ARRIVAL_RATE, b)
    w_num = mg1_num.get_w()
    p_num = mg1_num.get_p(dist_type='Pa')
    v_num = mg1_num.get_v()

    qs = QsSim(1)
    qs.set_servers(pareto_params, "Pa")
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.run(NUM_OF_JOBS)
    w_sim = qs.w
    p_sim = qs.get_p()
    v_sim = qs.v

    assert np.allclose(np.array(p_sim[:10]), np.array(p_num[:10]), atol=1e-2)

    times_print(w_sim, w_num, True)
    times_print(v_sim, v_num, False)
    probs_print(p_sim, p_num, 10)


if __name__ == "__main__":
    test_mg1()
