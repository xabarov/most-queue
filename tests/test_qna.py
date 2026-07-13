"""
Test QNA (Whitt): on a network with non-Markovian service the two-moment
propagation of internal-flow variability must beat the plain decomposition
(which treats internal flows as Poisson).
"""

import numpy as np

from most_queue.random.distributions import H2Distribution
from most_queue.sim.networks.network import NetworkSimulator
from most_queue.theory.networks.open_network import OpenNetworkCalc
from most_queue.theory.networks.qna import OpenNetworkCalcQNA

ARRIVAL_RATE = 1.0
SERVICE_CV = 2.0  # c2_s = 4 — highly variable service
UTILIZATION = 0.8
NUM_OF_NODES = 3
NUM_OF_CHANNELS = [1, 1, 1]

# Tandem: source -> 1 -> 2 -> 3 -> out
TRANSITION_MATRIX = np.matrix(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)


def _service_moments():
    b = []
    serv_params = []
    for i in range(NUM_OF_NODES):
        b1 = UTILIZATION * NUM_OF_CHANNELS[i] / ARRIVAL_RATE
        h2_params = H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_CV)
        serv_params.append({"type": "H", "params": h2_params})
        b.append(H2Distribution.calc_theory_moments(h2_params, 4))
    return b, serv_params


def test_qna_beats_plain_decomposition():
    """|QNA - sim| must not exceed |plain decomposition - sim| for the mean
    network sojourn time on a high-cv tandem."""
    b, serv_params = _service_moments()

    qna = OpenNetworkCalcQNA()
    qna.set_sources(arrival_rate=ARRIVAL_RATE, R=TRANSITION_MATRIX)
    qna.set_nodes(b=b, n=NUM_OF_CHANNELS)
    qna_res = qna.run()

    plain = OpenNetworkCalc()
    plain.set_sources(arrival_rate=ARRIVAL_RATE, R=TRANSITION_MATRIX)
    plain.set_nodes(b=b, n=NUM_OF_CHANNELS)
    plain_res = plain.run()

    np.random.seed(42)
    sim = NetworkSimulator()
    sim.set_sources(arrival_rate=ARRIVAL_RATE, R=TRANSITION_MATRIX)
    sim.set_nodes(serv_params=serv_params, n=NUM_OF_CHANNELS)
    sim_res = sim.run(300_000)

    v_sim = sim_res.v[0]
    err_qna = abs(qna_res.v[0] - v_sim) / v_sim
    err_plain = abs(plain_res.v[0] - v_sim) / v_sim

    print(
        f"sim={v_sim:.3f}, qna={qna_res.v[0]:.3f} ({100 * err_qna:.1f}%), plain={plain_res.v[0]:.3f} ({100 * err_plain:.1f}%)"
    )

    assert err_qna <= err_plain
    assert err_qna < 0.10


def test_qna_reduces_to_erlang_c_for_markovian():
    """With exponential service and Poisson arrivals QNA must reproduce
    the exact M/M/n answer at every node (c2_a = c2_s = 1)."""
    mu = 1.25
    b = [[1 / mu, 2 / mu**2, 6 / mu**3]] * 2
    routing = np.matrix(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    qna = OpenNetworkCalcQNA()
    qna.set_sources(arrival_rate=1.0, R=routing)
    qna.set_nodes(b=b, n=[1, 1])
    res = qna.run()

    # M/M/1: W = 1 / (mu - lambda)
    w_exact = 1.0 / (mu - 1.0)
    assert np.allclose(res.v_node, [w_exact, w_exact], rtol=1e-8)
    assert np.allclose(qna.arrival_cv2_nodes, [1.0, 1.0], atol=1e-8)


if __name__ == "__main__":
    test_qna_beats_plain_decomposition()
    test_qna_reduces_to_erlang_c_for_markovian()
