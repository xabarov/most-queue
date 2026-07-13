"""
Test a network with a MAP (bursty, non-Poisson) external flow: QNA fed with
the MAP interarrival cv2 must beat QNA under the Poisson assumption.
"""

import numpy as np

from most_queue.random.map_ph import MAP, MAPParams
from most_queue.sim.networks.network import NetworkSimulator
from most_queue.theory.networks.qna import OpenNetworkCalcQNA, map_arrival_cv2

# Bursty MMPP-2: high-rate and low-rate phases with slow switching
MAP_PARAMS = MAPParams(
    D0=np.array([[-2.6, 0.1], [0.1, -0.35]]),
    D1=np.array([[2.5, 0.0], [0.0, 0.25]]),
)

# Tandem: source -> 1 -> 2 -> out
ROUTING = np.matrix(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
)
NUM_OF_CHANNELS = [1, 1]


def _qna(rate, cv2, b):
    qna = OpenNetworkCalcQNA()
    qna.set_sources(arrival_rate=rate, R=ROUTING, arrival_cv2=cv2)
    qna.set_nodes(b=b, n=NUM_OF_CHANNELS)
    return qna.run()


def test_map_cv2_helper():
    """Helper must reproduce MAP.calc_theory_moments-based rate and cv2 > 1."""
    rate, cv2 = map_arrival_cv2(MAP_PARAMS)
    assert np.isclose(rate, MAP.arrival_rate(MAP_PARAMS), rtol=1e-9)
    assert cv2 > 1.5  # bursty by construction


def test_qna_with_map_cv2_beats_poisson_assumption():
    """On a MAP-driven tandem, QNA(cv2 of MAP) must be closer to simulation
    than QNA(Poisson); residual error is due to ignored autocorrelation."""
    rate, cv2 = map_arrival_cv2(MAP_PARAMS)

    # Exponential service, utilization 0.6 at both nodes
    b1 = 0.6 / rate
    mu = 1.0 / b1
    b = [[b1, 2 / mu**2, 6 / mu**3]] * 2

    qna_map = _qna(rate, cv2, b)
    qna_poisson = _qna(rate, 1.0, b)

    np.random.seed(42)
    sim = NetworkSimulator()
    sim.set_sources(arrival_rate=rate, R=ROUTING, source_kendall="MAP", source_params=MAP_PARAMS)
    sim.set_nodes(serv_params=[{"type": "M", "params": mu}] * 2, n=NUM_OF_CHANNELS)
    sim_res = sim.run(300_000)

    v_sim = sim_res.v[0]
    err_map = abs(qna_map.v[0] - v_sim) / v_sim
    err_poisson = abs(qna_poisson.v[0] - v_sim) / v_sim

    print(
        f"sim={v_sim:.3f}, qna_map={qna_map.v[0]:.3f} ({100 * err_map:.1f}%), "
        f"qna_poisson={qna_poisson.v[0]:.3f} ({100 * err_poisson:.1f}%), cv2={cv2:.2f}"
    )

    assert err_map < err_poisson


if __name__ == "__main__":
    test_map_cv2_helper()
    test_qna_with_map_cv2_beats_poisson_assumption()
