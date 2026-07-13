"""
Test the exact Jackson product-form solver: against the approximate
decomposition (OpenNetworkCalc, Markovian mode) and against simulation.
"""

import numpy as np

from most_queue.sim.networks.network import NetworkSimulator
from most_queue.theory.networks.jackson_network import JacksonNetworkCalc
from most_queue.theory.networks.open_network import OpenNetworkCalc

ARRIVAL_RATE = 1.0
NUM_OF_NODES = 5
NUM_OF_CHANNELS = [3, 2, 3, 4, 3]
UTILIZATION_FACTOR = 0.7

TRANSITION_MATRIX = np.matrix(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 0.4, 0.6, 0, 0, 0],
        [0, 0, 0.2, 0.4, 0.4, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ]
)


def _service_rates(intensities):
    """Service rates giving the target utilization at every node."""
    return [intensities[i] / (UTILIZATION_FACTOR * NUM_OF_CHANNELS[i]) for i in range(NUM_OF_NODES)]


def _jackson():
    calc = JacksonNetworkCalc()
    calc.set_sources(arrival_rate=ARRIVAL_RATE, R=TRANSITION_MATRIX)
    intensities = calc.solve_intensities()
    mu = _service_rates(intensities)
    calc.set_nodes(mu=mu, n=NUM_OF_CHANNELS)
    return calc.run(), mu


def test_jackson_vs_decomposition():
    """On a Markovian network the approximate decomposition must be close to
    the exact product-form answer; the error is documented by the assert."""
    jackson_results, mu = _jackson()

    open_calc = OpenNetworkCalc(is_markovian=True)
    open_calc.set_sources(R=TRANSITION_MATRIX, arrival_rate=ARRIVAL_RATE)
    b = [[1.0 / m, 2.0 / m**2, 6.0 / m**3] for m in mu]
    open_calc.set_nodes(b=b, n=NUM_OF_CHANNELS)
    decomposition_results = open_calc.run()

    assert np.allclose(jackson_results.intensities, decomposition_results.intensities, rtol=1e-8)
    assert np.allclose(jackson_results.loads, decomposition_results.loads, rtol=1e-6)
    # Mean sojourn: decomposition error on a Markovian network stays within 10%
    assert np.isclose(jackson_results.v[0], decomposition_results.v[0], rtol=0.1)


def test_jackson_vs_simulation():
    """Exact Jackson means against event-driven simulation."""
    np.random.seed(42)
    jackson_results, mu = _jackson()

    sim = NetworkSimulator()
    sim.set_sources(arrival_rate=ARRIVAL_RATE, R=TRANSITION_MATRIX)
    serv_params = [{"type": "M", "params": m} for m in mu]
    sim.set_nodes(serv_params=serv_params, n=NUM_OF_CHANNELS)
    sim_results = sim.run(200_000)

    assert np.isclose(jackson_results.v[0], sim_results.v[0], rtol=0.05)


def test_single_node_reduces_to_mmn():
    """A one-node network must reproduce the M/M/3 formula."""
    routing = np.matrix([[1.0, 0.0], [0.0, 1.0]])
    calc = JacksonNetworkCalc()
    calc.set_sources(arrival_rate=2.0, R=routing)
    calc.set_nodes(mu=[1.0], n=[3])
    res = calc.run()

    # Erlang-C for M/M/3, lambda=2, mu=1: W = Wq + 1/mu
    big_l, w = JacksonNetworkCalc._mmn_metrics(2.0, 1.0, 3)
    assert np.isclose(res.v[0], w, rtol=1e-12)
    assert np.isclose(res.mean_jobs[0], big_l, rtol=1e-12)


if __name__ == "__main__":
    test_jackson_vs_decomposition()
    test_jackson_vs_simulation()
    test_single_node_reduces_to_mmn()
