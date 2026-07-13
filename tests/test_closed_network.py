"""
Test closed queueing networks: exact MVA vs Buzen convolution (machine
precision), Schweitzer approximate MVA vs exact, and MVA vs simulation.
"""

import numpy as np

from most_queue.sim.networks.closed_network import ClosedNetworkSim
from most_queue.theory.networks.closed_network import ClosedNetworkCalc

# Central-server model: node 0 — CPU, nodes 1-2 — disks
ROUTING = np.array(
    [
        [0.1, 0.5, 0.4],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
)
SERVICE_MEANS = [0.02, 0.06, 0.08]
POPULATION = 8


def _run(method, n_channels, routing=ROUTING, b=SERVICE_MEANS, n_pop=POPULATION):
    calc = ClosedNetworkCalc(method=method)
    calc.set_sources(R=routing, N=n_pop)
    calc.set_nodes(b=b, n=n_channels)
    return calc.run()


def test_mva_equals_convolution_single_server():
    """MVA and Buzen convolution must agree to machine precision."""
    mva = _run("mva", [1, 1, 1])
    buzen = _run("convolution", [1, 1, 1])

    assert np.isclose(mva.throughput, buzen.throughput, rtol=1e-12)
    assert np.allclose(mva.mean_jobs, buzen.mean_jobs, rtol=1e-10)
    assert np.allclose(mva.v_node, buzen.v_node, rtol=1e-10)


def test_mva_equals_convolution_multi_server_and_delay():
    """Load-dependent convolution (multi-server + delay nodes) matches exact MVA."""
    n_channels = [2, 1, None]  # 2-channel CPU, one disk, delay (think) node
    mva = _run("mva", n_channels)
    buzen = _run("convolution", n_channels)

    assert np.isclose(mva.throughput, buzen.throughput, rtol=1e-10)
    assert np.allclose(mva.mean_jobs, buzen.mean_jobs, rtol=1e-8)


def test_schweitzer_close_to_exact():
    """Schweitzer approximate MVA is within a few percent of exact MVA."""
    exact = _run("mva", [1, 1, 1])
    approx = _run("schweitzer", [1, 1, 1])

    assert np.isclose(approx.throughput, exact.throughput, rtol=0.03)
    assert np.allclose(approx.mean_jobs, exact.mean_jobs, rtol=0.15, atol=0.1)


def test_throughput_bounded_by_bottleneck():
    """X(N) must not exceed the bottleneck service rate 1 / (e_i * b_i)."""
    res = _run("mva", [1, 1, 1])
    calc = ClosedNetworkCalc()
    calc.set_sources(R=ROUTING, N=POPULATION)
    calc.set_nodes(b=SERVICE_MEANS, n=[1, 1, 1])
    e = calc.visit_ratios()
    bottleneck_rate = 1.0 / max(e[i] * SERVICE_MEANS[i] for i in range(3))

    assert res.throughput <= bottleneck_rate * (1 + 1e-12)
    assert all(load <= 1.0 + 1e-12 for load in res.loads)


def test_mva_vs_simulation():
    """Exact MVA against the event-driven simulator (exponential service)."""
    n_channels = [2, 1, 1]
    mva = _run("mva", n_channels)

    sim = ClosedNetworkSim(seed=42)
    sim.set_sources(R=ROUTING, N=POPULATION)
    sim.set_nodes(
        serv_params=[{"type": "M", "params": 1.0 / b} for b in SERVICE_MEANS],
        n=n_channels,
    )
    sim_res = sim.run(300_000)

    assert np.isclose(sim_res.throughput, mva.throughput, rtol=0.03)
    assert np.allclose(sim_res.mean_jobs, mva.mean_jobs, rtol=0.1, atol=0.15)
    assert np.allclose(sim_res.loads, mva.loads, rtol=0.05, atol=0.02)


def test_mva_with_delay_node_vs_simulation():
    """Classic machine-repair / terminals model: delay node + FCFS server."""
    routing = np.array([[0.0, 1.0], [1.0, 0.0]])
    b = [2.0, 0.1]  # think time 2.0 at delay node, service 0.1
    n_channels = [None, 1]
    n_pop = 10

    mva = _run("mva", n_channels, routing=routing, b=b, n_pop=n_pop)

    sim = ClosedNetworkSim(seed=42)
    sim.set_sources(R=routing, N=n_pop)
    sim.set_nodes(
        serv_params=[{"type": "M", "params": 0.5}, {"type": "M", "params": 10.0}],
        n=n_channels,
    )
    sim_res = sim.run(200_000)

    assert np.isclose(sim_res.throughput, mva.throughput, rtol=0.03)
    assert np.allclose(sim_res.mean_jobs, mva.mean_jobs, rtol=0.1, atol=0.1)


if __name__ == "__main__":
    test_mva_equals_convolution_single_server()
    test_mva_equals_convolution_multi_server_and_delay()
    test_schweitzer_close_to_exact()
    test_throughput_bounded_by_bottleneck()
    test_mva_vs_simulation()
    test_mva_with_delay_node_vs_simulation()
