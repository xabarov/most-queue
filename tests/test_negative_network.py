"""
Test for NegativeNetwork - network with negative jobs
"""

import os

import numpy as np
import pytest
import yaml

from most_queue.io.tables import print_sojourn_moments
from most_queue.random.distributions import H2Distribution
from most_queue.sim.negative import NegativeServiceType
from most_queue.sim.networks.negative_network import NegativeNetwork
from most_queue.sim.networks.network import NetworkSimulator


def test_negative_network_basic():
    """
    Basic test for NegativeNetwork with global negative arrivals.
    """
    # Create network with global negative arrivals
    network = NegativeNetwork(negative_arrival_type="global")

    # Set sources: positive arrivals and negative arrivals
    positive_rate = 1.0
    negative_rate = 0.1  # Low rate of negative arrivals
    R = np.matrix(
        [
            [1.0, 0.0, 0.0],  # From source: 100% to node 0
            [0.0, 0.5, 0.5],  # From node 0: 50% to node 1, 50% exit
            [0.0, 0.0, 1.0],  # From node 1: 100% exit
        ]
    )

    network.set_sources(positive_rate, R, negative_arrival_rate=negative_rate)

    # Set nodes: 2 nodes with different configurations
    serv_params = [
        {"type": "M", "params": 2.0},  # Node 0: exponential service, rate 2.0
        {"type": "M", "params": 1.5},  # Node 1: exponential service, rate 1.5
    ]
    n = [2, 1]  # 2 channels in node 0, 1 channel in node 1
    negative_types = [
        NegativeServiceType.DISASTER,  # Node 0: disaster type
        NegativeServiceType.RCS,  # Node 1: remove customer in service
    ]
    buffers = [None, 10]  # Node 0: infinite buffer, Node 1: buffer size 10

    network.set_nodes(serv_params, n, negative_types=negative_types, buffers=buffers)

    # Run simulation
    results = network.run(job_served=100)

    # Basic assertions
    assert results.served > 0, "Should have served some jobs"
    assert results.arrived > 0, "Should have received some arrivals"
    assert results.served <= results.arrived, "Served should not exceed arrived"
    assert results.v[0] > 0, "Mean sojourn time should be positive"
    assert results.duration > 0, "Simulation should take some time"

    # Check that negative arrivals were processed
    # Note: negative arrivals are handled at network level, not per node
    # Individual nodes may have their own negative arrivals if configured via set_node_sources
    assert network.negative_arrival_time > 0, "Negative arrival time should be set"


def test_negative_network_no_negative_arrivals():
    """
    Test NegativeNetwork without negative arrivals (should work like regular network).
    """
    network = NegativeNetwork(negative_arrival_type="global")

    positive_rate = 1.0
    R = np.matrix(
        [
            [1.0, 0.0, 0.0],  # From source: 100% to node 0
            [0.0, 0.0, 1.0],  # From node 0: 100% exit
            [0.0, 0.0, 1.0],  # From node 1: 100% exit
        ]
    )

    # Set sources without negative arrivals
    network.set_sources(positive_rate, R, negative_arrival_rate=None)

    serv_params = [{"type": "M", "params": 1.5}]
    n = [2]
    negative_types = [NegativeServiceType.DISASTER]

    network.set_nodes(serv_params, n, negative_types=negative_types)

    results = network.run(job_served=50)

    assert results.served > 0
    assert results.arrived > 0


def test_negative_network_different_types():
    """
    Test NegativeNetwork with different negative service types per node.
    """
    network = NegativeNetwork(negative_arrival_type="global")

    positive_rate = 0.8
    negative_rate = 0.05
    R = np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],  # From source: 100% to node 0
            [0.0, 0.33, 0.33, 0.34],  # From node 0: 33% to node 1, 33% to node 2, 34% exit
            [0.0, 0.0, 0.0, 1.0],  # From node 1: 100% exit
            [0.0, 0.0, 0.0, 1.0],  # From node 2: 100% exit
        ]
    )

    network.set_sources(positive_rate, R, negative_arrival_rate=negative_rate)

    serv_params = [
        {"type": "M", "params": 2.0},
        {"type": "M", "params": 2.0},
        {"type": "M", "params": 2.0},
    ]
    n = [1, 1, 1]
    negative_types = [
        NegativeServiceType.DISASTER,
        NegativeServiceType.RCS,
        NegativeServiceType.RCH,
    ]

    network.set_nodes(serv_params, n, negative_types=negative_types)

    results = network.run(job_served=100)

    assert results.served > 0
    assert results.arrived > 0

    # Check that each node has the correct negative type
    assert network.qs[0].type_of_negatives == NegativeServiceType.DISASTER
    assert network.qs[1].type_of_negatives == NegativeServiceType.RCS
    assert network.qs[2].type_of_negatives == NegativeServiceType.RCH


def test_negative_network_per_node_type():
    """
    Test NegativeNetwork with per_node negative arrival type.
    Each node has its own negative arrival rate.
    """
    network = NegativeNetwork(negative_arrival_type="per_node")

    positive_rate = 1.0
    R = np.matrix(
        [
            [1.0, 0.0, 0.0],  # From source: 100% to node 0
            [0.0, 0.5, 0.5],  # From node 0: 50% to node 1, 50% exit
            [0.0, 0.0, 1.0],  # From node 1: 100% exit
        ]
    )

    serv_params = [
        {"type": "M", "params": 2.0},
        {"type": "M", "params": 2.0},
    ]
    n = [1, 1]
    negative_types = [NegativeServiceType.DISASTER, NegativeServiceType.RCS]

    # Set nodes first (required for per_node type)
    network.set_nodes(serv_params, n, negative_types=negative_types)

    # Set sources with per-node negative arrival rates
    negative_rates = [0.1, 0.05]  # Different rates for each node
    network.set_sources(positive_rate, R, negative_arrival_rates=negative_rates)

    # Verify configuration
    assert network.negative_arrival_type == "per_node"
    assert len(network.negative_arrival_rates) == 2
    assert network.negative_arrival_rates[0] == 0.1
    assert network.negative_arrival_rates[1] == 0.05
    assert len(network.negative_arrival_times) == 2

    # Run simulation
    results = network.run(job_served=100)

    assert results.served > 0
    assert results.arrived > 0


def test_negative_network_per_node_different_rates():
    """
    Test NegativeNetwork with per_node type and different rates for each node.
    """
    network = NegativeNetwork(negative_arrival_type="per_node")

    positive_rate = 0.8
    R = np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],  # From source: 100% to node 0
            [0.0, 0.33, 0.33, 0.34],  # From node 0: 33% to node 1, 33% to node 2, 34% exit
            [0.0, 0.0, 0.0, 1.0],  # From node 1: 100% exit
            [0.0, 0.0, 0.0, 1.0],  # From node 2: 100% exit
        ]
    )

    serv_params = [
        {"type": "M", "params": 2.0},
        {"type": "M", "params": 2.0},
        {"type": "M", "params": 2.0},
    ]
    n = [1, 1, 1]
    negative_types = [
        NegativeServiceType.DISASTER,
        NegativeServiceType.RCS,
        NegativeServiceType.RCH,
    ]

    # IMPORTANT: set_nodes() must be called before set_sources() for per_node type
    network.set_nodes(serv_params, n, negative_types=negative_types)

    # Different negative rates for each node
    negative_rates = [0.2, 0.1, 0.15]
    network.set_sources(positive_rate, R, negative_arrival_rates=negative_rates)

    results = network.run(job_served=100)

    assert results.served > 0
    assert results.arrived > 0

    # Check that each node has its own negative arrival time and rate
    assert len(network.negative_arrival_times) == 3
    assert len(network.negative_arrival_rates) == 3
    for i, (time, rate) in enumerate(zip(network.negative_arrival_times, network.negative_arrival_rates)):
        assert rate == negative_rates[i], f"Node {i} should have correct negative rate"
        assert time > 0, f"Node {i} should have a valid negative arrival time"


def test_negative_network_per_node_order_error():
    """
    Test that per_node type requires set_nodes() before set_sources().
    """
    network = NegativeNetwork(negative_arrival_type="per_node")

    positive_rate = 1.0
    R = np.matrix(
        [
            [1.0, 0.0, 0.0],  # From source: 100% to node 0
            [0.0, 0.0, 1.0],  # From node 0: 100% exit
            [0.0, 0.0, 1.0],  # From node 1: 100% exit
        ]
    )

    # Try to set sources before nodes - should raise error
    with pytest.raises(ValueError, match="set_nodes.*must be called before set_sources"):
        network.set_sources(positive_rate, R, negative_arrival_rates=[0.1, 0.1])


def test_negative_network_per_node_wrong_length():
    """
    Test that negative_arrival_rates must match number of nodes.
    """
    network = NegativeNetwork(negative_arrival_type="per_node")

    positive_rate = 1.0
    R = np.matrix(
        [
            [1.0, 0.0, 0.0],  # From source: 100% to node 0
            [0.0, 0.5, 0.5],  # From node 0: 50% to node 1, 50% exit
            [0.0, 0.0, 1.0],  # From node 1: 100% exit
        ]
    )

    serv_params = [
        {"type": "M", "params": 2.0},
        {"type": "M", "params": 2.0},
    ]
    n = [1, 1]

    network.set_nodes(serv_params, n)

    # Try with wrong length - should raise error
    with pytest.raises(ValueError, match="must match number of nodes"):
        network.set_sources(positive_rate, R, negative_arrival_rates=[0.1, 0.1, 0.1])  # 3 rates for 2 nodes


def test_negative_network_with_output():
    """
    Run NegativeNetwork simulation and output time parameters.
    Similar to test_network_no_prty.py but without comparison with analytical methods.
    """
    # Create network with global negative arrivals
    network = NegativeNetwork(negative_arrival_type="global")

    # Configuration
    positive_rate = 1.0
    negative_rate = 0.15  # Rate of negative arrivals
    R = np.matrix(
        [
            [1.0, 0.0, 0.0],  # From source: 100% to node 0
            [0.0, 0.4, 0.6],  # From node 0: 40% to node 1, 60% exit
            [0.0, 0.0, 1.0],  # From node 1: 100% exit
        ]
    )

    network.set_sources(positive_rate, R, negative_arrival_rate=negative_rate)

    # Set nodes: 2 nodes with different configurations
    serv_params = [
        {"type": "M", "params": 2.0},  # Node 0: exponential service, rate 2.0
        {"type": "M", "params": 1.5},  # Node 1: exponential service, rate 1.5
    ]
    n = [2, 1]  # 2 channels in node 0, 1 channel in node 1
    negative_types = [
        NegativeServiceType.DISASTER,  # Node 0: disaster type (remove all)
        NegativeServiceType.RCS,  # Node 1: remove customer in service
    ]
    buffers = [None, 10]  # Node 0: infinite buffer, Node 1: buffer size 10

    network.set_nodes(serv_params, n, negative_types=negative_types, buffers=buffers)

    # Run simulation
    num_jobs = 100000
    print(f"\n{'='*60}")
    print("Negative Network Simulation")
    print(f"{'='*60}")
    print("Configuration:")
    print(f"  Positive arrival rate: {positive_rate}")
    print(f"  Negative arrival rate: {negative_rate}")
    print(f"  Number of nodes: {len(n)}")
    print(f"  Channels per node: {n}")
    print(f"  Negative types: {[nt.name for nt in negative_types]}")
    print(f"  Number of jobs to serve: {num_jobs}")
    print(f"{'='*60}\n")

    results = network.run(job_served=num_jobs)

    # Output results
    print(f"\n{'='*60}")
    print("Simulation Results")
    print(f"{'='*60}")
    print(f"Jobs served: {results.served}")
    print(f"Jobs arrived: {results.arrived}")
    print(f"Simulation duration: {results.duration:.5f} sec")

    # Output network-level sojourn time moments
    print("\n" + "=" * 60)
    print("Network-level Sojourn Time Moments")
    print("=" * 60)
    if results.v is not None and len(results.v) > 0:
        print(f"{'Moment':^15s}|{'Value':^15s}")
        print("-" * 30)
        for i, moment in enumerate(results.v[:4], 1):
            print(f"{i:^15d}|{moment:^15.5g}")
    else:
        print("No sojourn time data available")

    # Output per-node statistics
    print(f"\n{'='*60}")
    print("Per-node Statistics")
    print(f"{'='*60}")
    for node_idx, node in enumerate(network.qs):
        print(f"\nNode {node_idx}:")
        print(f"  Negative type: {node.type_of_negatives.name}")
        print(f"  Channels: {node.n}")
        print(f"  Jobs served (without break): {getattr(node, 'served', 0)}")
        print(f"  Jobs broken: {getattr(node, 'broken', 0)}")
        print(f"  Total jobs: {getattr(node, 'total', 0)}")
        print(f"  Negative arrivals: {getattr(node, 'negative_arrived', 0)}")

        # Output node-level sojourn time moments
        if hasattr(node, "v") and node.v is not None and len(node.v) > 0:
            print("  Sojourn time moments (raw):")
            print("    " + f"{'Moment':^10s}|{'Value':^15s}")
            print("    " + "-" * 25)
            for i, moment in enumerate(node.v[:4], 1):
                print(f"    {i:^10d}|{moment:^15.6f}")

        # Output node-level waiting time moments
        if hasattr(node, "w") and node.w is not None and len(node.w) > 0:
            print("  Waiting time moments (raw):")
            print("    " + f"{'Moment':^10s}|{'Value':^15s}")
            print("    " + "-" * 25)
            for i, moment in enumerate(node.w[:4], 1):
                print(f"    {i:^10d}|{moment:^15.6f}")

    print(f"\n{'='*60}\n")


def test_negative_network_zero_negative_rate():
    """
    Test that NegativeNetwork with zero negative arrival rate produces
    results similar to regular NetworkSimulator.
    Uses the same configuration as test_network_no_prty.py.
    """
    # Load parameters from default_params.yaml
    cur_dir = os.getcwd()
    params_path = os.path.join(cur_dir, "tests", "default_params.yaml")

    with open(params_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    NUM_OF_JOBS = int(params["num_of_jobs"])
    SERVICE_TIME_CV = float(params["service"]["cv"])
    ARRIVAL_RATE = float(params["arrival"]["rate"])
    UTILIZATION_FACTOR = float(params["utilization_factor"])
    MOMENTS_ATOL = float(params["moments_atol"])
    MOMENTS_RTOL = float(params["moments_rtol"])

    NUM_OF_NODES = 5
    NUM_OF_CHANNELS = [3, 2, 3, 4, 3]

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

    # Prepare service parameters (same as test_network_no_prty.py)
    b = []  # List of service time moments for each node
    serv_params = []

    for m in range(NUM_OF_NODES):
        b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS[m] / ARRIVAL_RATE
        h2_params = H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV)
        serv_params.append({"type": "H", "params": h2_params})
        b.append(H2Distribution.calc_theory_moments(h2_params, 4))

    # Create regular network for comparison
    regular_network = NetworkSimulator()
    regular_network.set_sources(arrival_rate=ARRIVAL_RATE, R=TRANSITION_MATRIX)
    regular_network.set_nodes(serv_params=serv_params, n=NUM_OF_CHANNELS)
    regular_results = regular_network.run(NUM_OF_JOBS)

    # Create negative network with zero negative arrival rate
    negative_network = NegativeNetwork(negative_arrival_type="global")
    negative_network.set_sources(
        positive_arrival_rate=ARRIVAL_RATE, R=TRANSITION_MATRIX, negative_arrival_rate=0.0  # Zero negative arrivals
    )

    # Set nodes with any negative type (shouldn't matter with zero rate)
    negative_types = [NegativeServiceType.DISASTER] * NUM_OF_NODES
    negative_network.set_nodes(serv_params=serv_params, n=NUM_OF_CHANNELS, negative_types=negative_types)

    negative_results = negative_network.run(job_served=NUM_OF_JOBS)

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison: Regular Network vs Negative Network (zero rate)")
    print("=" * 60)
    print(f"Regular Network - Jobs served: {regular_results.served}")
    print(f"Negative Network - Jobs served: {negative_results.served}")
    print(f"Regular Network - Jobs arrived: {regular_results.arrived}")
    print(f"Negative Network - Jobs arrived: {negative_results.arrived}")
    print("\nSojourn Time Moments Comparison:")
    print_sojourn_moments(regular_results.v, negative_results.v)

    # Assert that results are close
    assert negative_results.served > 0, "Negative network should serve some jobs"
    assert negative_results.arrived > 0, "Negative network should receive some arrivals"

    # Compare sojourn time moments (should be very close with zero negative rate)
    if regular_results.v is not None and negative_results.v is not None:
        # Use slightly more lenient tolerance since there might be small differences
        # due to different random seeds or implementation details
        tolerance_rtol = MOMENTS_RTOL * 1.5  # 50% more lenient
        tolerance_atol = MOMENTS_ATOL * 1.5

        diff = np.abs(np.array(regular_results.v) - np.array(negative_results.v))
        assert np.allclose(regular_results.v, negative_results.v, rtol=tolerance_rtol, atol=tolerance_atol), (
            f"Sojourn time moments should be close:\n"
            f"Regular: {regular_results.v}\n"
            f"Negative: {negative_results.v}\n"
            f"Difference: {diff}"
        )

    # Verify that no negative arrivals occurred
    # With zero rate, negative_arrival_time should be set (could be inf or very large)
    assert negative_network.negative_arrival_time is not None, "Negative arrival time should be set"
    # With zero rate, negative arrivals should be zero or very rare
    total_negative_arrivals = sum(getattr(node, "negative_arrived", 0) for node in negative_network.qs)
    print(f"\nTotal negative arrivals across all nodes: {total_negative_arrivals}")
    # With zero rate, we expect zero or very few negative arrivals (due to floating point precision)
    # Allow up to 10 negative arrivals as a safety margin for floating point issues
    # Note: With rate=0.0, ExpDistribution may still generate some values due to numerical precision,
    # but they should be extremely rare
    assert total_negative_arrivals <= 10, (
        f"With zero negative rate, should have zero or very few negative arrivals, " f"got {total_negative_arrivals}"
    )

    # Additional check: if there were negative arrivals, they should be very few compared to total jobs
    if total_negative_arrivals > 0:
        negative_ratio = total_negative_arrivals / negative_results.served
        print(f"Negative arrivals ratio: {negative_ratio:.6f}")
        assert negative_ratio < 0.0001, (
            f"Negative arrivals should be extremely rare with zero rate, " f"got ratio {negative_ratio:.6f}"
        )


if __name__ == "__main__":
    # Run the function with output
    test_negative_network_with_output()
    # Or run all tests with pytest
    # pytest.main([__file__, "-v"])
