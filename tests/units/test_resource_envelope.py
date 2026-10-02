"""Scalar-resource feasibility and independent half-open occupancy integrals."""

import numpy as np
import pytest

from most_queue.sim.utils.resource_envelope import ResourceRequest, assess_envelope, observed_occupancy


def test_projection_preserves_requested_node_count_not_ceil_gpu_count():
    request = ResourceRequest(1, 2)
    assert request.demand("gpu_pool") == 1
    assert request.demand("exclusive_nodes") == 2
    assert ResourceRequest(4, 2, 2).demand("exclusive_nodes") == 2
    with pytest.raises(ValueError):
        request.demand("placement")


@pytest.mark.parametrize("arguments", [(0, 1), (1, 0), (9, 1), (True, 1), (1.0, 1), (1, 1, 0), (1, 1, True)])
def test_invalid_resource_requests(arguments):
    with pytest.raises(ValueError):
        ResourceRequest(*arguments)


def test_infeasible_is_audited_not_filtered_or_clipped():
    report = assess_envelope(4, [5, 1], running=[3, 2], waiting=[8])
    assert not report["feasible"]
    assert report["reasons"] == ["individual_demand_exceeds_capacity", "initial_running_exceeds_capacity"]
    assert report["maximum_individual_demand"] == 8
    assert report["initial_running_demand"] == 5
    assert report["oversized"] == {"running": 0, "waiting": 1, "arrivals": 1}
    assert report["counts"] == {"running": 2, "waiting": 1, "arrivals": 2}


def test_queue_sum_is_not_a_capacity_feasibility_constraint():
    assert assess_envelope(4, [4] * 100, waiting=[4] * 100, running=[2, 2])["feasible"]


@pytest.mark.parametrize("capacity,arrivals", [(True, [1]), (0, [1]), (1, []), (1, [False]), (2, [1.5])])
def test_bad_envelope_arguments(capacity, arrivals):
    with pytest.raises(ValueError):
        assess_envelope(capacity, arrivals)


def test_occupancy_integral_and_simultaneous_release_arrival():
    # Occupancy: [0,1)=2, [1,2)=3, [2,3)=4, [3,4)=3.
    result = observed_occupancy([(0, 2, 2), (1, 3, 1), (2, 4, 3)], [4, 2])
    assert result["peak_concurrent_demand"] == 4
    assert result["maximum_individual_demand"] == 3
    assert result["resource_time"] == 12
    assert result["capacities"] == [
        {
            "capacity": 4,
            "compatible_with_recorded_intervals": True,
            "seconds_above_capacity": 0,
            "excess_resource_time": 0,
        },
        {
            "capacity": 2,
            "compatible_with_recorded_intervals": False,
            "seconds_above_capacity": 3,
            "excess_resource_time": 4,
        },
    ]


def test_occupancy_counts_idle_gaps_but_no_fabricated_load():
    result = observed_occupancy([(1, 2, 1), (8, 10, 2)], [1])
    assert result["first_start"] == 1 and result["last_end"] == 10
    assert result["resource_time"] == 5
    assert result["capacities"][0]["seconds_above_capacity"] == 2


@pytest.mark.parametrize(
    "intervals,capacities",
    [
        ([], [1]),
        ([(0, 1, 1)], []),
        ([(0, 1, 1)], [1, 1]),
        ([(0, 1, 1)], [True]),
        ([(1, 1, 1)], [1]),
        ([(2, 1, 1)], [1]),
        ([(0, np.inf, 1)], [1]),
        ([(0, 1, 0)], [1]),
        ([(True, 2, 1)], [1]),
        ([(-1, 2, 1)], [1]),
    ],
)
def test_invalid_occupancy_contract(intervals, capacities):
    with pytest.raises(ValueError):
        observed_occupancy(intervals, capacities)
