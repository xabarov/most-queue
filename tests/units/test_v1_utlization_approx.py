from most_queue.theory.utils.utilization_approx import v1_on_utilization_approx, find_delta_utilization


if __name__ == "__main__":
    poly1 = v1_on_utilization_approx(3, 1)
    poly2 = v1_on_utilization_approx(5, 1)
    load1 = 0.9
    load2 = 0.7
    min_u = find_delta_utilization(poly1, poly2, load1, load2)
    print(f"Minimum delta utilization: {min_u}")
