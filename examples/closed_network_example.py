"""
Classic capacity-planning question answered by a closed queueing network:
how many terminals can one server sustain before response time explodes?

Each of N terminals "thinks" for 5 s on average (a delay node with infinitely
many servers), then sends a request that the server processes in 0.2 s on
average (a single-channel FCFS node). Exact MVA gives the throughput and the
mean response time for every N; the response-time knee appears near the
saturation point N* = 1 + think/service.
"""

import numpy as np

from most_queue.theory.networks.closed_network import ClosedNetworkCalc

THINK_TIME = 5.0
SERVICE_TIME = 0.2

# Terminals (delay node 0) <-> server (FCFS node 1)
ROUTING = np.array([[0.0, 1.0], [1.0, 0.0]])


def response_time(n_terminals: int) -> tuple[float, float]:
    """Mean response time (server sojourn per request) and throughput."""
    calc = ClosedNetworkCalc(method="mva")
    calc.set_sources(R=ROUTING, N=n_terminals)
    calc.set_nodes(b=[THINK_TIME, SERVICE_TIME], n=[None, 1])
    res = calc.run()
    return res.v_node[1], res.throughput


if __name__ == "__main__":
    saturation = 1 + THINK_TIME / SERVICE_TIME
    print(f"Saturation point N* = {saturation:.0f} terminals\n")
    print(f"{'N':>4} {'throughput':>11} {'response, s':>12}")
    for n in (1, 5, 10, 15, 20, 25, 26, 30, 40, 60):
        resp, x = response_time(n)
        print(f"{n:>4} {x:>11.3f} {resp:>12.3f}")
