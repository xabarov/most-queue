"""Independent physical CTMC + tagged-job oracle for Л8, NOT production code.

The stationary queue is capped by rejecting arrivals at Q=cap. The tagged
process retains actual busy counts, jobs ahead and jobs behind (the latter
can be capped at max(T)-1 without changing threshold decisions). No geometric
ansatz, free/busy aggregate, PH compression or production helper is used.
Palm initialization is approximate because the underlying stationary CTMC
is capped; cap refinement and boundary mass are reported separately.
"""

import itertools
import math
from collections import deque

import numpy as np
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import expm_multiply, spsolve


def physical_oracle(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    counts, rates, thresholds, capacities, arrival_rate, cap=70, times=(0, 1, 3)
):
    """Return stationary queue and physically tagged W/V moments and tails."""
    order = sorted(range(len(counts)), key=lambda i: (thresholds[i], i))

    def settle(busy, q):
        busy = list(busy)
        while True:
            eligible = [i for i in order if busy[i] < counts[i] and q >= thresholds[i]]
            if not eligible:
                return tuple(busy), q
            sigma = eligible[0]
            busy[sigma] += 1
            q -= min(q, capacities[sigma])

    states = [
        (vec, q)
        for vec in itertools.product(*(range(n + 1) for n in counts))
        for q in range(cap + 1)
        if settle(vec, q) == (vec, q)
    ]
    index = {s: i for i, s in enumerate(states)}
    matrix = np.zeros((len(states), len(states)))
    for idx, (busy, q) in enumerate(states):
        if q < cap:
            matrix[idx, index[settle(busy, q + 1)]] += arrival_rate
        for sigma, n in enumerate(busy):
            if n:
                free = list(busy)
                free[sigma] -= 1
                matrix[idx, index[settle(free, q)]] += n * rates[sigma]
        matrix[idx, idx] -= matrix[idx].sum()
    equation = csr_matrix(matrix.T).tolil()
    equation[-1, :] = 1
    rhs = np.zeros(len(states))
    rhs[-1] = 1
    mass = spsolve(equation.tocsr(), rhs)
    queue_mean = float(sum(p * q for p, (_, q) in zip(mass, states)))
    boundary = float(sum(p for p, (_, q) in zip(mass, states) if q == cap))

    def tagged_settle(busy, ahead, behind):
        busy = list(busy)
        while True:
            q = ahead + 1 + behind
            eligible = [i for i in order if busy[i] < counts[i] and q >= thresholds[i]]
            if not eligible:
                return (tuple(busy), ahead, behind), None
            sigma = eligible[0]
            take = min(q, capacities[sigma])
            if take > ahead:
                return None, sigma
            busy[sigma] += 1
            ahead -= take

    tagged = []
    tag_index = {}
    initial = {}
    atom = np.zeros(len(counts))
    todo = deque()

    def add(state):
        if state not in tag_index:
            tag_index[state] = len(tagged)
            tagged.append(state)
            todo.append(state)
        return tag_index[state]

    for p, (busy, q) in zip(mass, states):
        state, absorption = tagged_settle(busy, q, 0)
        if absorption is None:
            idx = add(state)
            initial[idx] = initial.get(idx, 0.0) + p
        else:
            atom[absorption] += p
    transitions = []
    behind_cap = max(thresholds) - 1
    while todo:
        state = todo.popleft()
        busy, ahead, behind = state
        events = []
        events.append((busy, ahead, min(behind + 1, behind_cap), arrival_rate))
        for sigma, n in enumerate(busy):
            if n:
                free = list(busy)
                free[sigma] -= 1
                events.append((tuple(free), ahead, behind, n * rates[sigma]))
        row = {}
        exits = np.zeros(len(counts))
        for vec, r, ell, rate in events:
            target, absorption = tagged_settle(vec, r, ell)
            if absorption is not None:
                exits[absorption] += rate
            elif target != state:
                idx = add(target)
                row[idx] = row.get(idx, 0.0) + rate
        transitions.append((tag_index[state], row, exits))
    rr, cc, vv = [], [], []
    exits = np.zeros((len(tagged), len(counts)))
    for idx, row, leave in transitions:
        exits[idx] = leave
        for j, rate in row.items():
            rr.append(idx)
            cc.append(j)
            vv.append(rate)
        rr.append(idx)
        cc.append(idx)
        vv.append(-sum(row.values()) - leave.sum())
    waiting = csr_matrix((vv, (rr, cc)), shape=(len(tagged), len(tagged)))
    beta = np.zeros(len(tagged))
    for idx, value in initial.items():
        beta[idx] = value
    response = bmat([[waiting, csr_matrix(exits)], [None, -diags(rates)]], format="csr")
    response_beta = np.concatenate([beta, atom])

    def moments(generator, pi):
        rhs = np.ones(len(pi))
        values = []
        for k in range(1, 4):
            rhs = spsolve(-generator, rhs)
            values.append(float(math.factorial(k) * (pi @ rhs)))
        return values

    return {
        "mean_queue": queue_mean,
        "boundary_mass": boundary,
        "queue_pmf": [float(sum(p for p, (_, q) in zip(mass, states) if q == k)) for k in range(cap + 1)],
        "w": moments(waiting, beta),
        "v": moments(response, response_beta),
        "zero_wait_probability": float(atom.sum()),
        "served_probabilities": (atom + beta @ spsolve(-waiting, exits)).tolist(),
        "waiting_tails": [float(beta @ expm_multiply(waiting * t, np.ones(len(beta)))) for t in times],
        "response_tails": [
            float(response_beta @ expm_multiply(response * t, np.ones(len(response_beta)))) for t in times
        ],
        "stationary_states": len(states),
        "tagged_states": len(tagged),
    }
