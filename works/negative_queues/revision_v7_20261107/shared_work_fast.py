"""The same finite shared-work model with incomplete block preconditioning."""

import works.negative_queues.revision_v7_20261107.ph_queue as engine
from works.negative_queues.revision_v7_20261107.preconditioned_stationary_ilu import solve
from works.negative_queues.revision_v7_20261107.stationary_check import calculate as check


def calculate(cv, classes, stages, levels, n=3, lam=1, delta=0.02, mode="disaster", **options):
    if mode != "disaster":
        raise ValueError("Independent own-service convolution is only used for common restart")
    previous = engine.solve_jump_chain
    engine.solve_jump_chain = solve
    try:
        result = check(
            cv, classes, stages, levels, n, lam, delta, tolerance=1e-12, progress=options.get("progress", False)
        )
    finally:
        engine.solve_jump_chain = previous
    result["solver"] = "independent phase assembly; incomplete block-Jacobi preconditioned stationary equations"
    return result
