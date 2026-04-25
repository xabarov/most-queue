---
name: add-theory-calculator
description: >-
  Add a new analytical/numerical queueing-theory calculator to most-queue:
  subclass BaseQueue, wire set_sources / set_servers, return QueueResults from
  run(), reuse shared utilities, expose via theory/__init__.py. Use whenever the
  user asks to implement a new theoretical model (M/G/1 variant, priority,
  network, fork-join, vacations, …).
---

# Adding a Theory Calculator

All analytical models in this repo follow the same contract. Stick to it so
the calculator interoperates with `io.tables`, the test pattern (see skill
`theory-vs-sim-test`), and downstream tooling.

## Where to put the file

| Model type | Directory |
|------------|-----------|
| Single class FIFO (M/G/1, GI/M/n, …) | `most_queue/theory/fifo/` |
| Priorities | `most_queue/theory/priority/` |
| Vacations / warm-up / cooling | `most_queue/theory/vacations/` |
| Negative customers / disasters | `most_queue/theory/negative/` |
| Fork–join | `most_queue/theory/fork_join/` |
| Closed networks | `most_queue/theory/closed/` |
| Open networks | `most_queue/theory/networks/` |
| Impatience | `most_queue/theory/impatience/` |
| Batch arrivals | `most_queue/theory/batch/` |
| Size-based scheduling (SRPT/SJF/SPJF/…) | `most_queue/theory/srpt/` |

## Required class skeleton

```python
from most_queue.structs import QueueResults
from most_queue.theory.base_queue import BaseQueue


class MyModelCalc(BaseQueue):
    """One-line summary. Cite the source paper / formula in the docstring."""

    def __init__(self, n: int = 1, calc_params=None, buffer=None):
        super().__init__(n=n, calc_params=calc_params, buffer=buffer)
        # model-specific state

    def set_sources(self, ...):                   # accept ?, or (params, kendall)
        ...
        self.is_sources_set = True                # MUST set the flag

    def set_servers(self, ...):                   # accept moments b, or (params, kendall)
        ...
        self.is_servers_set = True                # MUST set the flag

    def get_w(self, num_of_moments: int = 4) -> list[float]: ...
    def get_v(self, num_of_moments: int = 4) -> list[float]: ...
    def get_p(self) -> list[float]: ...

    def run(self, num_of_moments: int = 4) -> QueueResults:
        self._check_if_servers_and_sources_set()
        start = self._measure_time()

        w = self.get_w(num_of_moments)
        v = self.get_v(num_of_moments)
        p = self.get_p()

        result = QueueResults(v=v, w=w, p=p, utilization=<your ?>)
        self._set_duration(result, start)
        return result
```

## Conventions to follow

1. **Set the flags.** `is_sources_set` / `is_servers_set` must be set to
   `True` at the end of the corresponding setter — `_check_if_servers_and_sources_set`
   relies on them.
2. **Return RAW moments.** `w` and `v` are raw (non-central) moments; the test
   pattern compares raw. If you compute centrals internally, convert via
   `most_queue.theory.utils.moments`.
3. **Always set `utilization`** on `QueueResults` — the standard test asserts
   `abs(UTILIZATION_FACTOR - result.utilization) < PROBS_ATOL`.
4. **Time the run.** Use `start = self._measure_time()` and
   `self._set_duration(result, start)` — `io.tables` printers display this.
5. **Return the right results dataclass:**
   - `QueueResults` — single-class FIFO and most special models
   - `PriorityResults` — priorities
   - `NetworkResults` — networks
   - `VacationResults` — vacations / warm-up / cooling
   - `NegativeArrivalsResults` — negative customers / disasters

## Reuse, don't reinvent

Before writing math, check `most_queue/theory/utils/`:

| Module | Use for |
|--------|---------|
| `utils/conv.py` (`conv_moments`) | Convolution of two moment sequences |
| `utils/moments.py` | Raw ? central moments, factorial moments |
| `utils/transforms.py` (`lst_exp`, `lst_h2`, `lst_gamma`, …) | Laplace–Stieltjes transforms |
| `utils/busy_periods.py` | Busy-period moments via numerical differentiation of LST |
| `utils/residual.py` | Residual service-time distributions |
| `utils/flow_sum.py` | Combining renewal flows |
| `utils/q_poisson_arrival_calc.py` | PASTA-style state probabilities |
| `utils/conditional.py` | Conditional moments (e.g. `min(Exp, H2)`) |

## Distribution interface (when you need a PDF/CDF or to align with sim)

`most_queue.random.distributions` provides classes (Exp, H2, Gamma, Cox,
Erlang, Pareto, Normal, …) with the contract:

```python
ClassDist.get_params_by_mean_and_cv(mean, cv)
ClassDist.calc_theory_moments(params, k)        # first k raw moments
inst = ClassDist(params, generator)
inst.generate()                                 # sample one
```

Use this to keep the simulation and the theory aligned (see skill
`theory-vs-sim-test`).

## Checklist before opening a PR

- [ ] Subclassed `BaseQueue`; both `is_*_set` flags set in setters
- [ ] `run()` returns the appropriate dataclass with `utilization` and
      `duration`
- [ ] Reused `theory/utils/*` rather than reimplementing convolution / LST
- [ ] Added `from .<module> import MyModelCalc` to `theory/__init__.py` if
      the class is part of the public API
- [ ] New test file in `tests/` following the `theory-vs-sim-test` skill
- [ ] `pytest tests/` is fully green (see `.cursor/rules/pytest-quality.mdc`)
- [ ] At least 2 distributions tested (e.g. Exp + H2 or Gamma)
