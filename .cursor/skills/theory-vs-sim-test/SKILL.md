---
name: theory-vs-sim-test
description: >-
  Author or fix a sim-vs-theory pytest in most-queue: load default_params.yaml,
  set up matched theory + QsSim, compare moments via np.allclose with shared
  tolerances, print io.tables for diagnostics. Use whenever a new theory
  calculator is added, an existing test fails, or simulation/theory results need
  cross-validation in this repo.
---

# Theory-vs-Sim Test Pattern

Every numerical model in `most_queue/theory/**` is validated against
discrete-event simulation (`QsSim` and friends). The pattern below is the
project-wide convention — follow it for any new test.

## Required imports

```python
import os
import numpy as np
import yaml

from most_queue.io.tables import print_sojourn_moments, print_waiting_moments, probs_print
from most_queue.sim.base import QsSim                    # or specialised sim
from most_queue.theory.<area>.<model> import <ModelCalc>  # the theory class
from most_queue.random.distributions import H2Distribution  # or whichever fits the test
```

## Load shared tolerances from `tests/default_params.yaml`

Always read these constants — never hard-code tolerances in the test:

```python
params_path = os.path.join(os.getcwd(), "tests", "default_params.yaml")
with open(params_path, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

ARRIVAL_RATE       = float(params["arrival"]["rate"])
SERVICE_TIME_CV    = float(params["service"]["cv"])
NUM_OF_JOBS        = int(params["num_of_jobs"])
UTILIZATION_FACTOR = float(params["utilization_factor"])
MOMENTS_ATOL       = float(params["moments_atol"])
MOMENTS_RTOL       = float(params["moments_rtol"])
PROBS_ATOL         = float(params["probs_atol"])
PROBS_RTOL         = float(params["probs_rtol"])
ERROR_MSG          = params["error_msg"]
```

## Standard test body

```python
def test_<model>():
    # 1. Match service distribution moments between theory and sim.
    b1 = UTILIZATION_FACTOR * NUM_OF_CHANNELS / ARRIVAL_RATE
    h2_params = H2Distribution.get_params_by_mean_and_cv(b1, SERVICE_TIME_CV)
    b = H2Distribution.calc_theory_moments(h2_params, 5)

    # 2. Theory.
    calc = <ModelCalc>(...)            # any constructor args your model needs
    calc.set_sources(ARRIVAL_RATE)     # or (params, kendall) for non-Markovian arrivals
    calc.set_servers(b)
    calc_results = calc.run()
    assert abs(UTILIZATION_FACTOR - calc_results.utilization) < PROBS_ATOL

    # 3. Simulation with the SAME distribution.
    qs = QsSim(NUM_OF_CHANNELS)
    qs.set_sources(ARRIVAL_RATE, "M")
    qs.set_servers(h2_params, "H")
    sim_results = qs.run(NUM_OF_JOBS)

    # 4. Diagnostics (always print — helps debugging when CI flips).
    print_waiting_moments(sim_results.w, calc_results.w)
    print_sojourn_moments(sim_results.v, calc_results.v)
    probs_print(sim_results.p, calc_results.p, 10)

    # 5. Assertions. Compare RAW moments. Higher moments are noisy at
    #    NUM_OF_JOBS=100k ? typically validate the first 2 only.
    assert np.allclose(sim_results.w[:2], calc_results.w[:2],
                       rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(sim_results.v[:2], calc_results.v[:2],
                       rtol=MOMENTS_RTOL, atol=MOMENTS_ATOL)
    assert np.allclose(np.array(sim_results.p[:10]),
                       np.array(calc_results.p[:10]),
                       atol=PROBS_ATOL, rtol=PROBS_RTOL), ERROR_MSG
```

## Why each piece is non-negotiable

| Step | Reason |
|------|--------|
| Same distribution moments on both sides | Otherwise the comparison is meaningless. Use the distribution's `get_params_by_mean_and_cv` + `calc_theory_moments`. |
| `default_params.yaml` constants | Project-wide tolerances — changing them silently is forbidden (see rule `pytest-quality.mdc`). |
| Compare only first 2 moments by default | Sample 3rd/4th moments require >1M jobs to converge; doing fewer means false negatives. |
| Always print `io.tables` | When a test fails in CI you need the actual numbers, not just the assertion. |

## When defaults are not enough

- **High utilization (? ? 0.9)** or **heavy-tailed service (Pareto, large CV)**:
  raise `NUM_OF_JOBS` to 500k–1M *in the test only*. Do **not** lower
  `MOMENTS_RTOL` instead.
- **Multi-channel (n > 1)** with priorities or vacations: prefer the
  specialised simulator (`QsSim` subclass in `most_queue/sim/`) over composing
  with hooks.
- **Non-Markovian arrivals**: use `Gamma` for the sim side and pass the same
  `(mean, cv)` to the theory class.

## Diagnostic helpers (in `most_queue/io/tables`)

- `print_waiting_moments(sim.w, calc.w)`
- `print_sojourn_moments(sim.v, calc.v)`
- `print_raw_moments(sim_moments, calc_moments)`
- `probs_print(sim.p, calc.p, n_states)`

## Reference tests in this repo

| File | What it validates |
|------|-------------------|
| `tests/test_mg1_calc.py` | Canonical M/G/1 vs sim — minimal, copy-as-template |
| `tests/test_gi_m_n_calc.py` | Non-Markovian arrivals (Gamma) |
| `tests/test_mgn_tt.py` | Takahashi–Takami M/G/n |
| `tests/test_mg1_warmup.py` | Vacation/warm-up extension to the pattern |
