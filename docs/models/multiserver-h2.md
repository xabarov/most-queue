# Multiserver H₂ systems (Takahashi–Takami method)

[🇷🇺 Русская версия](multiserver-h2.ru.md) · [← Model catalog](../models.md) ·
[← FIFO systems](fifo.md)

**In plain words:** H₂ (a "mixture of two exponentials") is a universal building block: by fitting
its parameters to the mean and coefficient of variation, one can approximate almost any real
distribution (for CV < 1 — with complex-valued parameters). The Takahashi–Takami method is an
iterative numerical algorithm that solves multi-server phase-type models exactly, for either side
(arrivals, service, or both) approximated this way.

### H₂/M/c

**Description:** Multi-server system with hyperexponential arrivals (H₂) and exponential service (M). Uses the simplified algorithm of §7.6.1 (formulas for z_j, x_j, t_{j,i}, level 0).

**Calculator class:** `H2MnCalc`

**Example:**

```python
from most_queue.theory.fifo.gmc_takahasi import H2MnCalc
from most_queue.random.distributions import H2Distribution

calc = H2MnCalc(n=3)

h2_params = H2Distribution.get_params_by_mean_and_cv(1.0, 1.2, is_clx=True)  # mean, cv
#
# For CV<1 use the complex fit: is_clx=True.
# Important: the `QsSim` simulator cannot generate H2 with complex parameters,
# so comparison with simulation is only possible when the parameters are real-valued.
calc.set_sources(h2_params)

calc.set_servers(b=2.0)  # mean service time

results = calc.run()
```

### H₂/H₂/c

**Description:** Multi-server system with hyperexponential arrivals and hyperexponential service. Uses the algorithm of §7.6.2 (CH7).

**Calculator class:** `HkHkNCalc`

**Example:**

```python
from most_queue.theory.fifo.hkhk_takahasi import HkHkNCalc
from most_queue.random.distributions import H2Distribution

calc = HkHkNCalc(n=3, k=2)

h2_arr = H2Distribution.get_params_by_mean_and_cv(1.0, 1.2)
# For CV<1 use the complex fit: is_clx=True (the parameters may then be complex).
calc.set_sources(u=[h2_arr.p1, 1 - h2_arr.p1], lam=[h2_arr.mu1, h2_arr.mu2])

h2_srv = H2Distribution.get_params_by_mean_and_cv(2.0, 1.2)
calc.set_servers(y=[h2_srv.p1, 1 - h2_srv.p1], mu=[h2_srv.mu1, h2_srv.mu2])

results = calc.run()
```

**Note on CV<1:** for \(CV<1\) the H₂ approximation uses a *complex fit* (complex-valued parameters).
The `QsSim` simulator does not generate H₂ with complex parameters, so for validation it is
convenient to compare the calculation (H₂ complex fit) with a simulation of an equivalent
`Gamma` model matched by mean/CV (see the tests in `tests/test_tt_vs_sim_gamma_cvl1.py`).

### M/H₂/c

**Description:** Multi-server system with Poisson arrivals and hyperexponential service. Uses the Takahashi–Takami numerical method with complex parameters.

**Calculator class:** `MGnCalc`

**Example:**

```python
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.random.distributions import H2Distribution

calc = MGnCalc(n=5)

calc.set_sources(l=2.0)

h2_params = H2Distribution.get_params_by_mean_and_cv(mean=2.0, cv=1.2, is_clx=True)
calc.set_servers(h2_params)

results = calc.run()
```
