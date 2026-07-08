# SRPT / SPJF in Most-Queue: Calculation Methods and Comparison with Simulation

[🇷🇺 Русская версия](srpt_spjf_methods.ru.md)

This page complements the [roadmap](roadmaps/srpt_spjf_roadmap.md) and the short sections in [numerical methods](calculation.md) and [simulation](simulation.md). It gathers the **formulas** the calculators work with, the **practical numerical scheme** (grids, integration), and **how the results are verified against `SizeBasedQsSim`**.

Sources: Schrage–Miller (1966), Conway–Maxwell–Miller (continuous size-based priority), Mitzenmacher (2020) for SPJF; write-ups in the repository: [SPJF.md](../works/queueing_systems_review/SRPT/SPJF.md), [shrage.md](../works/queueing_systems_review/SRPT/shrage.md).

---

## 1. What Has Been Added to the Library (per the roadmap)

| Area | Component | Purpose |
|--------|-----------|------------|
| Theory | `most_queue.theory.srpt.MG1SrptCalc` | M/G/1, **SRPT** |
| Theory | `MG1SjfCalc` | M/G/1, **SJF** (continuous SPT) |
| Theory | `MG1PsjfCalc` | M/G/1, **PSJF** |
| Theory | `MG1SpjfCalc` + `Predictor` | M/G/1, **SPJF** based on the \((X,Y)\) model |
| Theory | `most_queue.theory.srpt.utils.load_below`, `predictor` | \(\rho_x\), \(\rho'_y\), marginal \(g_Y\), "perfect" and noisy predictors |
| Simulation | `most_queue.sim.size_based.SizeBasedQsSim` | Single-channel queue: FCFS, SJF, PSJF, SRPT, SPJF, PSPJF, SPRPT |
| Simulation | `Task.original_size`, `predicted_size`, `service_remaining` | Size at arrival, prediction, remaining work upon preemption |
| Simulation | `PrioritySizeQueue`, sim-layer predictors | Rank-ordered heap; `PerfectSimPredictor`, noise via `sim.utils.predictor` |

The calculators inherit the common `BaseQueue` pattern: `set_sources(λ)`, `set_servers(params, kendall_notation)` — same as `MG1Calc`, but for the size a **numerical** density/CDF is built via `build_pdf_cdf` (distributions from `most_queue.random.distributions` with a known parameterization are supported).

---

## 2. Analytical Formulas (what exactly is computed)

Notation: arrival rate \(\lambda\), size PDF/CDF \(f,F\), \(\mathbb{E}[S]=b_0\), \(\mathbb{E}[S^2]=b_1\). Partial load due to jobs of size **at most** \(x\):

$$
\rho_x = \lambda \int_0^x t\,f(t)\,dt .
$$

(In the literature this is also frequently written as \(\rho_{\le x}\) for continuous size-based priority.)

### 2.1. SRPT (Schrage–Miller)

Conditional mean **sojourn time** of a job of size \(x\):

$$
\mathbb{E}[T^{\mathrm{SRPT}}(x)] = \frac{\lambda \int_0^x t^2 f(t)\,dt \;+\; \lambda\, x^2 \bigl(1-F(x)\bigr)}{2\bigl(1-\rho_x\bigr)^2} \;+\; \int_0^x \frac{dt}{1-\rho_t}.
$$

Unconditional means:

$$
\mathbb{E}[T^{\mathrm{SRPT}}] = \int_0^\infty f(x)\,\mathbb{E}[T^{\mathrm{SRPT}}(x)]\,dx, \qquad \mathbb{E}[W^{\mathrm{SRPT}}] = \mathbb{E}[T^{\mathrm{SRPT}}] - b_0 .
$$

In the code, `MG1SrptCalc` uses an equivalent form of the first term (see the class docstring); the second term is a separately accumulated integral over the grid.

### 2.2. SJF (continuous size-based priority, non-preemptive)

$$
\mathbb{E}[W^{\mathrm{SJF}}(x)] = \frac{\lambda\, b_1}{2\bigl(1-\rho_x\bigr)^2}, \qquad \mathbb{E}[W^{\mathrm{SJF}}] = \int_0^\infty f(x)\,\mathbb{E}[W^{\mathrm{SJF}}(x)]\,dx, \qquad \mathbb{E}[T^{\mathrm{SJF}}] = \mathbb{E}[W^{\mathrm{SJF}}] + b_0 .
$$

### 2.3. PSJF (preemption based on the **original** size)

$$
\mathbb{E}[T^{\mathrm{PSJF}}(x)] = \frac{\lambda \int_0^x t^2 f(t)\,dt}{2\bigl(1-\rho_x\bigr)^2} \;+\; \frac{x}{1-\rho_x},
$$

$$
\mathbb{E}[T^{\mathrm{PSJF}}] = \int_0^\infty f(x)\,\mathbb{E}[T^{\mathrm{PSJF}}(x)]\,dx, \qquad \mathbb{E}[W^{\mathrm{PSJF}}] = \mathbb{E}[T^{\mathrm{PSJF}}] - b_0 .
$$

The difference from SRPT: there is no \(\int_0^x dt/(1-\rho_t)\) term, since the rank does not "track" the remaining work.

### 2.4. SPJF (Mitzenmacher 2020)

Joint density of \((X,Y)\) — the true size and the prediction; effective load due to jobs with prediction \(\le y\):

$$
\rho'_y = \lambda \int_0^y \!\! \int_0^\infty t\, g(t,z)\,dt\,dz .
$$

Conditional mean **waiting time** for a fixed prediction \(y\):

$$
\mathbb{E}[W^{\mathrm{SPJF}}(y)] = \frac{\lambda\, b_1}{2\bigl(1-\rho'_y\bigr)^2}, \qquad \mathbb{E}[W^{\mathrm{SPJF}}] = \int_0^\infty g_Y(y)\,\mathbb{E}[W^{\mathrm{SPJF}}(y)]\,dy, \qquad \mathbb{E}[T^{\mathrm{SPJF}}] = \mathbb{E}[W^{\mathrm{SPJF}}] + b_0 ,
$$

where \(g_Y\) is the marginal density of \(Y\). With **perfect** predictions (\(Y=X\)) the result coincides with SJF (`PerfectPredictor` on the theory side / `PerfectSimPredictor` on the simulation side).

---

## 3. How This Is Computed in Code (numerically)

### 3.1. Common Base for SRPT / SJF / PSJF — `_SizeBasedCalcBase`

File: `most_queue/theory/srpt/_base.py`.

1. **Upper bound on \(x\)**  
   `x_max = upper_bound(cdf, p=1e-7)` — the practical "tail" of the distribution; the mass beyond it is negligible.

2. **Hybrid grid over \([0, x_{\max}]\)**  
   Some of the nodes form a **logarithmic** neighborhood of 0 (for sharp PDFs like Gamma/Pareto), the rest are **uniform** on the remainder. The constant `_N_GRID = 4000` sets the number of nodes.

3. **One-shot integrals over the grid** (`scipy.integrate.cumulative_trapezoid`):
   - \(\rho_x\) as \(\lambda \int_0^x t f(t)\,dt\);
   - \(\int_0^x t^2 f(t)\,dt\);
   - for SRPT: \(\int_0^x dt/(1-\rho_t)\) over the grid nodes (with \(\rho_t\) clipped slightly below 1 to avoid numerical overflow);
   - CDF values at the same nodes.

4. **Fast lookups**  
   For an arbitrary \(x\), **`np.interp`** against the precomputed arrays is used — this replaces nested quadratures "at every point of the outer integral".

5. **Unconditional averaging over \(X\)** — computing

   $$
   \int_0^{x_{\max}} f(x)\, h(x)\,dx
   $$

   where \(h(x)\) is the conditional \(\mathbb{E}[T(x)]\) or \(\mathbb{E}[W(x)]\) via interpolation. Implementation: **`scipy.integrate.simpson`** on **the same grid**, i.e. the vector `pdf(xs) * h(xs)` and `simpson(..., x=xs)`.

   **Why Simpson rather than an outer `quad`:** the integrand contains factors of the form \((1-\rho_x)^{-2}\); as \(\rho \to 1\), adaptive `quad` often emits roundoff warnings on the tail near \(x_{\max}\). Integration over a **fixed** dense grid with Simpson is more stable and consistent with the already computed \(\rho_x\).

### 3.2. Separately: `MG1SpjfCalc`

Here there is no shared grid over \(x\) for the outer integral over \(y\): the marginal \(g_Y(y)\) and \(\rho'_y\) are supplied by a **`Predictor`** object (`marginal_y_pdf`, `load_below_y`). The unconditional \(\mathbb{E}[W]\) is computed with **`scipy.integrate.quad`** over \(y \in [0, \infty)\) with tightened `epsabs`, `epsrel`, `limit` (see `mg1_spjf.py`), because the integral runs over a different variable and over a density supplied by the predictor.

### 3.3. Stability Under High Load

- Theory: dense grid + Simpson for the outer averaging over \(X\) (SRPT/SJF/PSJF); dedicated regression tests at \(\rho \in \{0.95, 0.99\}\) for the Mitzenmacher–Shahout scenario (see `tests/test_mg1_predictions_table.py`).
- Simulation: the variance of the \(\mathbb{E}[T]\) estimates, especially for SRPT, is large; the tables from the paper use hundreds of thousands to millions of jobs (see `examples/srpt_table.py`).

---

## 4. The `SizeBasedQsSim` Simulation and Its Correspondence to Theory

The roadmap idea: **sample-at-arrival** — the size \(X\) (and, if needed, the prediction \(Y\)) is known at the moment of arrival; the actual service time equals the sampled size (the remainder is tracked in `service_remaining`).

- **`set_servers(params, notation)`** in `SizeBasedQsSim` sets **the same size distribution** that is used to build \(f\) in the theory (via the same Kendall / H2 parameters, etc.).
- **`set_sources(λ, "M")`** — a Poisson flow with the same \(\lambda\) as in the calculator's `set_sources`.
- The discipline is set with `discipline=...`; preemption is implemented by rank comparison and returning the task to the heap with the updated remainder.

Then, for fixed \((\lambda, f)\) and the same discipline, the **simulation means** (e.g., `results.v[0]`, `results.w[0]`) must converge to the **analytical** ones as the number of jobs grows; any discrepancy is only statistical error plus the discretization of the theory.

**PSPJF / SPRPT:** the roadmap notes that, in the first iteration, **the analytics for these may be absent from the package**; simulation is available, theory is an extension point. The "theory vs sim" comparison in the tests targets SRPT, SJF, PSJF, SPJF (+ FCFS as the baseline).

---

## 5. How Exactly We Compare Against Simulation

1. **Unit tests** (`tests/test_mg1_srpt.py`, `test_mg1_sjf.py`, `test_mg1_psjf.py`, `test_mg1_spjf.py`): the same scenario (e.g., H₂ with a given mean and CV), `MG1*Calc.run()` versus `SizeBasedQsSim(...).run(N)`; tolerances **`MOMENTS_RTOL` / `MOMENTS_ATOL`** from `tests/default_params.yaml`, the number of jobs **`NUM_OF_JOBS`** chosen so that the typical error of the mean fits within the tolerance.

2. **Inequalities and sanity checks** (`tests/test_srpt_vs_fcfs.py`): for instance, \(\mathbb{E}[T^{\mathrm{SRPT}}] \le \mathbb{E}[T^{\mathrm{FCFS}}]\) for the tested parameters.

3. **Regression against the paper's table** (`tests/test_mg1_predictions_table.py`): a fixed prediction model \(Y \mid X=x \sim \mathrm{Exp}(1/x)\), several values of \(\rho\); sim + (where available) theory against the published numbers.

4. **Example for reproducing the table and plots**: `examples/srpt_table.py` — theory, simulation, and reference values in a single script (see also the tutorial `tutorials/srpt_basics.ipynb`).

Recommendation for your own experiments: fix the **`np.random.Generator`** (`sim.generator = np.random.default_rng(seed)`), increase \(N\) at large \(\rho\), and compare against theory where it is implemented.

---

## 6. Related Documentation Sections

- [Core Concepts — size-based terminology](concepts.md)
- [Models — the M/G/1 SRPT/SJF/PSJF/SPJF rows](models.md)
- [Numerical Methods — concise formulas and API examples](calculation.md) (section *Size-based M/G/1 calculators*)
- [Simulation — `SizeBasedQsSim`, predictors, slowdown](simulation.md)
