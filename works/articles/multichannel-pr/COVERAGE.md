# RDR paper — model coverage & correctness audit

Paper: Harchol-Balter, Osogami, Scheller-Wolf, Wierman, *"Multi-Server Queueing Systems with
Multiple Priority Classes"* (M/PH/k, m preemptive-resume priority classes; methods RDR & RDR-A).

This audit maps every **numerical** model in the paper to a library implementation and records the
measured accuracy. Cross-checks use the exact truncated-CTMC solver `MMkPriorityExact` (noise-free,
exact for exponential service) where tractable, and the discrete-event `PriorityQueueSimulator`
otherwise.

## Methods available in the library

| Method | Class | Scope |
|---|---|---|
| Exact CTMC (reference) | `MMkPriorityExact` | M/M/k, any m, exponential, small state space |
| RDR-A (aggregated) | `RDRAPriorityCalc` | M/M/k, any m, per-class exponential — fast |
| Two-class RDR (exact busy-period) | `MMnPR2ClsBusyApprox` | M/M/k, 2 classes |
| Three-class RDR | `MM2BusyApprox3Classes` | M/M/2, 3 classes |
| Two-class M/PH/k RDR | `MPhNPrty` | M/PH/k, 2 classes (high PH, low exp) |

## Coverage of the paper's numerical figures

| Figure | Configuration | Paper method | Library | Status | Measured |
|---|---|---|---|---|---|
| **Fig 5a** | M/M/2, m=4, identical classes | RDR | `RDRAPriorityCalc`, `MMkPriorityExact` | ✅ covered | RDR-A vs **exact CTMC** <0.5% (ρ≤0.7) |
| **Fig 9** | 2-class short/long, SMART & STUPID, k∈{1,2,4} | RDR | `MMnPR2ClsBusyApprox` | ✅ covered | vs simulation ~2% on E[T] |
| **Fig 7** | M/PH/k, m=2, high PH, varying C²_H, k∈{1,2,4} | RDR | `MPhNPrty` | ✅ k=1,2 / ⚠️ k=4 | k=1,2 within 1–5% vs sim; k=4 extreme regime → clean error |
| **Fig 5b, 6** | M/PH/2, m=4, **all** classes PH (C² up to 128) | RDR-A | `RDRAPriorityPH` (+ `MPhPhK2Class`) | ✅ covered | vs FCFS-resume sim 0–1.4% (C²=8, ρ≤0.75) |
| **Fig 10** | M/PH/2, m=4 SMART, different means, PH | RDR-A vs MK-N | `RDRAPriorityPH` | ✅ (RDR-A part) | per-class means via PH RDR-A |
| Fig 8 | BB approximation error study | BB (not RDR) | n/a | — | out of scope (BB is a rival method) |
| **§2.4** | response-time **variance** / higher moments | tagged-job passage times | `MMkPriorityExact(with_variance=True)` | ✅ exponential | highest class = closed form exactly; tagged mean = Little to 1e-6 |
| Fig 1–4, 11 | illustrative Markov chains | — | — | — | no numerics |

### Verified numbers (samples)

- **Fig 5a** (M/M/2, m=4, μ=1, load split evenly), RDR-A vs exact CTMC:
  - ρ=0.5: classes agree to 0.0% (1.016 / 1.117 / 1.358 / 1.842).
  - ρ=0.7: worst class 0.4% (1.032 / 1.248 / 1.862 / 3.69).
- **Fig 9** (short mean 1, long mean 10, load balanced), overall E[T] calc vs sim:
  - k=2 SMART 1.909/1.903, STUPID 2.828/2.825; k=4 SMART 1.389/1.389, STUPID 1.656/1.617.
- **m=3 heterogeneous μ** (0.3/0.3/0.3, μ 1.2/1.0/0.8), RDR-A vs exact CTMC: 0.8466 / 1.1301 /
  1.8626 — agreement <0.1% (also exposes and confirms the fixed two-class convergence).

## PH service in the multiclass case (Fig 5b, 6, 10) — now covered

When **every** class has a phase-type (non-exponential) service and the target is a **low** class,
the target's own service variance shapes its response time. This is handled exactly by
**`MPhPhK2Class`** (two-class M/PH/PH/k) and, for m > 2 classes, by **`RDRAPriorityPH`** which
aggregates the higher classes into one PH stream and calls `MPhPhK2Class`.

**Key idea that makes it tractable:** under preemptive-**resume** a preempted PH job freezes its
phase, so a naive CTMC would track every queued job's phase (explosion). But on k servers only the
≤ k in-service jobs (plus ≤ k frozen ones) carry a live phase; tracking the low class's active
jobs as an **age-ordered tuple** keeps FCFS-resume exact with a finite state space.

**Validation (three independent references):**
- both classes exponential ⇒ matches `MMkPriorityExact` exactly;
- high PH / low exponential ⇒ matches `MPhNPrty` exactly;
- both PH ⇒ matches an independent FCFS-resume work-based simulation (~0.3%);
- `RDRAPriorityPH` reduces to `MPhPhK2Class` at m=2 and, for m=4 all-PH (C²=8), sits within
  0–1.4% of the FCFS-resume simulation across ρ.

### Multi-server discipline effect (found here)

Unlike the single-server case, for **multi-server + PH** the within-class service order affects even
the **mean** response time. `MPhPhK2Class`/`RDRAPriorityPH` implement the standard **FCFS-resume**
order (verified above). The library's `PriorityQueueSimulator` reinserts a preempted job at the
back of its class queue, so its PH multi-server means differ from FCFS-resume — this is why PH
validation uses the controlled FCFS-resume work simulation, not the default simulator.

## Solver-robustness fixes (found while re-verifying Fig 7)

Re-running Fig 7 across its full C² and server range exposed real solver defects, now fixed:

- **k=1 `LinAlgError: Singular matrix`** — a regression from the passage-time G-matrix
  optimization: the closed-form `_G_calc` fails when its coefficient matrix is singular (a level
  state that cannot reach the level below on its own). Now falls back to functional iteration.
- **Infinite loop / silent garbage in the base Takahashi-Takami solver** (`MGnCalc.run`) — the
  iteration had **no maximum-iteration cap**, so a numerical breakdown (e.g. `1/x` with `x→0` on
  the larger k=4 chain) spun forever, or "converged" to a non-physical distribution returning a
  low-class mean of ≈ −7·10⁴⁰. Added: an iteration cap, a non-finite/overflow guard, and a
  physical-probability check — the solver now **fails loudly** (`FloatingPointError`) instead of
  hanging or emitting garbage. This protects all ~15 `MGnCalc`-based models.

The k=4 fixed-capacity, high-C² regime remains numerically hard for the multi-server
Takahashi-Takami method itself (large, ill-conditioned chain); it now errors cleanly rather than
misleading. Tests: `test_m_ph_n_robustness.py`.

## Within-class discipline of the priority simulator (found while validating §2.4)

While validating the exact response-time **variance**, the discrete-event `PriorityQueueSimulator`
disagreed with the exact value on the second moment (but not the mean). Cause: when a high-priority
arrival preempts a low-priority job in service, the simulator reinserts the preempted job at the
**back** of its class queue, i.e. it does *not* preserve arrival order (standard FCFS-resume).

Consequences:
- **Means are unaffected** (work-conserving; discipline-invariant) — all mean validations stand.
- **Higher moments differ.** `MMkPriorityExact(with_variance=True)` computes the standard
  FCFS-resume second moment; verified against the M/M/1 closed form (highest class, exact), a
  controlled FCFS-resume simulation (~1%), and the tagged-mean = Little's-law identity (1e-6).

This is a documented property of the simulator, not a bug in the exact solver.
