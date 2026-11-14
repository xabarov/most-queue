# Contributing to Most-Queue

Useful contributions include reproducible bug reports, examples that answer a
practical question, documentation improvements, and new queueing models. You do
not need to implement a solver to help. English and Russian issues are welcome.

## Set up a checkout

Use Python 3.10 or later in a virtual environment:

```bash
git clone https://github.com/xabarov/most-queue.git
cd most-queue
python -m venv .venv
# Linux/macOS; on Windows use .venv\Scripts\activate
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Run a small example without a GPU or external services:

```bash
python -m examples.llm_burst_recovery
python -m pytest tests/units/test_transient_occupancy.py -q
```

Examples live in the repository and are not included in the published wheel.
See the [model catalog](docs/models.md) before adding another implementation.

## Report a problem or propose a model

Use [GitHub Issues](https://github.com/xabarov/most-queue/issues/new/choose).
For bugs, include the smallest runnable example, package version or Git commit,
parameters, seed for simulation, and expected versus actual output. If a result
disagrees with a paper, give its equation/page and the assumptions being compared.

For a new model, explain the question it answers, arrival and service laws,
discipline, capacity/buffer rules, and desired metrics. Link the primary source
and suggest a known special case or independent simulator for validation.
For hardware measurements, state the engine/version, hardware, request-length
distribution, timestamp definitions, and which data may be redistributed.

## Make a focused change

Follow `set_sources()` / `set_servers()` / `run()` and the result structures used
by the relevant family. Document exact formulas, approximations, and numerical
truncation separately. A numerical cutoff must not silently become a loss buffer.
New mathematical methods need a derivation or primary source and a meaningful
validation oracle; checking only a second implementation of the same equations
is insufficient.

Add an English model page and runnable example; a Russian page is welcome.
The [burst-recovery page](docs/models/transient-occupancy.md) illustrates the
expected assumptions, API, diagnostics, and limitations.

Format changed Python files with `black` and `isort`, and run `pylint` on affected
modules. Run relevant tests during development, then the required repository checks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/ -m 'not slow' -q -n auto
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest tests/ -q -n auto
git diff --check
```

For a pull request, describe the concrete behavior change, the assumptions and
limitations, and the checks actually run. Include failures and unresolved
questions. See [project conventions](docs/PROJECT.md),
[infrastructure](docs/INFRASTRUCTURE.md), and [Definition of Done](docs/DOD.md).

The [LLM/GPU roadmap](docs/roadmaps/llm_gpu_growth_roadmap.md) lists proposed work.
Small contributions are especially useful: run an example on another platform,
improve an explanation, or contribute a minimal, redistributable benchmark
fixture with documented metrics.
