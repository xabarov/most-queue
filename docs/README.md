# Most-Queue Library Documentation

[🇷🇺 Русская версия](README.ru.md)

![Queue](../assets/most-queue-nano1.jpeg)

**Most-Queue** is a Python library for simulation and numerical analysis of queueing systems and queueing networks.

## 📚 Documentation Guide

### Getting Started
- **[Quick Start](getting_started.md)** — installation and first usage examples
- **[Core Concepts](concepts.md)** — queueing theory and terminology
- **[Thesaurus](tesaurus.md)** — glossary of queueing theory terms

### Main Sections
- **[Queueing System Simulation](simulation.md)** — guide to modeling queueing systems (including the *SizeBasedQsSim* section — SRPT, SJF, SPJF, etc.)
- **[Numerical Methods](calculation.md)** — computing queueing system characteristics with analytical methods (including the *Size-based M/G/1 calculators* section)
- **[Distributions](distributions.md)** — reference for supported distributions
- **[Queueing Models](models.md)** — catalog of supported queueing system types

### For Developers
- **[Project Overview](PROJECT.md)** (in Russian) — repository layout, conventions, where to find the theory
- **[Infrastructure](INFRASTRUCTURE.md)** (in Russian) — building, testing, code quality, publishing
- **[Definition of Done](DOD.md)** (in Russian) — task completion criteria
- **[Epics](epics/README.md)** (in Russian) — planning of major development directions
- **[Model Gap Analysis](models_gap_analysis.md)** (in Russian) — what is implemented, what is missing, priorities (EPIC-001)

### Specialized Topics
- **[SRPT / SPJF: Calculation Methods and Verification](srpt_spjf_methods.md)** — formulas, numerical scheme (grids, Simpson, quad), comparison with `SizeBasedQsSim` and tests (per the [roadmap](roadmaps/srpt_spjf_roadmap.md) (in Russian))
- **[Queueing Networks](networks.md)** — simulation and calculation of queueing networks
- **[Priority Systems](priorities.md)** — systems with priority service
- **[Disasters (negative customers): Time Characteristics](negative_disasters_time_characteristics.md)** — computing \(W, V, V_{served}, V_{broken}\) via LST
- **[RCS (negative customers): Time Characteristics](negative_rcs_time_characteristics.md)** — computing \(W, V, V_{served}, V_{broken}\) in the RCS model
- **[Advanced Examples](examples.md)** — practical cases and complex scenarios

## 🎯 Key Features

### Simulation
- Modeling of various queueing system types (M/M/c, M/G/1, GI/M/c, and others)
- Support for a variety of arrival and service time distributions
- Systems with priorities, vacations, negative customers
- Fork-Join and Split-Join systems
- Queueing networks

### Numerical Methods
- Analytical calculation of steady-state characteristics
- Methods for various queueing system types
- High calculation accuracy
- Fast results

### Result Analysis
- Moments of waiting and sojourn times
- System state probabilities
- Utilization factor
- Visualization and result tables

## 📦 Installation

```bash
pip install most-queue
```

Or install from the repository:

```bash
pip install -e .
```

## 🔍 Quick Example

```python
from most_queue.sim.base import QsSim

# Create an M/M/1 system simulator
qs = QsSim(num_of_channels=1)

# Configure the arrival process (Poisson with rate 0.5)
qs.set_sources(0.5, "M")

# Configure the service (exponential with rate 1.0)
qs.set_servers(1.0, "M")

# Run the simulation for 10000 jobs
results = qs.run(10000)

# Get the results
print(f"Mean waiting time: {results.w[0]:.4f}")
print(f"Mean sojourn time: {results.v[0]:.4f}")
print(f"Utilization factor: {results.utilization:.4f}")
```

## 📖 Library Structure

The library consists of two main modules:

- **`most_queue.sim`** — queueing system simulation module
- **`most_queue.theory`** — numerical calculation methods module

## 🧪 Examples and Tests

- Usage examples can be found in the [`tests/`](../tests/) folder
- [Jupyter tutorials](../tutorials/README.md) — counter-intuitive queueing insights for engineers

## 📝 Version

Current version: **2.9**

## 🔗 Useful Links

- [GitHub Repository](https://github.com/xabarov/most-queue)
- [Issues and Suggestions](https://github.com/xabarov/most-queue/issues)

## 👥 Contact

For questions and suggestions: xabarov1985@gmail.com

---

**Note**: This documentation is in English. Russian versions of the pages are available via the language-switcher links at the top of each page.
