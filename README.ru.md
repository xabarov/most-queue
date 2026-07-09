<div align="center">

# Most-Queue

**Теория очередей на Python: точные аналитические решатели в паре с имитационным моделированием — 40+ моделей от M/M/1 до SRPT-планирования, отпусков, отрицательных заявок и сетей очередей.**

[🇬🇧 English version](README.md)

[![Tests](https://github.com/xabarov/most-queue/actions/workflows/tests.yml/badge.svg)](https://github.com/xabarov/most-queue/actions/workflows/tests.yml)
[![PyPI version](https://img.shields.io/pypi/v/most-queue)](https://pypi.org/project/most-queue/)
[![Python versions](https://img.shields.io/pypi/pyversions/most-queue)](https://pypi.org/project/most-queue/)
[![License](https://img.shields.io/pypi/l/most-queue)](https://github.com/xabarov/most-queue/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/most-queue)](https://pepy.tech/project/most-queue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21268402.svg)](https://doi.org/10.5281/zenodo.21268402)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/xabarov/most-queue)](https://github.com/xabarov/most-queue/commits/main)

<img src="https://raw.githubusercontent.com/xabarov/most-queue/main/assets/most-queue-nano1.jpeg" alt="Most-Queue" width="720"/>

</div>

## Почему Most-Queue?

- **Аналитика и симуляция вместе.** Почти у каждого аналитического калькулятора есть парный
  дискретно-событийный симулятор, а тесты сверяют их между собой. Быстрые точные числа —
  и способ их проверить.
- **Модели, которых нет в других open-source пакетах**: аналитика size-based дисциплин
  (SRPT, SJF, PSJF, SPJF с ML-предсказаниями размеров, FB/LAS), vacation-модели M/G/1,
  отрицательные заявки (RCS / disaster), ненадёжные приборы, многоканальные фазовые системы
  методом Такахаси–Таками (включая CV < 1 через complex-fit H₂).
- **Моменты, а не только средние**: начальные моменты времени ожидания и пребывания,
  вероятности состояний, загрузка — с единым API `set_sources() / set_servers() / run()`.
- **Чистый Python + NumPy/SciPy**, ставится через pip, лицензия MIT.

## Установка

```bash
pip install most-queue
```

Требуется Python ≥ 3.9. Для визуализации сетей может понадобиться системный пакет `graphviz`:

```bash
sudo apt-get install -y graphviz
```

## Быстрый старт: теория против симуляции за 20 строк

```python
from most_queue.theory.fifo.mmnr import MMnrCalc
from most_queue.sim.base import QsSim

# Аналитический расчёт M/M/3 с конечной очередью
calc = MMnrCalc(n=3, r=100)
calc.set_sources(l=2.0)
calc.set_servers(mu=1.0)
theory = calc.run()

# Та же система — симуляцией
sim = QsSim(3)
sim.set_sources(2.0, "M")
sim.set_servers(1.0, "M")
experiment = sim.run(100_000)

print(f"Среднее ожидание: теория {theory.w[0]:.3f} vs симуляция {experiment.w[0]:.3f}")
# Среднее ожидание: теория 0.444 vs симуляция 0.448
```

## Кто платит за дисциплину обслуживания?

Посчитано калькуляторами самой библиотеки — условное замедление `E[T(x)]/x` по размерам
заявки для FCFS, PS, FB (не знает размеров) и SRPT (знает):

<img src="https://raw.githubusercontent.com/xabarov/most-queue/main/docs/figures/slowdown.png" alt="Замедление по размерам заявки" width="720"/>

Исполняемое сравнение **9 дисциплин** — в ноутбуке
[`tutorials/disciplines_comparison.ipynb`](tutorials/disciplines_comparison.ipynb).

## Что внутри

| Семейство | Модели | Метод |
|---|---|---|
| Классика FIFO | M/M/c, M/M/c/r, Erlang B/C, M/G/1, GI/M/c, M/D/c, Eₖ/D/c, M/G/∞ | точный |
| Многоканальные фазовые | M/H₂/c, H₂/M/c, H₂/H₂/c (CV < 1 через complex-fit) | Такахаси–Таками |
| Size-based дисциплины | M/G/1 SRPT, SJF, PSJF, SPJF (с предикторами), FB/LAS, PS, LCFS-PR | точный (Schrage–Miller, Mitzenmacher) |
| Приоритеты | M/G/1 PR/NP мультикласс, M/G/c PR/NP, M/Ph/c PR | точный / инвариантная аппроксимация |
| Отпуска и прогрев | M/G/1 multiple vacations, N-policy, прогрев/охлаждение/задержка (M/Ph/c) | Fuhrmann–Cooper, Такахаси–Таками |
| Отрицательные заявки | M/G/1 и M/G/c с RCS или disaster | точный / Такахаси–Таками |
| Надёжность | M/G/1 с отказами и ремонтами | Avi-Itzhak–Naor |
| Матрично-аналитические | MAP/PH/1, M/PH/1, PH/PH/1, MAP/M/c — коррелированный (bursty) вход, одно- и многоканальные | QBD, logarithmic reduction |
| Retrial и уходы | M/M/1 и M/G/1 retrial (орбита), Erlang-A (M/M/n+M) со staffing | точный / Falin–Templeton |
| GI/G аппроксимации | GI/G/1, GI/G/m — среднее ожидание | Kingman, KLB, Allen–Cunneen |
| Batch, нетерпение, закрытые | Mˣ/M/1, M/M/1+M, Engset | точный |
| Параллельное обслуживание | Fork-Join, Split-Join | марковский / порядковые статистики |
| Сети | открытые, с приоритетами, с отрицательными заявками, оптимизация маршрутизации | декомпозиция |

У каждой модели — объяснение простым языком и схема в
[иллюстрированном каталоге моделей](docs/models.ru.md).

## Документация и туториалы

- 📖 [Документация](docs/README.ru.md) — концепции, руководства по расчётам и симуляции (основная версия — [английская](docs/README.md))
- 🎓 [Jupyter-туториалы](tutorials/) — от первой M/M/1 до Такахаси–Таками и SRPT
- 🗺 [Планы развития](docs/epics/README.md) — что дальше (BMAP/G/1, MAP-fitting, многоканальный retrial, polling)
- 🧪 [Тесты](tests/) — каждая модель сверена с симуляцией; запуск: `pytest -m "not slow"`

## Области применения

Планирование мощностей облачных сервисов и ЦОД · расчёт штата колл-центров ·
производственные линии · телеком-трафик · планирование ресурсов в здравоохранении ·
исследования планировщиков (SRPT/LAS с ML-предсказаниями размеров).

## Новости

- **2026** — **Матрично-аналитический стек**: PH-распределения и MAP-потоки
  (`most_queue.random.map_ph`), QBD-решатель с logarithmic reduction, точные калькуляторы
  MAP/PH/1 / M/PH/1 / PH/PH/1, MAP- и PH-источники в симуляторе; плюс retrial-очереди (орбита)
  и Erlang-A со staffing-помощником. См. [`tutorials/map_ph_correlation.ipynb`](tutorials/map_ph_correlation.ipynb).
- **2026** — Волна точной классики: Erlang B/C, M/G/∞, GI/G-аппроксимации, vacation-модели
  M/G/1 (multiple vacations, N-policy), PS, LCFS-PR, FB/LAS, ненадёжный прибор — каждая
  с парным симулятором и тестами. Иллюстрированный каталог моделей со схемами.
- **2026** — Size-based scheduling для M/G/1: аналитика SRPT / SJF / PSJF / SPJF
  (`most_queue.theory.srpt`) и симулятор `SizeBasedQsSim` с предикторами; воспроизведение
  таблицы Mitzenmacher–Shahout (2025) в тестах. См. [методы SRPT/SPJF](docs/srpt_spjf_methods.md).
- **2026 (препринт)** — Расчёт многоканальных СМО с отрицательными заявками методом
  Такахаси–Таками: [препринт и код воспроизведения](works/negative_queues/).
- **2026** — Метод Такахаси–Таками для FIFO-моделей **H₂/M/c** и **H₂/H₂/c**; улучшено API
  H₂-параметров; тесты валидации при CV≈1.05 и CV<1 (complex-fit vs Gamma-симуляция).
- **2025 (статья)** — Численный расчёт многоканальной СМО с разогревом, охлаждением и
  задержкой начала охлаждения: Лохвицкий В.А., Хабаров Р.С., Яковлев Е.Л. //
  Авиакосмическое приборостроение. 2025. № 1. С. 44–57.
  DOI [10.25791/aviakosmos.1.2025.1456](https://doi.org/10.25791/aviakosmos.1.2025.1456).

## Участие в разработке

Issues и pull requests приветствуются! Открывайте [issue](https://github.com/xabarov/most-queue/issues)
с багами и запросами моделей. Соглашения разработки: [docs/PROJECT.md](docs/PROJECT.md),
критерии готовности: [docs/DOD.md](docs/DOD.md). Контакт: xabarov1985@gmail.com

## Цитирование

Если Most-Queue пригодился в исследовании — сошлитесь на него (см. [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{most_queue,
  author  = {Khabarov, Roman},
  title   = {Most-Queue: queueing theory calculations and simulation in Python},
  url     = {https://github.com/xabarov/most-queue},
  doi     = {10.5281/zenodo.21268402},
  license = {MIT}
}
```

## Лицензия

[MIT](LICENSE) © Роман Хабаров
