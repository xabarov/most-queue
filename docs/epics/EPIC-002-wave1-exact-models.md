# EPIC-002: Волна 1 — точные базовые модели

- **Статус:** done (2026-07-08)
- **Создан:** 2026-07-08
- **Roadmap:** [../roadmaps/wave1_exact_models_roadmap.md](../roadmaps/wave1_exact_models_roadmap.md)

## Цель

Закрыть базовые ожидания от библиотеки ТМО семью точными моделями малой сложности,
переиспользующими готовую машинерию. Итог — библиотека покрывает «джентльменский набор»
классики и завершает линейку дисциплин обслуживания (size-aware + blind).

## Контекст

Приоритет **must** по [gap-анализу](../models_gap_analysis.md) (EPIC-001). Все модели —
точные формулы (кроме двухмоментных аппроксимаций GI/G, явно маркируемых), реализация
~2–3 недели суммарно.

## Задачи

- [x] 1. Erlang B/C + нечувствительность M/G/n/0 (`ErlangBCalc`, `ErlangCCalc`,
      `theory/fifo/erlang.py`; тесты `test_erlang.py`, включая insensitivity-тест
      с Gamma-обслуживанием; схема `figures/loss.png`) — 2026-07-08
- [x] 2. M/M/∞, M/G/∞ (`MGInfCalc`, `theory/fifo/m_g_inf.py`; тест `test_m_g_inf.py`
      с Gamma-обслуживанием CV≠1) — 2026-07-08
- [x] 3. GI/G/1 и GI/G/m аппроксимации: Kingman, KLB, Allen–Cunneen
      (`theory/fifo/gi_g_approx.py`; тесты `test_gi_g_approx.py`: точные частные случаи
      KLB=Полячек–Хинчин и AC=Erlang C, сетка сверки с симуляцией, свойство верхней
      границы Кингмана). Kimura отложена: интерполяция требует точных W для D/M/s
      и M/D/s — 2026-07-08
- [x] 4. Классические vacation-модели M/G/1: multiple vacations и N-policy
      (`theory/vacations/mg1_vacations.py`; симулятор расширен: флаг
      `is_multiple_vacations` и `NPolicyQueueSim` в `sim/vacations.py`; тесты
      `test_mg1_vacations.py` — сверка с симуляцией, вырожденные случаи, ручная
      проверка моментов добавки). **Single vacation отложен**: формула добавки
      требует сверки с Takagi Vol.1 §2.2 — не реализовывать по памяти;
      семантику симуляции уже поддерживает `cold` без прогрева — 2026-07-08
- [x] 5. M/G/1 PS и LCFS-PR (`theory/fifo/mg1_ps.py`, `theory/fifo/mg1_lcfs_pr.py`;
      новые мини-симуляторы `ProcessorSharingSim` и `LcfsPRSim` в
      `sim/single_server_disciplines.py`; тесты `test_mg1_ps_lcfs.py` — геометрические
      вероятности с Gamma-обслуживанием (нечувствительность), моменты sojourn LCFS-PR
      = busy period). Старшие моменты PS (Яшков/Отт) — отложены — 2026-07-08
- [x] 6. M/G/1 FB/LAS (`theory/srpt/mg1_fb.py` на сетках `_SizeBasedCalcBase`;
      симулятор `FBSim` с батч-разделением в `sim/single_server_disciplines.py` —
      в событийный `SizeBasedQsSim` FB не встраивается корректно, ранг непрерывно
      меняется; тесты `test_mg1_fb.py`: сверка с симуляцией, exp-обслуживание = PS,
      FB < PS при DHR) — 2026-07-08
- [x] 7. Ненадёжный прибор M/G/1 (`theory/vacations/mg1_unreliable.py` — completion time
      через кумулянты, до 4 моментов; симулятор `UnreliableQueueSim` в `sim/unreliable.py`;
      тесты `test_mg1_unreliable.py`: сверка с симуляцией отказов, ξ=0 ⇒ M/G/1 с точностью
      1e-12, тождество E[C]=b₁(1+ξr₁)) — 2026-07-08

## Критерии готовности (DoD эпика)

По каждой модели — DoD «новая модель СМО» ([../DOD.md](../DOD.md)): BaseQueue-контракт,
первоисточник в docstring, тест теория-vs-симуляция, запись в `docs/models.md`.
Аппроксимации явно маркированы. Сравнительный ноутбук дисциплин
(FCFS/PS/FB/SJF/SRPT/SPJF) в `tutorials/` — выполнено:
[tutorials/disciplines_comparison.ipynb](../../tutorials/disciplines_comparison.ipynb).

## Результаты

Все 7 пунктов выполнены за 2026-07-08. Итого:

- **10 новых калькуляторов**: `ErlangBCalc`, `ErlangCCalc`, `MGInfCalc`, `GIG1ApproxCalc`,
  `GIGmApproxCalc`, `MG1MultipleVacationsCalc`, `MG1NPolicyCalc`, `MG1PSCalc`,
  `MG1LcfsPrCalc`, `MG1FbCalc`, `MG1UnreliableCalc` — все с первоисточниками (DOI)
  в docstring, все pylint 10/10.
- **5 новых симуляторов**: `NPolicyQueueSim`, `ProcessorSharingSim`, `LcfsPRSim`, `FBSim`,
  `UnreliableQueueSim` + режим `is_multiple_vacations` в `VacationQueueingSystemSimulator`.
- **21 новый тест** (6 файлов), включая точные частные случаи (KLB=Полячек–Хинчин,
  Allen–Cunneen=Erlang C, ξ=0 ⇒ M/G/1, N=1 ⇒ M/G/1, exp-обслуживание: FB=PS)
  и проверку нечувствительности Севастьянова симуляцией с Gamma-обслуживанием.
- **Каталог**: записи для всех моделей в стиле EPIC-005 (суть + схемы, включая
  data-driven график замедления `figures/slowdown.png`).
- **Ноутбук**: `tutorials/disciplines_comparison.ipynb` — 9 дисциплин, таблица E[V],
  график замедления, кросс-чек с симуляцией; выполнен, выводы зафиксированы.

Отложено (задокументировано в задачах): формула Кимуры (нужен точный D/M/s),
single vacation (сверка с Takagi Vol.1), старшие моменты PS (Яшков/Отт).

## Замечания по ходу работ

При подготовке ноутбука обнаружена и исправлена ошибка в документации: среднее время
пребывания LCFS-PR совпадает с PS (b₁/(1−ρ), нечувствительность), но НЕ с FCFS —
у FCFS среднее зависит от b₂ (при CV=1.5: FCFS 3.354 vs PS/LCFS-PR 2.333).
