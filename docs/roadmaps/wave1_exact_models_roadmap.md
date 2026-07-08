# Roadmap: волна 1 — точные базовые модели

> Эпик: [EPIC-002](../epics/EPIC-002-wave1-exact-models.md). Основание: приоритет **must**
> в [gap-анализе](../models_gap_analysis.md). Все модели — точные формулы малой сложности,
> ложащиеся на существующую машинерию (`fifo/mg1.py`, `theory/utils/busy_periods`, `theory/srpt/`).

## Состав (7 моделей, порядок = порядок реализации)

### 1. Erlang B/C + нечувствительность M/G/n/0

- **Теория:** B(n,a) через устойчивую рекурсию `B(n,a)=aB(n-1,a)/(n+aB(n-1,a))`; Erlang C из B;
  блокировка M/G/n/0 = M/M/n/0 (Севастьянов 1957, doi:10.1137/1102005; Jagerman, BSTJ 1974).
- **Реализация:** `theory/fifo/erlang.py` — `ErlangBCalc`, `ErlangCCalc`; обратные задачи
  (мин. n при целевой блокировке). Выход: p, вероятность ожидания, w/v для Erlang C.
- **Валидация:** `QsSim` c r=0 (потери) и r=∞; частный случай MMnrCalc.

### 2. M/M/∞ и M/G/∞

- **Теория:** стационарное N ~ Poisson(λE[S]), нечувствительность; sojourn = service.
- **Реализация:** `theory/fifo/m_g_inf.py` — `MGInfCalc` (p — пуассоновские, моменты v = моменты b).
- **Валидация:** `QsSim` с n >> λE[S].

### 3. GI/G/1 и GI/G/m двухмоментные аппроксимации

- **Теория:** Kingman (1961, doi:10.1017/s0305004100036094) — верхняя граница/heavy-traffic;
  поправка Krämer–Langenbach-Belz (ITC-8, 1976); Allen–Cunneen (Allen, 1990) для m каналов.
  Kimura (Manag. Sci. 1986, doi:10.1287/mnsc.32.6.751) **отложена**: это интерполяция по
  точным W для D/M/s, M/D/s, M/M/s, а точного решателя D/M/s в библиотеке нет
  (кандидат в бэклог после волны 1).
- **Реализация:** `theory/fifo/gi_g_approx.py` — `GIG1ApproxCalc`, `GIGmApproxCalc`;
  вход — первые два момента a и b (естественно для API); в результатах помечать
  approximation-статус. В docstring — области точности (KLB: погрешность единицы % при cv² ≤ ~2).
- **Валидация:** симулятор GI/G/n на сетке (ρ, cv_a, cv_b) — по образцу таблиц
  `test_mg1_predictions_table.py`; сверка частных случаев с M/M/n, M/G/1, GI/M/1 (точными).

### 4. Классические vacation-модели M/G/1

- **Теория:** декомпозиция Fuhrmann–Cooper (Oper. Res. 1985, doi:10.1287/opre.33.5.1117):
  W = W_{M/G/1} + остаточное время отпуска; multiple/single vacations, N-policy, setup
  (Doshi 1986, doi:10.1007/bf01149327; Takagi 1991).
- **Реализация:** `theory/vacations/mg1_vacations.py` — `MG1MultipleVacationsCalc`,
  `MG1SingleVacationCalc`, `MG1NPolicyCalc`; вход — моменты b и моменты отпуска v_moments.
  Моментная арифметика: свёртка моментов через готовые утилиты `theory/utils/moments`.
- **Валидация:** `VacationQueueingSystemSimulator`; кросс-чек с `MG1WarmCalc` на вырожденных
  случаях; предельный случай нулевого отпуска = MG1Calc.

### 5. M/G/1 PS и LCFS-PR

- **Теория:** PS: p_n = (1-ρ)ρ^n (BCMP 1975, doi:10.1145/321879.321887), E[T|x] = x/(1-ρ)
  (Kleinrock 1967, doi:10.1145/321386.321388), дисперсия — Ott (1984, doi:10.2307/3213646) /
  Яшков (1987, doi:10.1007/bf01182931), для первой итерации — среднее + условное среднее +
  slowdown. LCFS-PR: sojourn = busy period → моменты из рекурсий Такача
  (уже в `theory/utils/busy_periods`).
- **Реализация:** `theory/fifo/mg1_ps.py` (`MG1PSCalc`), `theory/fifo/mg1_lcfs_pr.py`
  (`MG1LcfsPrCalc`).
- **Валидация:** PS-дисциплина в симуляторе (квант/идеальный PS — новый режим `QsSim` или
  `SizeBasedQsSim`); LCFS-PR — режим стека с прерыванием.

### 6. M/G/1 FB/LAS

- **Теория:** E[T(x)] через ρ_x и второй момент min(S,x) (Nuyens–Wierman, Perform. Eval. 2008,
  doi:10.1016/j.peva.2007.06.028; Harchol-Balter 2013, гл. 28–33) — те же усечённые интегралы,
  что в `theory/srpt/utils/load_below.py`.
- **Реализация:** `theory/srpt/mg1_fb.py` — `MG1FbCalc` в семействе size-based
  (хотя FB — blind-дисциплина, машинерия общая); интеграция в сравнительные таблицы SRPT.
- **Валидация:** `SizeBasedQsSim` с ранжированием по attained service (ключ сортировки уже
  трекается для SRPT); сравнение FB vs SRPT vs FCFS vs PS.

### 7. Ненадёжный прибор M/G/1 (breakdowns & repairs)

- **Теория:** Avi-Itzhak–Naor (Oper. Res. 1963, doi:10.1287/opre.11.3.303): прерывания
  прибора → generalized service (completion time), далее обычный P-K; структурно эквивалентно
  vacation-декомпозиции.
- **Реализация:** `theory/negative/mg1_unreliable.py` или `theory/vacations/` (решить при
  реализации: концептуально ближе к vacations) — `MG1UnreliableCalc`; вход — моменты b,
  интенсивность отказов, моменты ремонта.
- **Валидация:** симуляция отказов через события в `QsSim` (новый механизм breakdown/repair —
  минимальное расширение) либо через эквивалентный vacation-сценарий.

## Общие требования

- Каждая модель: DoD «новая модель СМО» из [DOD.md](../DOD.md) — BaseQueue-контракт,
  первоисточник в docstring, проверка стабильности, тест теория-vs-симуляция,
  запись в `docs/models.md`.
- Аппроксимации (п. 3) явно маркируются в docstring и результатах.
- Порядок выбран по нарастанию работы в симуляторе: пп. 1–4 не требуют его изменений,
  пп. 5–7 добавляют режимы дисциплин/отказов.

## Оценка

~2–3 недели суммарно. Пп. 1–2 — дни; п. 3 — 2–3 дня с таблицами валидации; п. 4 — 2–3 дня;
пп. 5–7 — по 2–4 дня каждый (в основном симуляторная часть).
