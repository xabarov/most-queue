# Roadmap: retrial-очереди и Erlang-A

> Эпик: [EPIC-004](../epics/EPIC-004-retrial-erlang-a.md). Основание:
> [gap-анализ](../models_gap_analysis.md) — оба направления **should**; объединены в один
> roadmap, т.к. описывают родственное поведение абонента (повторная попытка vs уход)
> и оба востребованы в телеком/колл-центрах.

## Первоисточники

**Retrial:**
- Falin G.I., Templeton J.G.C. *Retrial Queues*. Chapman & Hall, 1997 — базовые решения
  M/M/1 и M/G/1 retrial (PGF/LST).
- Artalejo J.R. *Accessible bibliography on retrial queues*. Math. Comput. Model., 1999,
  doi:10.1016/s0895-7177(99)00128-4; продолжение 2000–2009: doi:10.1016/j.mcm.2009.12.011.
- Artalejo, Falin. *Standard and retrial queueing systems: a comparative analysis*.
  Rev. Mat. Complutense, 2002, doi:10.5209/rev_rema.2002.v15.n1.16950.

**Erlang-A:**
- Garnett O., Mandelbaum A., Reiman M. *Designing a Call Center with Impatient Customers*.
  M&SOM 4(3), 2002, doi:10.1287/msom.4.3.208.7753.
- Mandelbaum A., Zeltyn S. *The Palm/Erlang-A Queue...* Springer, 2007,
  doi:10.1007/978-3-540-29860-1_2.

## Часть A. Erlang-A (M/M/n+M) — начать с неё, это проще

- [ ] `theory/impatience/mmn.py` — `MMnImpatienceCalc` (обобщение `MM1Impatience`):
      birth-death с интенсивностями ухода (k-n)θ выше уровня n; стационарные вероятности
      через устойчивые рекурсии (incomplete gamma, scipy); P(wait), P(abandon), моменты W
      (условные на «дождался» и безусловные), E[N].
- [ ] Обратные задачи staffing: мин. n при целевых P(abandon)/P(wait) — прикладной сценарий.
- [ ] Валидация: `ImpatientQueueSim` (уже поддерживает нетерпение); частные случаи —
      θ→0 = M/M/n, n=1 = `MM1Impatience`.
- [ ] Опционально: QED-аппроксимации Garnett–Mandelbaum–Reiman как справочные формулы.

## Часть B. Retrial M/M/1 и M/G/1

- [ ] `theory/retrial/mm1.py` — `MM1RetrialCalc`: классическая линейная retrial-политика,
      точное стационарное распределение (ряды гипергеометрического типа); характеристики
      орбиты (среднее число, моменты), блокировка, w/v.
- [ ] `theory/retrial/mg1.py` — `MG1RetrialCalc`: PGF/LST-решение Falin–Templeton (гл. 1),
      моменты дифференцированием — стандартный для библиотеки моментный стиль;
      стохастическая декомпозиция относительно M/G/1 (использовать как внутренний кросс-чек).
- [ ] Симуляция: механизм орбиты в `QsSim` — заблокированная заявка уходит в орбиту и
      возвращается через Exp(γ)-задержку; новый класс `RetrialQsSim` в `sim/` по образцу
      `impatient.py`.
- [ ] Валидация: теория vs симуляция на сетке (ρ, γ); предельный случай γ→∞ = обычная
      M/G/1 (декомпозиция вырождается) — тестовая проверка.
- [ ] Многоканальный retrial (M/M/c retrial, c>2): точных формул нет — НЕ реализуем здесь;
      после QBD-ядра ([EPIC-003](../epics/EPIC-003-qbd-map-ph.md)) — через
      level-dependent QBD/усечение (Neuts–Rao), отдельной задачей.

## Синергия с EPIC-003

Retrial + MAP (коррелированный вход + орбита) — активная исследовательская тема
(Vishnevsky и др., 2025). После закрытия обоих эпиков открывается MAP/PH/1 retrial —
кандидат в научную публикацию на базе библиотеки.

## Оценка

Часть A — ~1 неделя (расчёт дни + staffing + тесты). Часть B — ~2 недели
(теория быстрее, основное — механизм орбиты в симуляторе и таблицы валидации).
Общий DoD — [DOD.md](../DOD.md).
