# Roadmap: QBD/MAP/PH-стек (матрично-аналитические методы)

> Эпик: [EPIC-003](../epics/EPIC-003-qbd-map-ph.md). Флагманское направление по
> [gap-анализу](../models_gap_analysis.md): наивысшая научная ценность (коррелированный
> трафик), явная ниша (BuTools заморожен с 2016 г.), прямой фит в научный фокус проекта
> (retrial+MAP, см. `works/queueing_systems_review/general.md`).

## Первоисточники

- Neuts M.F. *Matrix-Geometric Solutions in Stochastic Models*. Johns Hopkins UP, 1981.
- Latouche G., Ramaswami V. *Introduction to Matrix Analytic Methods in Stochastic Modeling*.
  SIAM, 1999. doi:10.1137/1.9780898719734 — основной алгоритмический источник.
- Latouche G., Ramaswami V. *A logarithmic reduction algorithm for quasi-birth-death processes*.
  J. Appl. Probab., 1993. doi:10.2307/3214773.
- Bini D., Meini B. *Improved cyclic reduction...* SIAM J. Matrix Anal. Appl., 2002,
  doi:10.1137/s0895479800371955 (альтернатива log-reduction).
- Fischer W., Meier-Hellstern K. *The MMPP cookbook*. Perform. Eval., 1993.
  doi:10.1016/0166-5316(93)90035-s.
- Lucantoni D. *New results on the single server queue with a BMAP*. Stoch. Models, 1991.
  doi:10.1080/15326349108807174 (фаза 2).
- Референс-реализации для сверки: BuTools 2 (BME, python-порт), SMCSolver (Van Houdt).

## Фазы

### Фаза 0. Инфраструктура PH/MAP в `most_queue.random`

- [ ] Класс `PHDistribution` (α, T): моменты, CDF/PDF, генерация сэмплов (для симулятора),
      частные случаи — готовые Exp/Erlang/H2/Cox конвертируются в PH-представление.
- [ ] Класс `MAP` (D0, D1): интенсивность, моменты интервалов, лаг-корреляции; частный случай
      MMPP; генерация сэмплов. Fitting по моментам+лаг-1 корреляции (KPC-подход — фаза 2).
- [ ] Юнит-тесты: сверка моментов/корреляций с аналитикой и с BuTools на эталонных параметрах.

### Фаза 1. QBD-ядро

- [ ] `theory/matrix/qbd.py` — решатель QBD (A0, A1, A2): R- и G-матрицы через
      logarithmic reduction (+ простая итерация как fallback/проверка), граничные уровни,
      π_k = π_0 R^k, агрегаты. Численные проверки: спектральный радиус R < 1, невязка.
- [ ] Юнит-тесты на аналитически известных случаях: M/M/1 (скалярный QBD), M/Er/1,
      сверка с MMnrCalc и MGnCalc (H2 — частный случай PH).

### Фаза 2. Калькуляторы очередей на QBD-ядре

- [ ] `M/PH/1` → `MPh1Calc`: p, w, v (waiting time — phase-type, моменты в замкнутой форме).
- [ ] `PH/PH/1` → `PhPh1Calc` (обобщает H2/H2/1; кросс-валидация с `HkHkNCalc` при n=1).
- [ ] `MAP/PH/1` → `MapPh1Calc` — целевая модель направления (коррелированный вход).
- [ ] `MAP/M/n` (QBD по уровням занятости) — если объём позволит; иначе — следующий эпик.
- [ ] Симуляция: `QsSim` учится принимать PH/MAP-источники из фазы 0 (генерация уже есть —
      подключение через `set_sources`/`set_servers`).

### Фаза 3 (отдельный эпик по итогам). BMAP/G/1 и MAP-fitting

Lucantoni-машинерия (M/G/1-тип, G-матрица через uniformization), фиттинг MAP по трейсам.
Не начинать до закрытия фаз 0–2.

## Валидация и DoD

- Каждый калькулятор: сверка с симуляцией (PH/MAP-генераторы из фазы 0) и, где возможно,
  с BuTools-результатами на опубликованных примерах (зафиксировать эталоны в тестах).
- Демонстрационный ноутбук: влияние корреляции входа (лаг-1 MAP) на w/v при равных
  mean/cv — ключевой сюжет «зачем MAP, если есть M и GI».
- Общий DoD — [DOD.md](../DOD.md).

## Оценка

Фаза 0 — ~1 неделя; фаза 1 — ~1 неделя; фаза 2 — 2–3 недели. Итого 4–5 недель до MAP/PH/1
включительно. Риски: численная устойчивость при плохо обусловленных D0 (жёсткие MMPP) —
закладывать масштабирование/балансировку в QBD-ядро с самого начала.
