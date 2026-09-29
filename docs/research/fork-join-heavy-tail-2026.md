# Fork-Join с тяжёлыми хвостами: обзор литературы и gap-анализ (2026-09)

- **Дата:** 2026-09-29
- **Источники:** OpenAlex, Crossref, arXiv (скилл `lit-search`: fork-join heavy-tailed/
  subexponential, extreme value order statistics, redundancy systems heavy tail) +
  инвентаризация кода
- **Эпик по итогам:** [EPIC-022](../epics/EPIC-022-fork-join-heavy-tail.md)

Продолжение серии пост-EPIC-020 обзоров ([../research/sla-deadline-queueing-2026.md](sla-deadline-queueing-2026.md)
§4 «резерв»). Направление подтверждено активным при обоих проходах lit-search (июль и сентябрь
2026): «Extreme values for the waiting time in large fork-join queues» (Queueing Systems 2025) и
«Fork–join and redundancy systems with heavy-tailed job sizes» (Queueing Systems 2022) — оба
продолжают набирать цитирования, плюс смежные работы про tail asymptotics в GI/GI/2 и
multi-server очередях с Weibull/subexponential обслуживанием.

## 1. Что есть в most_queue сейчас

- `ForkJoinMarkovianCalc` (`theory/fork_join/m_m_n.py`) — M/M/n (n,k) fork-join через
  интерполяционные формулы Varma и Nelson–Tantawi (только среднее, только экспоненциальное
  обслуживание).
- `SplitJoinCalc` (`theory/fork_join/split_join.py`) — произвольное распределение времени
  обслуживания подзадачи **через моменты**: `MaxDistribution(b, n, approximation)` подгоняет
  H2/Gamma/Erlang по переданным моментам и численно (квадратура Гаусса–Лагерра) считает моменты
  максимума n таких величин, дальше — точная M/G/1 (P-K) поверх этого максимума.
- `MaxDistribution` (`theory/utils/max_dist.py`) — ядро расчёта максимума; жёстко привязано к
  light-tailed аппроксимациям (H2/Gamma/Erlang), других семейств нет.
- Симулятор `ForkJoinSim` (`sim/fork_join.py`, наследует `QsSim`) — принимает любое распределение
  через `set_servers(params, kendall_notation)`, включая `"Pa"` (Pareto) — **DES-кросс-валидация
  для тяжёлых хвостов уже технически возможна**, просто аналитики нет.

**Ключевой пробел:** весь fork-join-стек библиотеки математически предполагает конечную дисперсию
(и обычно light-tailed/exponential-family) время обслуживания подзадачи. Тяжёлые хвосты (Pareto,
Weibull с shape<1) — качественно другой режим: максимум/сумма n таких величин асимптотически
определяется одним «большим скачком», а не концентрируется вокруг среднего.

## 2. Литература

- Boxma O., Mathijsen B., Nazarathy Y. и др., *Fork–join and redundancy systems with heavy-tailed
  job sizes*, Queueing Systems, 2022, doi:10.1007/s11134-022-09856-6 — асимптотика времени
  отклика fork-join/redundancy систем при subexponential размере задач («one big jump»: хвост
  времени ответа ~ хвост самой медленной подзадачи).
- *Extreme values for the waiting time in large fork-join queues*, Queueing Systems, 2025,
  doi:10.1007/s11134-025-09937-2 — предельное распределение экстремумов времени ожидания при
  n→∞ подзадач (extreme-value theory поверх fork-join).
- *Tail asymptotics for delay in a half-loaded GI/GI/2 queue with heavy-tailed job sizes*,
  Queueing Systems, 2015, doi:10.1007/s11134-015-9451-0.
- *Queue length asymptotics for the multiple-server queue with heavy-tailed Weibull service
  times*, Queueing Systems, 2019, doi:10.1007/s11134-019-09640-z.
- *Tail asymptotics for the delay in a Brownian fork-join queue*, Stochastic Processes and their
  Applications, 2023, doi:10.1016/j.spa.2023.06.013.
- *Correlation in redundancy systems*, Queueing Systems, 2022, doi:10.1007/s11134-022-09829-9 —
  смежная тема (redundancy = запуск нескольких копий одной задачи, забирается первый результат;
  математически похоже на fork-join с k=1).
- *Fork–join and redundancy systems with heavy-tailed job sizes*, arXiv-препринт той же группы —
  основной первоисточник для «one big jump» асимптотики очереди (не только максимума).

## 3. Что реально реализуемо (без overreach в предельные теоремы)

Максимум n **независимых одинаково Pareto(α, K)**-распределённых величин имеет точную замкнутую
форму — не нужна ни аппроксимация, ни асимптотика:

- CDF: `F_max(x) = (1 - (K/x)^α)^n`, `x ≥ K` — точно, для любого n и α.
- Сырые моменты: заменой `U = F(X)` (тогда `X = K·(1-U)^{-1/α}`, а максимум n Pareto соответствует
  максимуму n Uniform(0,1), плотность `n·u^{n-1}`) получаем
  **`E[max^k] = K^k · n · B(n, 1 - k/α)`** (Beta-функция), корректно при `k < α` (как и для самого
  Pareto — моменты порядка ≥ α расходятm независимо от n).

Это даёт **точный** (не численно-квадратурный) калькулятор максимума для Pareto — сильнее, чем
существующий light-tailed путь через `MaxDistribution`, и ложится точно в нишу «точные моменты».
Дальше эта точная замкнутая форма стыкуется с уже готовым `SplitJoinCalc` (макс → M/G/1 P-K) **при
α > 2** (когда дисперсия конечна и P-K применим); при `α ≤ 2` M/G/1 P-K неприменим в принципе
(нужна дисперсия), но точная CDF/хвост максимума остаётся валидной сама по себе — это уже
самостоятельно полезный результат (P(время обслуживания худшей подзадачи > x)) и якорь для
DES-кросс-валидации.

**Асимптотика «one big jump» для полного времени отклика очереди** (не просто максимума
подзадач, а с учётом M/G/1-очереди поверх него) при `α ≤ 2` — genuine research-level результат
Boxma et al. 2022, требует аккуратного воспроизведения формул из первоисточника. Оставлен в
резерве (см. gap-анализ) — не наспех выводится из вторичных источников, чтобы не внести тонкую
ошибку в «точный» калькулятор.

## 4. Gap-анализ и решение

| Направление | Статус | Реализуемо точно? |
|---|---|---|
| Точный max n·Pareto (CDF + моменты k<α, Beta-функция) | нет в коде | да, точно — берём в EPIC-022 |
| `SplitJoinCalc` с `approximation="pareto"` (при α>2 → точный M/G/1 поверх max) | нет | да, композиция уже готовых точных кусков |
| DES-кросс-валидация (`ForkJoinSim` уже поддерживает `"Pa"`) | инфраструктура есть | да |
| «One big jump» асимптотика полного времени отклика очереди при α≤2 | нет | резерв — нужен аккуратный разбор Boxma et al. 2022 |
| (n,k)-fork-join order statistics для Pareto (k-я порядковая статистика, не максимум) | нет | резерв — тот же Beta-подход обобщается на `E[X_(k)^m]`, но не первый приоритет |
| Extreme-value предельные распределения (QS 2025) | нет | резерв — асимптотическая теория, не точный калькулятор |

**Решение:** реализовать точный max-n-Pareto слой (CDF + моменты) и подключить его к `SplitJoinCalc`
и DES-кросс-валидации — см. [EPIC-022](../epics/EPIC-022-fork-join-heavy-tail.md). «One big jump»
асимптотика очереди и (n,k)-Pareto порядковые статистики — в резерве на будущее.
