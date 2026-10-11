# Roadmap исследовательских пилотов после серии Л

2026-10-11. Автор выбрал **ГЛ2 → ГЛ3**. Это отдельная исследовательская
ветка, не добавление отсутствующего Л9 в [серии Л](llm_serving_catchup_roadmap.md).

| Этап | Реализация и опыт | Научный вывод | Статус |
|---|---|---|---|
| [ГЛ2 / EPIC-092](../epics/EPIC-092-llm-mark-order.md) | Audited BurstGPT, frozen pair, measured-law-preserving controls, independent oracles | Rank reversal не подтверждён, effect0.708п.п. ниже5п.п. | done research pilot |
| [ГЛ3 / EPIC-093](../epics/EPIC-093-stage-aware-ttft.md) | Stage-aware PK/finite-loss calculator, independent loss/token DES,48 frozen cases | Mean accuracy повысилась; compactness и decision gain не установлены | done research pilot |
| [Продолжение ГЛ3 / EPIC-094](../epics/EPIC-094-stage-aware-ttft-reduction.md) | Exact iid integration, decoder symmetry, full/ablation/paired timing | Finite means сохранены; mixed13x меньше states,18–27x быстрее full; real-support scalability не установлена | done research pilot |
| [Infinite ГЛ3 / EPIC-095](../epics/EPIC-095-infinite-stage-aware-ttft.md) | Killed regenerative kernel, service-credit domination, overshoot/mean/tail bounds |45/48 bounded,3 sufficient-stability refusals; no verified rounding, exact threshold or novelty claim | done research pilot |
| [Фазовая устойчивость / EPIC-096](../epics/EPIC-096-phase-aware-stability.md) | Exact active reachability, multi-class corrector, rational drift, core/exterior continuation |48/48 bounds, все3 отказа сняты; critical unresolved, mean arithmetic floating | done research pilot |
| [Уточнение возвратов / EPIC-097](../epics/EPIC-097-core-aware-continuation.md) | Actual exit-phase moments, finite-core killed solve, full repeated excursions |48/48 original; near-critical28/32 вместо24/32,4refusals recovered; >=2xstates target failed1/32 | done research pilot |

[ГЛ2 полный отчёт](../research/llm-mark-order-validation-2026.md) ·
[ГЛ3 полный отчёт](../research/stage-aware-ttft-validation-2026.md).
Оба прогона сохраняют отрицательные выводы и не устанавливают новую научную
заявку. Полные source cards/данные/диссертационный синтез остаются в приватной
папке по принятому ранее режиму; commit/push её не резервируют.

## Возможные продолжения после отдельного выбора

Для ГЛ2 — новая preregistration training-selected load region/pair, другой
источник и hardware validation; не переобозначать старый negative test как tuning.
Для ГЛ3 автор выбрал следующий этап: exact iid reduction в EPIC-094.
Он снимает waiting-word explosion, но не комбинаторику active phases.
Далее автор выбрал infinite queue/error этап EPIC-095: выполнены analytic
truncation bounds при conservative sufficient condition (floating evaluation).
Автор затем явно выбрал EPIC-096: фазозависимый сертификат реализован и проверен
на frozen grid. [Доказательство и результаты](../research/phase-aware-stability-validation-2026.md).
Автор явно выбрал EPIC-097: phase-specific continuation и finite-core solve
на новой frozen near-critical сетке реализованы. [Результаты и отрицательный
engineering target](../research/core-aware-continuation-validation-2026.md).
Приближённая phase compression для real support, verified mean rounding,
SLO/hardware validation и сложный configuration trade-off требуют выбора.
ГЛ1 остаётся самостоятельным математическим кандидатом на сертификат усечения.
