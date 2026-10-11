# EPIC-092: ГЛ2 — порядок совместных длин и выбор конфигурации

- **Статус:** done (research pilot; hypothesis not confirmed)
- **Создан:** 2026-10-11
- **Основание:** автор выбрал ГЛ2 → ГЛ3; продолжение Л, не Л9.
- **Roadmap:** [исследовательские пилоты](../roadmaps/llm_research_pilots_roadmap.md).

## Зафиксированный протокол (до replay)

BurstGPT v2.0 `BurstGPT_1.csv`, SHA-256
`4bb3783693d0a435686fbfc885615d2349bd067239079fa4b749f2e679e12122`.
Один model=ChatGPT, API log, только положительные пары; весь CSV аудируется,
отказы не подменяются короткими успешными запросами. Дни <30 — история;
первичные test-дни 32,36,40,44,48,52,56,60. В каждом первые 1152 подходящих
запроса; первые128 — прогрев, исключённый из latency endpoint. Это ограниченный
prefix-of-day cohort одного источника, не репрезентативность всей продукции.

Две конфигурации `(B,M)=(4,256),(16,2048)`, costs
`(alpha,beta,gamma)=(0.00668,0.0000201,0.0000000552)` из прежнего Л3 примера,
TTFT SLO=1 секунда; исходные timestamps без масштабирования. Пара/порог/дни
не выбираются по replay test. Policy — существующий causally-blind
`PrefillDecodeSim`: boundary FCFS admission, decode-first, FCFS prefill budget.
Это length-only модель, не vLLM, не измерение GPU; KV capacity/cache отсутствуют.

Главная абляция — global shuffle whole pairs. Local clock-window300s,
block32 shuffle и circular shift — диагностические контролы, не дополнительные
подтверждающие тесты. Три seeds 92000..92002. Все варианты сохраняют multiset
целых пар, E[SO], timestamps и число запросов. Прогрев закреплён: перестановка
затрагивает только measured suffix, joint law сохраняется отдельно в нём.
Поправка протокола после software audit: первый недействующий run пересекал
границу warmup, он сохранён приватно; параметры/endpoint не изменены.
Global меняет также alignment
marks/intensity; local сохраняет joint multiset каждого окна.

Endpoint: разность долей TTFT<=1 для большой и малой конфигурации,
равновесное усреднение по дням, paired day bootstrap95% (10000 draws, seed92999).
Не считать requests/permutation seeds независимыми трассами. CI условен к
предположению переносимости блоков, не является доказательством независимости
дней. Подтверждение rank reversal требует противоположных знаков обоих CI
и сдвига>=5 п.п.; иначе кандидат не подтверждён в этой области.
Все first tokens вычисляются drain-ом, отдельно backlog at final arrival,
busy exposure и offered serialized-work diagnostics; drain не доказывает rho<1.

## Задачи

- [x] Аудит исходных данных и контроль инвариантов.
- [x] Независимый oracle (B=1 Lindley и ручной boundary trace).
- [x] Frozen replay, paired uncertainty, отрицательные результаты.
- [x] Документация; архивирование выполняется совместно с ГЛ3.

## Результаты

[Отчёт](../research/llm-mark-order-validation-2026.md): rank reversal не
подтверждён; corrected primary shift0.7080п.п., условный CI[−0.0244,2.0793], ниже5п.п.
Пилот выполнен без изменения pair/costs/threshold по test. Расширение на
второй источник и hardware — новый этап, не выполненный результат этого эпика.

## Итоговая проверка

Совместная проверка ГЛ2/ГЛ3: fast3236passed, full3256passed (16 прежних warnings),
37 новых targeted tests; black/isort чистые, pylint новых library/example
modules10/10, gitdiffcheck чистый. Source data/PDF/cards и negative result
сохранены в private dissertation folder; README/models/simulation обновлены.
