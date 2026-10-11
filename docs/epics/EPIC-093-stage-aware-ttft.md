# EPIC-093: ГЛ3 — стадийный TTFT и пределы точности

- **Статус:** done (research pilot; compact predictor/publication not established)
- **Создан:** 2026-10-11
- **Предшественник:** [ГЛ2](EPIC-092-llm-mark-order.md).
- **Roadmap:** [исследовательские пилоты](../roadmaps/llm_research_pilots_roadmap.md).

## Протокол до расчёта test-сетки

Сначала finite-support joint `(S,O)` и одна политика Л3 token DES: admission
только между итерациями, decode-first, затем FCFS prefill. Во внутреннем
semi-Markov состоянии сохраняются порядок, тип и достигнутое число токенов;
итоговая длина латентна для планировщика. Holding time детерминирован условно
на состояние; arrivals внутри holding учитываются непрерывно. TTFT считается
через reward «число запросов без первого токена», включая остаток текущей
итерации для каждого нового прибытия. Normalization — accepted arrival flow,
не unconditional active occupancy и не iteration-epoch distribution.

B=1: точный M/G/1 infinite FCFS, полное service для PK и только first phase
для TTFT. B>1: конечная очередь K; exact finite-loss model, диагностическая
аппроксимация infinite DES. K=2,4,6; loss и sensitivity публикуются, не
сертификат ошибки. Ограничения states/transition words с явным отказом.

Frozen grid: B=1,2; M=2,4; joint laws `[(2,2,1)]` и
`[(1,1,.6),(3,2,.4)]`; costs `(alpha,beta,gamma)=(.1,.02,.003)`
и heldout `(.07,.03,.005)`; offered load lambda*E[isolated full service]=.1,.5,.8.
Второй cost set не подгоняется после test. По 3 независимых DES seeds
93000..93002, 12000 запросов, 20% warmup. Изолированный предел отдельно.

Baselines: literal Л3; engineering clamp profile(max(1,Eactive)) с прежним
waiting; limited-PS Zhang2009 (6.28–6.30), применённый к isolated full/first
work с предположением постоянной aggregate capacity. Последний не exact
эквивалент continuous batching; различие дисциплин сохраняется в отчёте.

Primary: median absolute error mean TTFT на frozen grid, снижение>=25%
относительно literal Л3 (также отдельное B>1); времена расчёта и число states.
Случаи недопустимого baseline не выбрасываются молча. DES mean uncertainty
по независимым seeds; rank mean-TTFT решений B,M — diagnostic, не SLO.
GPU-tail validation не является результатом вычисления этих средних.

## Задачи

- [x] Релевантные разделы Zhang/Bekker и современные соседи.
- [x] Вывод reward, PK, residual arrival time; реализация и пределы.
- [x] Независимая finite-loss симуляция / oracle, infinite token DES.
- [x] Frozen evaluation и оценка вычислительной пригодности.
- [x] Документация; source cards/журнал/архив синхронизируются в итоговой проверке.

## Результаты

[Отчёт и все48 случаев](../research/stage-aware-ttft-validation-2026.md): median
MAE reduction94.51%, отдельно B>1 93.32%, heldout costs94.51%. Порог25% пройден
описательно относительно DES estimates, не proven infinite error bound.
Независимые PK/isolated/finite-loss checks проходят. K6 loss<=.001813,
K4→K6 TTFT sensitivity до4.51%, не сертификат. 42–1907states уже на коротких
длинах; универсальная компактность не достигнута. Configuration diagnostic
не улучшен: stage и literal верно выбирают все24 pairs. Общий semi-Markov
reward ratio известен, новизна не заявлена.

Scalable phase compression, tail/SLO analysis и hardware validation вынесены
в потенциальное продолжение, не считаются выполненными и не начаты автоматически.

## Итоговая проверка

Fast3236passed, full3256passed (16 прежних warnings),37 targeted tests;
black/isort/gitdiffcheck чистые, pylint новых library/example modules10/10.
Same-clock four-case timing: homogeneous solver/DES wall≈.067–.071,
mixed≈1.56–1.83; подтверждает отсутствие universal speed/compactness claim.
Source cards/гипотезы/план/концепция/журнал синхронизированы, прежний journal
prefix сохранён; public reports/JSON/roadmap архивированы с SHA-256.
