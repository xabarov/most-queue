# Л6: SMDP запуска пакета — воспроизведение и независимые проверки

Дата: 2026-10-11. [EPIC-089](../epics/EPIC-089-smdp-batching.md),
[контракт/API](../models/smdp-batching.ru.md),
[эксперимент](../../examples/smdp_batching_experiment.py),
[численные записи](smdp-batching-results-2026.json).

## 1. Что прочитано и что реализовано

Yaodan Xu, Jingzhou Sun, Sheng Zhou, Zhisheng Niu,
*SMDP-Based Dynamic Batching for Efficient Inference on GPU-Based Platforms*,
ICC 2023, [DOI](https://doi.org/10.1109/ICC45041.2023.10278962).
Метаданные названия/авторов/DOI сверены через Crossref; реализована открытая
[версия arXiv:2301.12865v3](https://arxiv.org/html/2301.12865v3), 2023-09-01,
а не неподтверждённая эквивалентность final proceedings и препринта.
Прочитаны §§I–VI и bibliography целиком. Полный HTML и formula-preserving
текст сохранены в приватной литературе диссертации. Код авторов не проверялся.

Это литературная репликация: конечный SMDP kernel/rewards, uniformization,
RVI и диагностики плюс отдельный unbounded DES. Произвольные конечные
профили вместо только affine и independent LP — библиотечные расширения,
не заявка на новый метод. Запуск нового исследовательского эпика не объявляется.

### Предшественники и четыре направления

Прицельные запросы: average-cost semi-Markov batch control; batch service
size-dependent time; SLA-aware batching; AoI SMDP power; inference batching
energy. Глубина чтения не повышается по факту наличия ссылки.

| Направление | Проверенный первичный материал | Значение и границы |
|---|---|---|
| Классическая ТМО | [Deb–Serfozo 1973](https://doi.org/10.2307/1426040), [Sennott 1989](https://doi.org/10.1017/S0269964800001121): аннотации издателя | Управление batch service/average-cost SMDP давно известно; полные proofs не прочитаны. Нельзя из введения Xu заключать, что все классические работы исключали управляемое время обслуживания |
| Монография | [Puterman, Wiley](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887): поиск, страница не открылась | Оглавление/главы не прочитаны; не выдаём это за независимый аудит average-cost теорем. Локальный Рыжиков, CH4w.TEX, начало Markov-contract перечитано, не заменяет SMDP монографию |
| Real-time / scheduling | [Choi–Kim–Rhu, Lazy Batching, HPCA 2021](https://pure.kaist.ac.kr/en/publications/lazy-batching-an-sla-aware-batching-system-for-cloud-machine-lear/): авторская карточка/аннотация | SLA-aware node-level batching; полные экспериментальные условия не проверены. Не наш mean-delay/energy SMDP |
| AoI | [Zhou–Zhang–Yang–Yates, 2505.14307](https://arxiv.org/abs/2505.14307): метаданные/абстракт | SMDP для age/power CPU scheduling, не batch response criterion; полного proof audit нет |
| ML systems | [Clipper, NSDI 2017](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw): авторский abstract; Inoue — полный источник ранее в Л1 | Serving batching как предшественник, но этот поиск не доказывает отсутствия других оптимизационных моделей |

Поиск достаточен для атрибутированной реализации согласованного Л6, не для
утверждения о свободной нише или новизне нового сертификата усечения.

## 2. Физическое время и конечный контракт

Контракт и формулы kernel/reward полностью записаны в
[модели](../models/smdp-batching.ru.md). Главные проверяемые различия:

- cost per time, не per decision;
- overflow-метка N+1 использует effective population N;
- хвостовое Δ — вклад стоимости, не доказанная ошибка усечения;
- embedded μ и μ*y/E[y] не равны реальному Q(t)-PMF;
- original SMDP bias = η × transformed MDP bias;
- finite optimality не заменяет проверку реального хвостового действия.

Компрессия оператора использует корреляцию центрированного bias с Poisson PMF;
нет хранения dense state/action tensor. Финальная fixed-policy stationary
оценка, напротив, плотная. Численные пределы N≤2000, B≤128 явные.
Ненормированные/нефинитные reward/transition arithmetic, singular stationary
система и исчерпание итераций не скрываются. Отрицательные μ только в пределах
roundoff обрезаются с повторной balance-проверкой; сверхмалый хвост может
оказаться нулём в double precision — это не доказательство отсутствия overflow.

## 3. Независимые арбитры

1. **Occupation LP.** На малой модели и отдельно source N=70 строятся переходы
   непосредственно по Poisson/source formulas, не через production `epoch`.
   Переменные x — decision frequency на физическое время:
   sum x(P−I)=0, sum x*y=1, minimize sum x*c. Проверяются flow/time residuals.
2. **Полный перебор 18 политик при N=2, B=2.** Отдельное least-squares
   стационарное решение, не production `solve`; gain совпадает с RVI и LP.
3. **Прямые Poisson-суммы и квадратура площади.** Обслуживаемые запросы
   остаются до конца пакета; independent integral проверяет service/idle costs.
4. **Dense uniformized operator.** По независимо собранному tensor проверены
   compressed correlation, η self-loop и original-time Bellman residual/bias.
5. **Ручной replay.** arrivals [0,.1,.2,1.2], times [1,1.5], energies [1,3],
   horizon3: area7.5, completed exposure5.7, unfinished1.8; arrived4,
   completed3, unfinished1, energy5. Warmup.5: observed area6.3, power4/2.5.
6. **M/D/1 угол.** B=1, λ=1, τ=.4, без energy weight:
   E[V]=.4+.16/(2*.6)=.5333333333; finite N=30 и шесть unbounded репликаций.
7. **Accounting/семантика.** Actual queue выше N не обрезается; seed reset,
   infeasible actions, event limits, replay ties, partial batches/no drain,
   reconfiguration и nonconvergence проверены отдельно.

Source LP при N70,c_o100, λ=.9*(32/10.8152): gain66.1341131829;
RVI policy evaluation66.1341127232 (разница4.60e−7);
LP flow residual1.55e−10, time normalization3.33e−16.
RVI tol1e−7, iterations3578, gain bracket
[66.1341127228,66.1341128218], original Bellman residual9.75e−7.
Численный LP optimum немного вне этого узкого bracket: LP имеет собственную
feasibility/optimality tolerance. Это независимое согласие на уровне1e−5,
не exact-arithmetic certificate и не обещание бесконечной optimality.

## 4. Опубликованный профиль и Table I

Численные входы: GoogLeNet/TESLA P4, τ_b=.3051b+1.052 **ms**,
ζ_b=19.90b+19.60 **mJ**, B32, веса1/1. Мощность mJ/ms — **W**.
Точная capacity из этих коэффициентов2.9587987277 requests/ms,
в статье округлена до2.96. Записаны обе λ: .9*exact и .9*2.96.
Это арифметическая репликация опубликованного профиля, не GPU calibration.

| N / c_o | Напечатанный gain | Наш gain: exact capacity | Наш Δ: exact capacity | Overflow action / real drift |
|---|---|---|---|---|
| 89 / 10000 | 66.1384 | 66.13477056 | 9.34545e−4 | 6 / +1.67613 |
| 78 / 1000 | 66.1383 | 66.13471086 | 9.76333e−4 | 6 / +1.67613 |
| 70 / 100 | 66.1377 | 66.13411272 | 8.34995e−4 | 6 / +1.67613 |
| 161 / 10 | 66.1374 | 66.13384839 | 6.03416e−12 | 32 / −3.2 |
| 192 / 0 | 66.1374 | 66.13384839 | numerical 0 | 32 / −3.2 |

Tol=.01, η=.9min(y), **max_iterations20000**. Два последних режима сошлись
за11044/11667 итераций: не воспроизводим остановку статьи на10000 как
converged. Остальные2032/1800/1633, не printed1847/1635/1483. В source table
η не зафиксирован; iteration savings автора не выдаются за наши измерения.

Расхождение gain с печатью около.00355–.00363 (≈.0054%). При rounded capacity
gain66.16710–66.16808: простое округление не устраняет расхождение.
Independent LP подтверждает нашу **заданную** finite model; авторский код,
неокруглённый GPU fit и точный numerical setup не доступны для аудита.
Причина несовпадения печатных gain не установлена; ошибку статьи не заявляем.
Для rounded case N78 Δ=.0010257 уже превышает printedδ=.001.

### Чувствительность усечения/penalty

16 режимов: N[32,50,70,100] × c_o[0,10,100,1000], tol1e−5.
Все конечные RVI сошлись. При c_o0/10 все эти N выбирают вечное ожидание
в overflow; например N70 gains26.28694/36.28694. При c_o100 N70 finite
gain66.13411 и Δ8.35e−4, N100 gain66.13384883 и Δ1.78e−6, но tail action6
в обоих даёт положительный реальный дрейф. Малый Δ не решает перенос политики.
Source уже обсуждает патологию малой penalty/«always wait»; она не новая.

## 5. Unbounded DES и стресс хвоста

Горизонт20000, warmup4000, seeds9100–9103. Профильные четыре сравнения,
два tail stress, девять synthetic(rate[.2,1,2.5]×energy weight[0,4,20])
с четырьмя политиками: всего42 настройки ×4 = **168** прогонов.
Добавлены три фиксированных горизонта always-wait контрпримера.
Общие потоки парные: политика не потребляет arrival RNG. Во всех запусках
сохраняются запросы и ∫Qdt=completed exposure+unfinished exposure.

Source N70 finite tail a6 имеет service-epoch drift+1.67613: эта extension
неустойчива. Явная замена только a выше N на32 даёт drift−3.2, то есть
стандартный Foster sufficient argument для этой конкретной extension.
Не делаем из него SLO, finite-gain equality или infinite optimality.

| Политика source | Пустой старт: observed cost mean ± SE | Initial1000: cost mean ± SE | Initial1000: unfinished по 4 seeds |
|---|---|---|---|
| Исходный overflow a6 | 65.87460 ± .18618 | 3015.26361 ± 17.13967 | 12522,12315,12452,12555 |
| Отдельная tail repair a32 | 65.87460 ± .18618 | 65.87343 ± .18641 | 39,32,25,22 |
| Work-conserving min(s,32) | 65.91460 ± .18449 | не запускался | — |
| Fixed batch32 | 71.83034 ± .15397 | не запускался | — |

Пустые траектории исходной и исправленной политики побайтно совпадают:
порог N не посещён на этих seeds. Это **не** доказательство её устойчивости.
Разница finite gain и sampled cost при пустом старте около1.4SE; четыре
репликации дают noisy estimates, не публикационный confidence certificate.
Положительный дрейф вместе с явно заданным constant-tail kernel объясняет
стресс, а не выводится из конечного горизонта наблюдений.

Synthetic λ1,w2=4: finite3.73315567; DES finite/repaired/fixed2
3.73778734±.01290312 против work-conserving4.33382673±.01755847.
При λ2.5 все finite policies имеют неустойчивую хвостовую extension,
хотя обычные DES и repair также совпали: опять опасность редкого выхода за N.
Записи содержат полные действия, статус, drift и accounting, не только means.

Always-wait counter: N2,λ.1,τ=[1,1],ζ=[1,1],w1=1e−8,w2=100,c_o0.
Finite gain=Δ=2e−7 и Bellman convergence, но реальный tail drift+1.
При horizons1000/10000/100000 unfinished106/980/9978,
response proxies576.32/4928.55/49717.84. Любой положительный holding weight
порождает растущий реальный cost; это наглядная проверка уже известной
патологии, не новая теорема. Нет completed-only цензурированной latency оценки.

## 6. Воспроизводимость и статус проверок

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m examples.smdp_batching_experiment
pytest tests/units/test_smdp_batching.py tests/test_smdp_batching_vs_sim.py -q
```

JSON исключает duration/bias/trace, сохраняет profiles/weights/rates/N/penalty,
η/tolerance/iteration limit, horizons/warmup/seeds и исходные counters.
Два запуска дали побайтно одинаковый JSON, SHA256
`6d46881822d72c053e8ee511568180578fcd6079365ae905812b1da84eec45cf`.
66 targeted passed, отдельный slow1passed; full3141passed/439.78s,
fast3130passed/334.25s. 16 прежних warnings, без reruns. Production/experiment
pylint10/10, package9.99/10 с прежними сообщениями; black/isort/pre-commit,
исполняемые EN/RU snippets, local links и две схемы проверены.
92 диссертационных снимка с SHA256 и семь тестов синхронизатора;
источники/леммы/план/концепция/гипотезы/обзор/append-only журнал обновлены.
Финальная запись — в [EPIC-089](../epics/EPIC-089-smdp-batching.md).

## 7. Что это меняет для диссертации

Управление batch launch/energy-delay не является новым механизмом.
Сертифицированная ошибка усечения и контроль стабильного продолжения — разные
задачи. ГЛ1 остаётся кандидатом, но drift/truncation SMDP имеет классическую
литературу; без полных соседей новизна не установлена. Это дополнение к карте
рисков, не изменение формального утверждения ГЛ1 и не запуск её исследования.
Слой С получает компонент управления, не интегрированный синтез реплик,
ускорителей, KV и hardSLO. Следующий согласованный пункт — Л7, с отдельной
проверкой доступа к закрытому первоисточнику перед реализацией.
