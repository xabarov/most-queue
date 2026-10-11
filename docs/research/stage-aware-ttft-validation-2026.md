# ГЛ3: стадийный mean TTFT, вывод и frozen validation

**Статус: research pilot выполнен; точность поддержана, компактность и новый
публикационный вклад не установлены.**
[Протокол](../epics/EPIC-093-stage-aware-ttft.md),
[API и область точности](../models/stage-aware-ttft.md),
[полные48 случаев](stage-aware-ttft-results-2026.json).

## 1. Модель и вывод

Poisson rate λ, iid finite-support whole pairs `(S,O)` с вероятностями p_k,
независимые от arrivals. Фиксированная boundary FCFS/decode-first дисциплина
Л3 DES. Никакой output oracle в policy нет. Состояние начала итерации —
упорядоченные active `(k,q)` с attained q и waiting FIFO type sequence.
Его holding d определяется **нынешними** действиями: alpha overhead,
beta*amount compute, gamma*(q+amount) KV access для работающих запросов.
Ожидание новых arrivals непрерывное, несмотря на embedded boundary chain.

При конечной waiting capacity K и h waiting jobs, r=K−h мест. Принято
min(Poisson(λd),r) поступлений; FIFO class words имеют iid product probabilities.
После итерации уходят завершившиеся, затем waiting заполняет active vacancies.
Idle state holding имеет mean1/λ; его first arrival принят и немедленно запускается.
Из любого состояния при отсутствии arrivals за конечное число итераций система
пуста; значит конечная достижимая цепь имеет один recurrent class с idle.
Ожидаемые visits за regenerative cycle вычисляются через `(I−Qᵀ)x=b`;
stochastic/balance residual проверяется, а не предполагается по solver success.

Пусть U — число старых запросов, не выдавших first token (active q<=S и queue).
Они вносят U*d. Принятые за iteration новые запросы вносят

```text
A(r,d) = ∫₀ᵈ E[min(Poisson(λt),r)] dt
       = Σ[j=1..r] {d G_j(λd) − (j/λ) G_(j+1)(λd)},
G_j(x) = P(Poisson(x)>=j) = regularized lower gamma(j,x).
```

Это **остатки d−arrival_offset**, а не d на каждое новое прибытие.
Reward для TTFT = U*d+A; для waiting = h*d+A;
для response = (active+h)*d+A. При embedded stationary π и expected
accepted counts a_i получаем accepted-arrival means

```text
E[TTFT|accepted] = Σ π_i reward_TTFT(i) / Σ π_i a_i,
accepted_rate = Σ π_i a_i / Σ π_i holding(i).
```

То же Little/renewal reward для W,V. Это classical method, не новая Palm
теорема. Не отождествляем π со временем или arrival occupancy; потому `p/pi`
публичного population result не заполняются. Loss affects accepted flow;
division by offered λ вместо accepted rate неверно. Finite-loss average не
становится infinite mean только от малого residual.

## 2. Точный B=1 и предел λ→0

При B=1 полный request service H равен сумме всех prefill/decode iterations,
F — сумме только до first output. Классический exact infinite FCFS результат:

```text
E[W] = λ E[H²] / (2(1−λE[H])),   λE[H]<1,
E[TTFT] = E[W]+E[F],   E[V]=E[W]+E[H].
```

H,F могут коррелировать через joint marks; waiting не зависит от собственного
iid нового mark. Нужен **full H**, не F, в загрузке и втором моменте PK.
В shortcut B=1 finite K не используется, это отдельный явно именованный контракт.
При λ→0 и B>1 overlapping iterations имеют исчезающую вероятность, поэтому
stage mean стремится к E[F]; проверено независимой isolated token enumeration.
Теоретическая isolated/formula проверка не называется аппаратной проверкой.

## 3. Арбитры и frozen experiment

Тесты: independent scalar full/first enumeration, PK для mixed marks,
zero waiting capacity cycle renewal, λ→0, overload/guards/malformed law.
Дополнительно независимый arrival-event finite-loss replay для K=0,2,4:
потери решаются по фактическим приходам внутри iteration, первый токен
timestamped, analytic transitions/rewards не импортируются.
У этого oracle неограниченный итоговый drain; measured admissions/denominators
не равны offered arrivals. Он не заменяет infinite DES.

Основная validation: прежний независимый infinite token DES. Все48 prespecified
сочетаний B∈{1,2}, M∈{2,4},2laws,2cost sets,3loads;
3 seeds по12000 requests,20%warmup. Не подгоняли cost/support по test.
Общее число generated requests1,728,000. Primary median absolute error —
описательная характеристика этой сетки; точная погрешность относительно
неизвестного infinite mean не определена 3 DES replications.
CI среднего DES — t interval по3 независимым seeds, не по correlated requests.

| Сетка | Literal Л3 MAE | Clamp MAE | Constant-speed LPS MAE | Stage MAE | Снижение к literal |
|---|---:|---:|---:|---:|---:|
| Все48 | .0361035 | .0343363 | .0138767 | .00198081 | 94.51% |
| Только B>1,24 | .0224232 | .0290307 | .129688 | .00149751 | 93.32% |
| Heldout costs,24 | .0361035 | .0350866 | .0125635 | .00198081 | 94.51% |
| B=1,24 | .118828 | .121169 | .00465041 | .00465041 | 96.09% |

Порог25% пройден описательно. В B=1 stage совпадает с известным PK,
и LPS baseline с K=1 тоже сводится к нему — **не новый результат**.
Диагностический homogeneous B>1 subset также даёт93.74% reduction;
для mixed B>1 —92.56%. Общую разницу нельзя приписывать исключительно
учёту «возраста»: одновременно меняются deterministic residuals, joint support
и boundary-state description; отдельная mechanism ablation здесь не выполнена.
При B>1 queue refinements2→4→6 дали max loss0.0018123 дляK6, но
max relative TTFT differenceK4→K6≈4.51%; sensitivity обязательно остаётся
в JSON. Это не rigorously bounded infinite-queue error.

## 4. Стоимость, выбор и границы гипотезы

K6 embedded state counts42,50,1270,1907 в разных laws/M. Solver wall timings
в этом запуске≈.0064–.1930s; median одного DES CPU run≈.1044s при B>1.
Wall и CPU — разные часы, поэтому точный speedup из них не выводится;
механизм state explosion и порядок цен уже не позволяют обещать универсальную
вычислительную экономию. [Дополнительный same-clock runtime check](stage-aware-ttft-timing-2026.json)
для4 прежних laws/M при load.5 (3 repeats,12000-job DES, generation исключён
для обоих методов) даёт solver/DES wall ratio.0711/.0666 для homogeneous
M2/M4 и1.8320/1.5602 для mixed M2/M4. CPU ratios практически такие же.
Следовательно homogeneous reference быстрее, mixed уже медленнее этого
короткого DES; benchmark diagnostic, machine-specific, не universal speed claim.
Это повторная проверка runtime без настройки accuracy parameters.
Большие empirical supports явно отказывают по budget.
Ни phase compression, ни scalable real-length predictor здесь не реализованы.

Mean-TTFT configuration diagnostic: сравниваются cap1/2 **при фиксированном M**,
одинаковой law/cost/load и rate. Across M rate меняется по isolated work,
так что cross-M choice/regret не считается. Все24 cap pairs одинаково верно
выбрали literal, clamp и stage по средним DES (во всех pairs выбран cap2);
LPS adaptation chose0/24. Это ranking по Monte Carlo reference, не доказанный
строгий порядок неизвестных точных средних. Последний baseline имеет
другую constant-speed discipline: это не опровержение Zhang, а предел применения
простейшей механической адаптации его (6.28–6.30) к этому batch-cost model.
Таким образом улучшение среднего не доказало улучшения проектного выбора.
SLO tails не рассчитывались; выводы ГЛ2 не экстраполируются этим iid predictor.

Локально доступна RTX3080Ti Laptop16GB; установлены torch/transformers,
но vLLM/SGLang и согласованная модель/cost calibration отсутствуют. GPU inference
не выполнялся. Нельзя писать «нет GPU», но нельзя объявлять hardware validation
по наличию оборудования. Весовые модели/внешние GPU не загружались/не арендовались.

## 5. Литература и решение

[Zhang2009](https://repository.gatech.edu/bitstreams/7e8688af-edf1-4c43-a81f-edae8dc4d4c2/download)
прочитан выборочно: Ch.2 (measure-valued residual state), §1.2 и §6.3
(6.27–6.31 и область LPS approximation). Остаточные размеры и ограниченный
sharing уже известны; нашей особенности boundary costs/first token недостаточно
для заявления нового общего метода. Все limit proofs не аудированы.
[Bekker2005](https://ir.cwi.nl/pub/14125) прочитаны выбранные §1.6.3 и §2.1–2.2:
finite workload buffers и measures наблюдения, general state-dependent rates.
Его workload buffer не тождествен count waiting capacity нашей модели.
[Bari et al.2508.01002v2](https://arxiv.org/abs/2508.01002v2) — уже сохранённый
сосед joint bounded S/O, RAD/SLAI и two-phase service; не replica нашего калькулятора.
[Li et al.2606.18431v1](https://arxiv.org/abs/2606.18431v1) показывает, что stage/
distribution awareness и robustness широко исследуются в настоящих serving stacks.

**Решение:** сохранить стадийный calculator как reference small-law tool.
Тезис о точности поддержан на frozen grid, тезис о компактности не выполнен,
новизна/публикационная готовность не заявлены. Следующий осмысленный шаг —
отдельный compression/closure method с контролем ошибки и выигрышем runtime,
затем hardware calibration и сложный выбор конфигурации. Простое добавление
более длинных состояний и клэмп формулы новым методом не являются.

```bash
python -m examples.stage_aware_ttft_experiment
pytest tests/units/test_stage_aware_ttft.py
```

Итоговая совместная проверка ГЛ2/ГЛ3: fast3236passed, full3256passed,
16 прежних warnings;37 новых targeted tests; black/isort/pylint новых модулей
чистые. Timing diagnostic отдельно воспроизводится через
`python -m examples.stage_aware_ttft_timing`.
