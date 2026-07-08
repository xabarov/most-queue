# SRPT / SPJF в Most-Queue: методы расчёта и сопоставление с симуляцией

[🇬🇧 English version](srpt_spjf_methods.md)

Эта страница дополняет [roadmap](roadmaps/srpt_spjf_roadmap.md) и краткие разделы в [численных методах](calculation.ru.md) и [симуляции](simulation.ru.md). Здесь собраны **формулы**, с которыми работают калькуляторы, **практическая численная схема** (сетки, интегрирование) и **то, как результаты сверяются с `SizeBasedQsSim`**.

Источники: Schrage–Miller (1966), Conway–Maxwell–Miller (непрерывный приоритет по размеру), Mitzenmacher (2020) по SPJF; разбор в репозитории: [SPJF.md](../works/queueing_systems_review/SRPT/SPJF.md), [shrage.md](../works/queueing_systems_review/SRPT/shrage.md).

---

## 1. Что добавлено в библиотеку (по roadmap)

| Область | Компонент | Назначение |
|--------|-----------|------------|
| Теория | `most_queue.theory.srpt.MG1SrptCalc` | M/G/1, **SRPT** |
| Теория | `MG1SjfCalc` | M/G/1, **SJF** (непрерывное SPT) |
| Теория | `MG1PsjfCalc` | M/G/1, **PSJF** |
| Теория | `MG1SpjfCalc` + `Predictor` | M/G/1, **SPJF** по модели \((X,Y)\) |
| Теория | `most_queue.theory.srpt.utils.load_below`, `predictor` | \(\rho_x\), \(\rho'_y\), маргиналь \(g_Y\), «идеальный» и шумовые предикторы |
| Симуляция | `most_queue.sim.size_based.SizeBasedQsSim` | Одноканальная очередь: FCFS, SJF, PSJF, SRPT, SPJF, PSPJF, SPRPT |
| Симуляция | `Task.original_size`, `predicted_size`, `service_remaining` | Размер при приходе, предсказание, остаток при прерывании |
| Симуляция | `PrioritySizeQueue`, предикторы sim-слоя | Heap по рангу; `PerfectSimPredictor`, шум через `sim.utils.predictor` |

Калькуляторы наследуют общий паттерн `BaseQueue`: `set_sources(λ)`, `set_servers(params, kendall_notation)` — как у `MG1Calc`, но для размера строится **численная** плотность/CDF через `build_pdf_cdf` (поддерживаются распределения из `most_queue.random.distributions` с известной параметризацией).

---

## 2. Аналитические формулы (что именно считается)

Обозначения: интенсивность входа \(\lambda\), PDF/CDF размера \(f,F\), \(\mathbb{E}[S]=b_0\), \(\mathbb{E}[S^2]=b_1\). Частичная нагрузка от заявок размера **не больше** \(x\):

$$
\rho_x = \lambda \int_0^x t\,f(t)\,dt .
$$

(В литературе это же часто пишут как \(\rho_{\le x}\) для непрерывного приоритета по размеру.)

### 2.1. SRPT (Schrage–Miller)

Условное среднее **время пребывания** заявки размера \(x\):

$$
\mathbb{E}[T^{\mathrm{SRPT}}(x)] = \frac{\lambda \int_0^x t^2 f(t)\,dt \;+\; \lambda\, x^2 \bigl(1-F(x)\bigr)}{2\bigl(1-\rho_x\bigr)^2} \;+\; \int_0^x \frac{dt}{1-\rho_t}.
$$

Безусловные средние:

$$
\mathbb{E}[T^{\mathrm{SRPT}}] = \int_0^\infty f(x)\,\mathbb{E}[T^{\mathrm{SRPT}}(x)]\,dx, \qquad \mathbb{E}[W^{\mathrm{SRPT}}] = \mathbb{E}[T^{\mathrm{SRPT}}] - b_0 .
$$

В коде `MG1SrptCalc` используется эквивалентная запись первого слагаемого (см. docstring класса); второе слагаемое — отдельный накопленный интеграл по сетке.

### 2.2. SJF (непрерывный размерный приоритет, non-preemptive)

$$
\mathbb{E}[W^{\mathrm{SJF}}(x)] = \frac{\lambda\, b_1}{2\bigl(1-\rho_x\bigr)^2}, \qquad \mathbb{E}[W^{\mathrm{SJF}}] = \int_0^\infty f(x)\,\mathbb{E}[W^{\mathrm{SJF}}(x)]\,dx, \qquad \mathbb{E}[T^{\mathrm{SJF}}] = \mathbb{E}[W^{\mathrm{SJF}}] + b_0 .
$$

### 2.3. PSJF (прерывание по **исходному** размеру)

$$
\mathbb{E}[T^{\mathrm{PSJF}}(x)] = \frac{\lambda \int_0^x t^2 f(t)\,dt}{2\bigl(1-\rho_x\bigr)^2} \;+\; \frac{x}{1-\rho_x},
$$

$$
\mathbb{E}[T^{\mathrm{PSJF}}] = \int_0^\infty f(x)\,\mathbb{E}[T^{\mathrm{PSJF}}(x)]\,dx, \qquad \mathbb{E}[W^{\mathrm{PSJF}}] = \mathbb{E}[T^{\mathrm{PSJF}}] - b_0 .
$$

Отличие от SRPT: нет члена \(\int_0^x dt/(1-\rho_t)\), так как ранг не «догоняет» остаток работы.

### 2.4. SPJF (Mitzenmacher 2020)

Совместная плотность \((X,Y)\) истинного размера и предсказания; эффективная нагрузка от заявок с предсказанием \(\le y\):

$$
\rho'_y = \lambda \int_0^y \!\! \int_0^\infty t\, g(t,z)\,dt\,dz .
$$

Условное среднее **ожидания** при фиксированном предсказании \(y\):

$$
\mathbb{E}[W^{\mathrm{SPJF}}(y)] = \frac{\lambda\, b_1}{2\bigl(1-\rho'_y\bigr)^2}, \qquad \mathbb{E}[W^{\mathrm{SPJF}}] = \int_0^\infty g_Y(y)\,\mathbb{E}[W^{\mathrm{SPJF}}(y)]\,dy, \qquad \mathbb{E}[T^{\mathrm{SPJF}}] = \mathbb{E}[W^{\mathrm{SPJF}}] + b_0 ,
$$

где \(g_Y\) — маргинальная плотность \(Y\). При **идеальных** предсказаниях (\(Y=X\)) результат совпадает с SJF (`PerfectPredictor` в теории / `PerfectSimPredictor` в симуляции).

---

## 3. Как это считается в коде (численно)

### 3.1. Общая база SRPT / SJF / PSJF — `_SizeBasedCalcBase`

Файл: `most_queue/theory/srpt/_base.py`.

1. **Верхняя граница по \(x\)**  
   `x_max = upper_bound(cdf, p=1e-7)` — практический «хвост» распределения, дальше масса пренебрежима.

2. **Гибридная сетка по \([0, x_{\max}]\)**  
   Часть узлов — **логарифмическая** окрестность 0 (острые PDF вроде Gamma/Pareto), часть — **равномерная** на остатке. Константа `_N_GRID = 4000` задаёт число узлов.

3. **Однократные интегралы по сетке** (`scipy.integrate.cumulative_trapezoid`):
   - \(\rho_x\) как \(\lambda \int_0^x t f(t)\,dt\);
   - \(\int_0^x t^2 f(t)\,dt\);
   - для SRPT: \(\int_0^x dt/(1-\rho_t)\) по узлам сетки (с \(\rho_t\), обрезанным чуть ниже 1, чтобы избежать численного переполнения);
   - значения CDF на тех же узлах.

4. **Быстрые обращения**  
   Для произвольного \(x\) используется **`np.interp`** к предвычисленным массивам — это замена вложенных квадратур «на каждой точке внешнего интеграла».

5. **Безусловное усреднение по \(X\)** — вычисление

   $$
   \int_0^{x_{\max}} f(x)\, h(x)\,dx
   $$

   где \(h(x)\) — условное \(\mathbb{E}[T(x)]\) или \(\mathbb{E}[W(x)]\) через интерполяцию. Реализация: **`scipy.integrate.simpson`** на **той же сетке**, т.е. вектор `pdf(xs) * h(xs)` и `simpson(..., x=xs)`.

   **Зачем Simpson, а не внешний `quad`:** подынтегральное выражение содержит множители вида \((1-\rho_x)^{-2}\); при \(\rho \to 1\) адаптивный `quad` часто даёт предупреждения о roundoff на хвосте у \(x_{\max}\). Интеграл по **фиксированной** плотной сетке с Simpson стабильнее и согласован с уже посчитанными \(\rho_x\).

### 3.2. Отдельно: `MG1SpjfCalc`

Здесь нет общего грида по \(x\) для внешнего интеграла по \(y\): маргиналь \(g_Y(y)\) и \(\rho'_y\) задаются объектом **`Predictor`** (`marginal_y_pdf`, `load_below_y`). Безусловное \(\mathbb{E}[W]\) считается **`scipy.integrate.quad`** по \(y \in [0, \infty)\) с ужесточёнными `epsabs`, `epsrel`, `limit` (см. `mg1_spjf.py`), потому что интеграл идёт по другой переменной и по плотности, которую задаёт предиктор.

### 3.3. Устойчивость при высокой загрузке

- Теория: плотная сетка + Simpson для внешнего среднего по \(X\) (SRPT/SJF/PSJF); отдельные регрессионные тесты на \(\rho \in \{0.95, 0.99\}\) для сценария из Mitzenmacher–Shahout (см. `tests/test_mg1_predictions_table.py`).
- Симуляция: дисперсия оценок \(\mathbb{E}[T]\), особенно для SRPT, велика; для таблиц из статьи используются сотни тысяч — миллионы заявок (см. `examples/srpt_table.py`).

---

## 4. Симуляция `SizeBasedQsSim` и соответствие теории

Идея roadmap: **sample-at-arrival** — размер \(X\) (и при необходимости предсказание \(Y\)) известны в момент прихода; фактическое время обслуживания равно сэмплированному размеру (остаток ведётся в `service_remaining`).

- **`set_servers(params, notation)`** в `SizeBasedQsSim` задаёт **то же распределение размера**, что используется для построения \(f\) в теории (через те же параметры Kendall / H2 и т.д.).
- **`set_sources(λ, "M")`** — пуассоновский поток с той же \(\lambda\), что в `set_sources` калькулятора.
- Дисциплина задаётся `discipline=...`; прерывание реализовано сравнением рангов и возвратом задачи в кучу с обновлённым остатком.

Тогда для фиксированных \((\lambda, f)\) и одной и той же дисциплины **средние по симуляции** (например, `results.v[0]`, `results.w[0]`) должны сходиться к **аналитическим** при росте числа заявок; расхождение — только статистическая ошибка + дискретизация теории.

**PSPJF / SPRPT:** в roadmap отмечено, что для них в первой итерации **аналитика в пакете может отсутствовать**; симуляция есть, теория — точка расширения. Сравнение «theory vs sim» в тестах ориентировано на SRPT, SJF, PSJF, SPJF (+ FCFS как база).

---

## 5. Как именно сравниваем с имитационным моделированием

1. **Модульные тесты** (`tests/test_mg1_srpt.py`, `test_mg1_sjf.py`, `test_mg1_psjf.py`, `test_mg1_spjf.py`): один и тот же сценарий (например, H₂ с заданными средним и CV), `MG1*Calc.run()` против `SizeBasedQsSim(...).run(N)`; допуски **`MOMENTS_RTOL` / `MOMENTS_ATOL`** из `tests/default_params.yaml`, число заявок **`NUM_OF_JOBS`** подобрано так, чтобы типичная ошибка среднего укладывалась в допуск.

2. **Неравенства и sanity-checks** (`tests/test_srpt_vs_fcfs.py`): например, \(\mathbb{E}[T^{\mathrm{SRPT}}] \le \mathbb{E}[T^{\mathrm{FCFS}}]\) для тестируемых параметров.

3. **Регрессия по таблице статьи** (`tests/test_mg1_predictions_table.py`): фиксированная модель предсказаний \(Y \mid X=x \sim \mathrm{Exp}(1/x)\), несколько \(\rho\); sim + при наличии — теория против опубликованных чисел.

4. **Пример для воспроизведения таблицы и графиков**: `examples/srpt_table.py` — теория, симуляция и эталонные значения в одном скрипте (см. также туториал `tutorials/srpt_basics.ipynb`).

Рекомендация при своих экспериментах: фиксировать **`np.random.Generator`** (`sim.generator = np.random.default_rng(seed)`), поднимать \(N\) при больших \(\rho\) и сравнивать с теорией там, где она реализована.

---

## 6. Связанные разделы документации

- [Основные концепции — size-based термины](concepts.ru.md)
- [Модели — строки M/G/1 SRPT/SJF/PSJF/SPJF](models.ru.md)
- [Численные методы — краткие формулы и примеры API](calculation.ru.md) (раздел *Size-based M/G/1 калькуляторы*)
- [Симуляция — `SizeBasedQsSim`, предикторы, slowdown](simulation.ru.md)
