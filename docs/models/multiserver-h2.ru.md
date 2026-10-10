# Многоканальные H₂-системы (метод Такахаси-Таками)

[🇬🇧 English version](multiserver-h2.md) · [← Каталог моделей](../models.ru.md) ·
[← FIFO системы](fifo.ru.md)

**Простыми словами:** H₂ («смесь двух экспонент») — универсальная заготовка: подобрав её
параметры по среднему и коэффициенту вариации, можно приблизить почти любое реальное
распределение (при CV < 1 — с комплексными параметрами). Метод Такахаси-Таками — итерационный
численный алгоритм, точно решающий многоканальные фазовые модели, где H₂ используется для потока,
обслуживания, или обоих сразу.

### H₂/M/c

**Описание:** Многоканальная система с гиперэкспоненциальным потоком поступления (H₂) и экспоненциальным обслуживанием (M). Использует упрощённый алгоритм §7.6.1 (формулы для z_j, x_j, t_{j,i}, уровень 0).

**Класс расчета:** `H2MnCalc`

**Пример:**

```python
from most_queue.theory.fifo.gmc_takahasi import H2MnCalc
from most_queue.random.distributions import H2Distribution

calc = H2MnCalc(n=3)

h2_params = H2Distribution.get_params_by_mean_and_cv(1.0, 1.2, is_clx=True)  # mean, cv
#
# Для CV<1 используйте complex-fit: is_clx=True.
# Важно: симулятор `QsSim` не умеет генерировать H2 с комплексными параметрами,
# поэтому сравнение с симуляцией возможно только когда параметры вещественные.
calc.set_sources(h2_params)

calc.set_servers(b=2.0)  # среднее время обслуживания

results = calc.run()
```

### H₂/H₂/c

**Описание:** Многоканальная система с гиперэкспоненциальным потоком поступления и гиперэкспоненциальным обслуживанием. Использует алгоритм §7.6.2 (CH7).

**Класс расчета:** `HkHkNCalc`

**Пример:**

```python
from most_queue.theory.fifo.hkhk_takahasi import HkHkNCalc
from most_queue.random.distributions import H2Distribution

calc = HkHkNCalc(n=3, k=2)

h2_arr = H2Distribution.get_params_by_mean_and_cv(1.0, 1.2)
# Для CV<1 используйте complex-fit: is_clx=True (тогда параметры могут быть комплексными).
calc.set_sources(u=[h2_arr.p1, 1 - h2_arr.p1], lam=[h2_arr.mu1, h2_arr.mu2])

h2_srv = H2Distribution.get_params_by_mean_and_cv(2.0, 1.2)
calc.set_servers(y=[h2_srv.p1, 1 - h2_srv.p1], mu=[h2_srv.mu1, h2_srv.mu2])

results = calc.run()
```

**Примечание про CV<1:** при \(CV<1\) H₂ аппроксимация использует *complex-fit* (комплексные параметры).
Симулятор `QsSim` не генерирует H₂ с комплексными параметрами, поэтому для валидации удобно
сравнивать расчёт (H₂ complex-fit) с симуляцией эквивалентной по mean/CV `Gamma`-модели (см. тесты `tests/test_tt_vs_sim_gamma_cvl1.py`).

### M/H₂/c

**Описание:** Многоканальная система с пуассоновским потоком и гиперэкспоненциальным обслуживанием. Использует численный метод Такахаси-Таками с комплексными параметрами.

**Класс расчета:** `MGnCalc`

**Пример:**

```python
from most_queue.theory.fifo.mgn_takahasi import MGnCalc
from most_queue.random.distributions import H2Distribution

calc = MGnCalc(n=5)

calc.set_sources(l=2.0)

h2_params = H2Distribution.get_params_by_mean_and_cv(f1=2.0, cv=1.2, is_clx=True)
calc.set_servers(h2_params)

results = calc.run()
```
