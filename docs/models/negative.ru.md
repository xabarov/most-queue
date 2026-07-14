# Системы с отрицательными заявками

[🇬🇧 English version](negative.md) · [← Каталог моделей](../models.ru.md)

![Отрицательные заявки: RCS и disaster](../figures/negative.ru.png)

**Простыми словами:** кроме обычных заявок в систему приходит второй, «вредительский» поток —
отрицательные заявки (Gelenbe, G-networks). Такая заявка сама не обслуживается, а уничтожает
чужую работу: в варианте **RCS** — сбивает заявку с прибора (сбой, вирус, отмена задачи),
в варианте **disaster** — очищает всю систему (перезагрузка, катастрофа). Модели считают,
сколько в итоге теряется и насколько дольше живут уцелевшие заявки.

### M/G/1 RCS (Remove Customer from Service)

**Описание:** Система, где отрицательные заявки удаляют заявку из обслуживания.

**Класс расчета:** `MG1NegativeCalcRCS`

**Пример:**

```python
from most_queue.theory.negative.mg1_rcs import MG1NegativeCalcRCS

calc = MG1NegativeCalcRCS()
calc.set_sources(l=0.5, l_neg=0.1)  # l_neg - интенсивность отрицательных заявок
calc.set_servers(b)
results = calc.run()
```

### M/G/1 Disaster

**Описание:** Одноканальная система, где отрицательная заявка удаляет все заявки из системы.

**Класс расчета:** `MG1Disasters` (`most_queue.theory.negative.mg1_disasters`)

**Пример:** См. тест `test_mg1_disaster.py`

### M/G/c RCS

**Описание:** Многоканальная система с отрицательными заявками типа RCS.

**Класс расчета:** `MGnNegativeRCSCalc`

### M/G/c Disaster

**Описание:** Система, где отрицательные заявки удаляют все заявки из системы (и из очереди, и из обслуживания).

**Класс расчета:** `MGnNegativeDisasterCalc`

**Пример:** См. тесты `test_mgn_disaster.py` и `test_mg1_disaster.py`
