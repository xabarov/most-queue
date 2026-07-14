# Systems with negative customers

[🇷🇺 Русская версия](negative.ru.md) · [← Model catalog](../models.md)

![Negative customers: RCS and disaster](../figures/negative.png)

**In plain words:** in addition to ordinary jobs, a second, "malicious" stream arrives —
negative customers (Gelenbe, G-networks). Such a customer is not served itself but destroys
someone else's work: in the **RCS** variant it knocks the job off the server (a failure, a
virus, a task cancellation); in the **disaster** variant it wipes out the whole system (a
reboot, a catastrophe). The models compute how much is ultimately lost and how much longer the
surviving jobs take.

### M/G/1 RCS (Remove Customer from Service)

**Description:** A system where negative customers remove the job in service.

**Calculator class:** `MG1NegativeCalcRCS`

**Example:**

```python
from most_queue.theory.negative.mg1_rcs import MG1NegativeCalcRCS

calc = MG1NegativeCalcRCS()
calc.set_sources(l=0.5, l_neg=0.1)  # l_neg - arrival rate of negative customers
calc.set_servers(b)
results = calc.run()
```

### M/G/1 Disaster

**Description:** Single-server system where a negative customer removes all jobs from the system.

**Calculator class:** `MG1Disasters` (`most_queue.theory.negative.mg1_disasters`)

**Example:** See the test `test_mg1_disaster.py`

### M/G/c RCS

**Description:** Multi-server system with RCS-type negative customers.

**Calculator class:** `MGnNegativeRCSCalc`

### M/G/c Disaster

**Description:** A system where negative customers remove all jobs from the system (both from the queue and from service).

**Calculator class:** `MGnNegativeDisasterCalc`

**Example:** See the tests `test_mgn_disaster.py` and `test_mg1_disaster.py`
