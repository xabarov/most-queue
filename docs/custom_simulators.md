# Руководство по созданию кастомных симуляторов

Это руководство описывает, как создавать собственные классы симуляторов для системы массового обслуживания, используя новую event-based архитектуру.

## Обзор архитектуры

Симуляторы теперь используют event-based подход с системой событий (events) и hook-методами для расширения функциональности. Это делает создание кастомных симуляторов значительно проще.

### Основные компоненты

1. **BaseSimulationCore** - базовый класс с общей функциональностью
2. **QsSim** - основной класс симулятора с event-based логикой
3. **EventScheduler** - менеджер событий для планирования
4. **Hook-методы** - точки расширения для кастомизации

## Базовый пример

Вот минимальный пример создания кастомного симулятора:

```python
from most_queue.sim.base import QsSim

class MyCustomSimulator(QsSim):
    def __init__(self, num_of_channels, buffer=None, verbose=True):
        super().__init__(num_of_channels, buffer, verbose)
        # Ваша инициализация
    
    def _get_custom_events(self):
        """Регистрация кастомных событий"""
        return {
            'my_event': self.my_event_time  # словарь: тип_события -> время
        }
    
    def _handle_custom_event(self, event_type):
        """Обработка кастомных событий"""
        if event_type == 'my_event':
            self.handle_my_event()
        else:
            super()._handle_custom_event(event_type)
    
    def handle_my_event(self):
        """Логика обработки вашего события"""
        # Ваша логика здесь
        pass
```

## Hook-методы

### Методы для регистрации событий

#### `_get_custom_events()`

Возвращает словарь с кастомными событиями. Ключ - тип события (строка), значение - время события.

```python
def _get_custom_events(self):
    events = {}
    if self.some_condition:
        events['special_event'] = self.special_event_time
    return events
```

#### `_handle_custom_event(event_type)`

Обрабатывает кастомные события. Должен быть переопределен для обработки ваших событий.

```python
def _handle_custom_event(self, event_type):
    if event_type == 'my_event':
        # обработка
        pass
    else:
        super()._handle_custom_event(event_type)  # для неизвестных событий
```

### Методы для расширения базового поведения

#### `_before_arrival()` / `_after_arrival()`

Вызываются до и после обработки события прибытия заявки.

```python
def _before_arrival(self, moment=None, ts=None):
    # Логика перед прибытием
    pass

def _after_arrival(self, moment=None, ts=None):
    # Логика после прибытия
    pass
```

#### `_before_serving()` / `_after_serving()`

Вызываются до и после завершения обслуживания.

```python
def _before_serving(self, channel, is_network=False):
    # Логика перед обслуживанием
    pass

def _after_serving(self, channel, task=None, is_network=False):
    # Логика после обслуживания
    pass
```

### Вспомогательные методы

#### `_update_state_probs(old_time, new_time, state)`

Обновляет массив вероятностей состояний системы. Используйте этот метод вместо прямого изменения `self.p`.

```python
# Правильно:
self._update_state_probs(self.ttek, new_time, self.in_sys)

# Неправильно:
self.p[self.in_sys] += new_time - self.ttek
```

#### `_mark_servers_time_changed()`

Помечает, что время серверов изменилось. Вызывайте после изменения `time_to_end_service`.

```python
self.servers[0].time_to_end_service = new_time
self._mark_servers_time_changed()
```

## Примеры использования

### Пример 1: Симулятор с дополнительным событием

Симулятор с периодическим событием обслуживания оборудования:

```python
from most_queue.sim.base import QsSim

class MaintenanceSimulator(QsSim):
    def __init__(self, num_of_channels, maintenance_interval, buffer=None):
        super().__init__(num_of_channels, buffer)
        self.maintenance_interval = maintenance_interval
        self.next_maintenance_time = maintenance_interval
    
    def _get_custom_events(self):
        return {'maintenance': self.next_maintenance_time}
    
    def _handle_custom_event(self, event_type):
        if event_type == 'maintenance':
            # Остановка всех серверов на обслуживание
            for server in self.servers:
                if not server.is_free:
                    # Прервать обслуживание
                    server.end_service()
                    self.free_channels += 1
                    self._free_servers.add(server.id - 1)
            
            # Запланировать следующее обслуживание
            self.next_maintenance_time = self.ttek + self.maintenance_interval
            self._mark_servers_time_changed()
        else:
            super()._handle_custom_event(event_type)
```

### Пример 2: Симулятор с изменением приоритетов

Симулятор, который изменяет приоритеты заявок в зависимости от времени ожидания:

```python
from most_queue.sim.base import QsSim
from most_queue.sim.utils.tasks import Task

class DynamicPrioritySimulator(QsSim):
    def __init__(self, num_of_channels, priority_threshold, buffer=None):
        super().__init__(num_of_channels, buffer)
        self.priority_threshold = priority_threshold
    
    def _after_arrival(self, moment=None, ts=None):
        # После прибытия проверяем время ожидания в очереди
        # и повышаем приоритет старых заявок
        for task in self.queue.queue:
            wait_time = self.ttek - task.start_waiting_time
            if wait_time > self.priority_threshold:
                # Переместить в начало очереди (повысить приоритет)
                self.queue.queue.remove(task)
                self.queue.queue.appendleft(task)
                break
```

### Пример 3: Расширение существующего симулятора

Добавление функциональности к существующему симулятору:

```python
from most_queue.sim.impatient import ImpatientQueueSim

class EnhancedImpatientSimulator(ImpatientQueueSim):
    def __init__(self, num_of_channels, buffer=None):
        super().__init__(num_of_channels, buffer)
        self.reneged_count = 0  # счетчик ушедших заявок
    
    def _handle_custom_event(self, event_type):
        if event_type == 'task_drop':
            self.reneged_count += 1
            # Вызвать базовую обработку
            super()._handle_custom_event(event_type)
        else:
            super()._handle_custom_event(event_type)
```

## Переопределение стандартных событий

Если вам нужно изменить логику обработки стандартных событий (arrival, serving), вы можете переопределить методы `_get_available_events()` и `_execute_event()`:

```python
def _get_available_events(self):
    """Переопределяем для добавления дополнительной логики"""
    events = super()._get_available_events()
    
    # Добавляем фильтрацию или дополнительные события
    if self.some_condition:
        events['special_arrival'] = self.special_arrival_time
    
    return events

def _execute_event(self, event_type):
    """Переопределяем для изменения логики выполнения"""
    if event_type == 'special_arrival':
        self.handle_special_arrival()
    else:
        super()._execute_event(event_type)
```

## Паттерны для типичных расширений

### Добавление нового типа прибытия заявок

```python
def _get_available_events(self):
    events = super()._get_available_events()
    events['vip_arrival'] = self.vip_arrival_time
    return events

def _execute_event(self, event_type):
    if event_type == 'vip_arrival':
        self.handle_vip_arrival()
    else:
        super()._execute_event(event_type)
```

### Добавление периодических событий

```python
def __init__(self, ...):
    super().__init__(...)
    self.periodic_event_interval = 100.0
    self.next_periodic_event = self.periodic_event_interval

def _get_custom_events(self):
    return {'periodic': self.next_periodic_event}

def _handle_custom_event(self, event_type):
    if event_type == 'periodic':
        self.handle_periodic_event()
        self.next_periodic_event = self.ttek + self.periodic_event_interval
```

### Добавление условных событий

```python
def _get_custom_events(self):
    events = {}
    # Событие происходит только при определенных условиях
    if self.in_sys > 10:  # перегрузка системы
        events['overload_alert'] = self.ttek  # немедленно
    return events
```

## Советы и лучшие практики

1. **Всегда вызывайте `super()`** в переопределенных методах, если не заменяете функциональность полностью
2. **Используйте `_update_state_probs()`** для обновления вероятностей состояний
3. **Вызывайте `_mark_servers_time_changed()`** после изменения времени серверов
4. **Документируйте кастомные события** - используйте понятные имена и описывайте их назначение
5. **Тестируйте изолированно** - убедитесь, что ваши изменения не ломают базовую функциональность

## Миграция существующих симуляторов

Если у вас есть существующий симулятор, который переопределяет `run_one_step()`, вы можете мигрировать его следующим образом:

**До:**
```python
def run_one_step(self):
    # Сложная логика выбора следующего события
    if condition1:
        self.handle_event1()
    elif condition2:
        self.handle_event2()
    # ...
```

**После:**
```python
def _get_custom_events(self):
    events = {}
    if condition1:
        events['event1'] = self.event1_time
    if condition2:
        events['event2'] = self.event2_time
    return events

def _handle_custom_event(self, event_type):
    if event_type == 'event1':
        self.handle_event1()
    elif event_type == 'event2':
        self.handle_event2()
    # run_one_step() теперь наследуется и автоматически использует события
```

## Создание кастомных сетей массового обслуживания

Сети массового обслуживания также поддерживают event-based архитектуру. Принципы работы аналогичны одиночным симуляторам, но есть специфика работы с узлами сети.

### Базовый пример кастомной сети

```python
from most_queue.sim.networks.network import NetworkSimulator

class MyCustomNetwork(NetworkSimulator):
    def __init__(self):
        super().__init__()
        # Ваша инициализация
    
    def _get_custom_network_events(self):
        """Регистрация кастомных событий сети"""
        return {
            'network_maintenance': self.next_maintenance_time
        }
    
    def _handle_custom_network_event(self, event_type):
        """Обработка кастомных событий сети"""
        if event_type == 'network_maintenance':
            self.perform_maintenance()
        else:
            super()._handle_custom_network_event(event_type)
```

### Hook-методы для сетей

#### `_before_network_arrival(k=None)` / `_after_network_arrival(k=None)`

Вызываются до и после внешнего прибытия заявки в сеть.

```python
def _before_network_arrival(self, k=None):
    # k - номер класса для PriorityNetwork, None для обычных сетей
    # Логика перед прибытием
    pass
```

#### `_before_node_serving(node, channel)` / `_after_node_serving(node, channel, task)`

Вызываются до и после обслуживания в конкретном узле сети.

```python
def _before_node_serving(self, node, channel):
    # node - номер узла
    # channel - номер канала
    # Логика перед обслуживанием
    pass
```

### Пример: Сеть с периодическим обслуживанием узлов

```python
from most_queue.sim.networks.network import NetworkSimulator

class MaintenanceNetwork(NetworkSimulator):
    """
    Сеть с периодическим обслуживанием всех узлов.
    """
    def __init__(self, maintenance_interval):
        super().__init__()
        self.maintenance_interval = maintenance_interval
        self.next_maintenance_time = maintenance_interval
        self.maintenance_count = 0
    
    def _get_custom_network_events(self):
        """Регистрация события обслуживания"""
        return {'network_maintenance': self.next_maintenance_time}
    
    def _handle_custom_network_event(self, event_type):
        """Обработка обслуживания"""
        if event_type == 'network_maintenance':
            self.maintenance_count += 1
            # Остановить все узлы на обслуживание
            for node_idx, node in enumerate(self.qs):
                # Логика обслуживания узла
                pass
            # Запланировать следующее обслуживание
            self.next_maintenance_time = self.ttek + self.maintenance_interval
        else:
            super()._handle_custom_network_event(event_type)
    
    def _before_node_serving(self, node, channel):
        """Проверка, не идет ли обслуживание сети"""
        # Можно добавить проверки перед обслуживанием
        pass
```

### Работа с событиями узлов

События обслуживания в узлах автоматически собираются методом `_get_node_serving_events()`. Формат события: `'node_serving_{node}_{channel}'`.

Вы можете переопределить `_get_node_serving_events()` для кастомизации сбора событий:

```python
def _get_node_serving_events(self):
    """Переопределить для фильтрации или модификации событий"""
    events = super()._get_node_serving_events()
    # Добавить фильтрацию или модификацию
    return events
```

## Дополнительные ресурсы

- См. примеры в `examples/custom_simulator_example.py`
- Изучите существующие симуляторы (`vacations.py`, `impatient.py`, `negative.py`) как примеры использования
- Изучите сети (`networks/network.py`, `networks/priority_network.py`, `networks/negative_network.py`) для примеров работы с сетями
- Документация по базовым классам: `most_queue/sim/base.py` и `most_queue/sim/networks/base_network_sim.py`

