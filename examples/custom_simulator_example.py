"""
Пример создания кастомного симулятора с дополнительным событием обслуживания

Этот пример демонстрирует создание симулятора с периодическим событием
обслуживания оборудования (maintenance), которое временно останавливает
все серверы.
"""

from most_queue.sim.base import QsSim


class MaintenanceSimulator(QsSim):
    """
    Симулятор с периодическим обслуживанием оборудования.

    Каждые maintenance_interval единиц времени все серверы останавливаются
    для обслуживания, прерывая текущее обслуживание заявок.
    """

    def __init__(self, num_of_channels, maintenance_interval, buffer=None, verbose=True):
        """
        Инициализация симулятора с обслуживанием.

        Args:
            num_of_channels: Количество каналов обслуживания
            maintenance_interval: Интервал между обслуживаниями
            buffer: Максимальная длина очереди (None = бесконечная)
            verbose: Выводить ли информацию во время симуляции
        """
        super().__init__(num_of_channels, buffer, verbose)

        self.maintenance_interval = maintenance_interval
        self.next_maintenance_time = maintenance_interval
        self.maintenance_count = 0
        self.tasks_interrupted = 0

    def _get_custom_events(self):
        """
        Регистрация кастомных событий.

        Возвращает словарь с кастомными событиями.
        Ключ - тип события, значение - время события.
        """
        return {"maintenance": self.next_maintenance_time}

    def _handle_custom_event(self, event_type):
        """
        Обработка кастомных событий.

        Args:
            event_type: Тип события для обработки
        """
        if event_type == "maintenance":
            self._perform_maintenance()
        else:
            # Для неизвестных событий вызываем базовый метод
            super()._handle_custom_event(event_type)

    def _perform_maintenance(self):
        """
        Выполнение обслуживания оборудования.

        Останавливает все активные серверы и прерывает обслуживание заявок.
        """
        self.maintenance_count += 1

        # Подсчитываем прерванные заявки
        interrupted = 0
        for i, server in enumerate(self.servers):
            if not server.is_free:
                # Прервать обслуживание
                interrupted_task = server.end_service()
                self._free_servers.add(i)
                self.free_channels += 1
                interrupted += 1

                # Вернуть заявку в очередь
                interrupted_task.start_waiting_time = self.ttek
                self.queue.append(interrupted_task)

        self.tasks_interrupted += interrupted

        # Запланировать следующее обслуживание
        self.next_maintenance_time = self.ttek + self.maintenance_interval

        # Обновить кэш времени серверов
        self._mark_servers_time_changed()

        if self.verbose:
            print(f"Maintenance #{self.maintenance_count} at time {self.ttek:.2f}, " f"interrupted {interrupted} tasks")

    def get_maintenance_stats(self):
        """
        Получить статистику по обслуживаниям.

        Returns:
            dict: Словарь со статистикой
        """
        return {
            "maintenance_count": self.maintenance_count,
            "tasks_interrupted": self.tasks_interrupted,
            "avg_interruptions_per_maintenance": (
                self.tasks_interrupted / self.maintenance_count if self.maintenance_count > 0 else 0
            ),
        }


def example_usage():
    """
    Пример использования MaintenanceSimulator.
    """
    from most_queue.random.distributions import H2Params

    # Создание симулятора
    # 3 канала, обслуживание каждые 100 единиц времени
    sim = MaintenanceSimulator(num_of_channels=3, maintenance_interval=100.0, buffer=None, verbose=True)

    # Настройка источников заявок (гиперэкспоненциальное распределение)
    source_params = H2Params(p1=0.5, mu1=1.0, mu2=2.0)
    sim.set_sources(source_params, kendall_notation="H")

    # Настройка серверов (экспоненциальное распределение)
    sim.set_servers(1.5, kendall_notation="M")

    # Запуск симуляции
    results = sim.run(total_served=1000)

    # Вывод результатов
    print("\n=== Результаты симуляции ===")
    print(f"Utilization: {results.utilization:.3f}")
    print(f"Mean sojourn time: {results.v[0]:.3f}")
    print(f"Mean wait time: {results.w[0]:.3f}")

    # Статистика по обслуживаниям
    maintenance_stats = sim.get_maintenance_stats()
    print("\n=== Статистика обслуживаний ===")
    print(f"Number of maintenances: {maintenance_stats['maintenance_count']}")
    print(f"Total tasks interrupted: {maintenance_stats['tasks_interrupted']}")
    print(f"Avg interruptions per maintenance: " f"{maintenance_stats['avg_interruptions_per_maintenance']:.2f}")


if __name__ == "__main__":
    example_usage()
