"""
Пример создания кастомной сети массового обслуживания с периодическим обслуживанием узлов

Этот пример демонстрирует создание сети с периодическим событием обслуживания,
которое временно останавливает все узлы сети.
"""

import numpy as np

from most_queue.random.distributions import ExpDistribution
from most_queue.sim.networks.network import NetworkSimulator


class MaintenanceNetwork(NetworkSimulator):
    """
    Сеть массового обслуживания с периодическим обслуживанием узлов.

    Каждые maintenance_interval единиц времени все узлы сети останавливаются
    для обслуживания, прерывая текущее обслуживание заявок.
    """

    def __init__(self, maintenance_interval):
        """
        Инициализация сети с обслуживанием.

        Args:
            maintenance_interval: Интервал между обслуживаниями узлов
        """
        super().__init__()

        self.maintenance_interval = maintenance_interval
        self.next_maintenance_time = maintenance_interval
        self.maintenance_count = 0
        self.tasks_interrupted = 0

    def _get_custom_network_events(self):
        """
        Регистрация кастомных событий сети.

        Returns:
            dict: Словарь с кастомными событиями
        """
        return {"network_maintenance": self.next_maintenance_time}

    def _handle_custom_network_event(self, event_type):
        """
        Обработка кастомных событий сети.

        Args:
            event_type: Тип события для обработки
        """
        if event_type == "network_maintenance":
            self._perform_maintenance()
        else:
            super()._handle_custom_network_event(event_type)

    def _perform_maintenance(self):
        """
        Выполнение обслуживания всех узлов сети.

        Останавливает все активные серверы и прерывает обслуживание заявок.
        """
        self.maintenance_count += 1

        # Подсчитываем прерванные заявки
        interrupted = 0
        for node_idx, node in enumerate(self.qs):
            for channel_idx, server in enumerate(node.servers):
                if not server.is_free:
                    # Прервать обслуживание
                    interrupted_task = server.end_service()
                    node._free_servers.add(channel_idx)
                    node.free_channels += 1
                    interrupted += 1

                    # Вернуть заявку в очередь узла
                    interrupted_task.start_waiting_time = self.ttek
                    node.queue.append(interrupted_task)

            # Обновить кэш времени серверов узла
            node._mark_servers_time_changed()

        self.tasks_interrupted += interrupted

        # Запланировать следующее обслуживание
        self.next_maintenance_time = self.ttek + self.maintenance_interval

        print(
            f"Maintenance #{self.maintenance_count} at time {self.ttek:.2f}, "
            f"interrupted {interrupted} tasks across all nodes"
        )

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
    Пример использования MaintenanceNetwork.
    """
    # Создание сети с обслуживанием каждые 100 единиц времени
    network = MaintenanceNetwork(maintenance_interval=100.0)

    # Настройка источников заявок
    arrival_rate = 1.0
    # Создаем простую матрицу маршрутизации для 2 узлов
    # R[0, 0] = переход из источника в узел 0
    # R[0, 1] = переход из источника в узел 1
    # R[0, 2] = выход из системы
    R = np.matrix(
        [
            [0.0, 0.5, 0.5],  # Из источника: 50% в узел 0, 50% в узел 1
            [0.0, 0.0, 1.0],  # Из узла 0: 100% выход
            [0.0, 0.0, 1.0],  # Из узла 1: 100% выход
        ]
    )

    network.set_sources(arrival_rate, R)

    # Настройка узлов сети
    # Узел 0: 2 канала, экспоненциальное обслуживание с параметром 1.5
    # Узел 1: 1 канал, экспоненциальное обслуживание с параметром 2.0
    serv_params = [
        {"type": "M", "params": 1.5},  # Узел 0
        {"type": "M", "params": 2.0},  # Узел 1
    ]
    n = [2, 1]  # Количество каналов в каждом узле

    network.set_nodes(serv_params, n)

    # Запуск симуляции
    results = network.run(job_served=1000)

    # Вывод результатов
    print("\n=== Результаты симуляции сети ===")
    print(f"Served: {results.served}")
    print(f"Arrived: {results.arrived}")
    print(f"Mean sojourn time: {results.v[0]:.3f}")

    # Статистика по обслуживаниям
    maintenance_stats = network.get_maintenance_stats()
    print("\n=== Статистика обслуживаний ===")
    print(f"Number of maintenances: {maintenance_stats['maintenance_count']}")
    print(f"Total tasks interrupted: {maintenance_stats['tasks_interrupted']}")
    print(f"Avg interruptions per maintenance: " f"{maintenance_stats['avg_interruptions_per_maintenance']:.2f}")


if __name__ == "__main__":
    example_usage()
