"""
Classes for QS Job (Task) 
"""
from colorama import Fore, Style


class Task:
    """
    QS Simple Job
    """
    id = 0

    def __init__(self, arr_time, is_network=False):
        """
        :param arr_time: Момент прибытия в СМО
        """
        
        if is_network:
            self.arr_network = arr_time
            self.wait_network = 0
            
        self.arr_time = arr_time

        self.start_waiting_time = 0

        self.wait_time = 0

        Task.id += 1
        self.id = Task.id

    def __str__(self):
        res = f"{Fore.GREEN}Task #{self.id}{Style.RESET_ALL}\n"
        res += f"\t{Fore.CYAN}Arrival moment: {self.arr_time:8.3f}{Style.RESET_ALL}"
        return res


class SubTask:
    """
    ForkJoin Sub Task
    """
    sub_task_id = 0

    def __init__(self, arr_time, task_id):
        self.arr_time = arr_time
        self.task_id = task_id
        self.id = SubTask.sub_task_id
        SubTask.sub_task_id += 1

    def __str__(self):
        res = f"{Fore.GREEN}\tSubTask #{self.id}{Style.RESET_ALL} parent Task #{self.task_id}\n"
        res += f"\t\t{Fore.BLUE}Arrival time: {self.arr_time}{Style.RESET_ALL}\n"
        return res


class ForkJoinTask:
    """
    Job, that contains subtask_num SubTask's
    """
    task_id = 0

    def __init__(self, subtask_num, arr_time):
        self.subtask_num = subtask_num
        self.arr_time = arr_time
        self.subtasks = []
        for i in range(subtask_num):
            self.subtasks.append(SubTask(arr_time, ForkJoinTask.task_id))
        self.id = ForkJoinTask.task_id
        ForkJoinTask.task_id += 1

    def __str__(self):
        res = f"{Fore.GREEN}\tTask #{self.id}{Style.RESET_ALL}\n"
        res += f"{Fore.BLUE}\t\tArrival time: {self.arr_time}{Style.RESET_ALL}\n"
        return res


class TaskPriority:
    """
    TaskPriority class represents a task with priority in the system.
    """
    id = 0

    def __init__(self, k, arr_time, is_network=False):
        """
        arr_time: arrival moment in the system.
        k - class of the task.
        is_network - if True, then this task is a network task.
        """
        if is_network:
            self.arr_network = arr_time
            self.wait_network = 0
            self.in_node_class_num = -1

        self.arr_time = arr_time
        self.k = k
        self.start_waiting_time = -1

        self.wait_time = 0
        self.time_to_end_service = 0
        Task.id += 1
        self.id = Task.id
        self.is_pr = False

    def __str__(self, tab=''):
        tab = "  "
        arr_time_str = f"{self.arr_time:8.3f}" if isinstance(
            self.arr_time, float) else str(self.arr_time)
        res = f"{Fore.GREEN}Task #{self.id}{Style.RESET_ALL}"
        res += f" Class: {Fore.BLUE}{self.k + 1}{Style.RESET_ALL}\n"
        res += f"{tab}\t{Fore.YELLOW}Arrival moment:{Style.RESET_ALL}"
        res += f"  {Fore.CYAN}{arr_time_str}{Style.RESET_ALL}\n"
        if self.time_to_end_service != 0:
            res += f"{tab}\t{Fore.MAGENTA}End service moment:{Style.RESET_ALL}"
            res += f" {Fore.LIGHTGREEN_EX}{self.time_to_end_service:.3f}{Style.RESET_ALL}\n"

        return res


class ImpatientTask(Task):
    """
    A task that can leave the queue if it has not been served by a server within a certain time.
    """

    def __init__(self, arr_time, moment_to_leave, is_network=False):
        super().__init__(arr_time, is_network)
        self.moment_to_leave = moment_to_leave

    def __str__(self):
        report = f'{Fore.GREEN}Task # {self.id}{Style.RESET_ALL}\n'
        report += f'{Fore.BLUE}Arrival moment: {self.arr_time:8.3f}{Style.RESET_ALL}\n'
        report += f'{Fore.CYAN}Moment to leave: {self.moment_to_leave:8.3f}{Style.RESET_ALL}'
        return report


class ImpatientTaskWithRepairs(ImpatientTask):
    """
    A task that can leave the queue if it has not been served by a server within a certain time
    Tracking if the task arrived in repair mode or not.
    """

    def __init__(self, arr_time, moment_to_leave, arrive_in_repair_mode=False, is_network=False):
        super().__init__(arr_time, moment_to_leave, is_network)
        self.arrive_in_repair_mode = arrive_in_repair_mode
        self.end_repair_time = 1e16
        self.is_end_repair = False

    def __str__(self):
        return f'{Fore.GREEN}Task # {self.id}{Style.RESET_ALL}\n' \
            f'{Fore.BLUE}Arrival moment: {self.arr_time:8.3f}{Style.RESET_ALL}\n' \
            f'{Fore.CYAN}Moment to leave: {self.moment_to_leave:8.3f}{Style.RESET_ALL}\n' \
            f'{Fore.MAGENTA}Arrive in repair mode: {self.arrive_in_repair_mode}{Style.RESET_ALL}'
