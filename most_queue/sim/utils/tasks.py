"""
Classes for QS Job (Task) 
"""


class Task:
    """
    QS Simple Job
    """
    id = 0

    def __init__(self, arr_time):
        """
        :param arr_time: Момент прибытия в СМО
        """
        self.arr_time = arr_time

        self.start_waiting_time = 0

        self.wait_time = 0

        Task.id += 1
        self.id = Task.id

    def __str__(self):
        res = f"Task #{self.id}\n"
        res += f"\tArrival moment: {self.arr_time:8.3f}"
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
        res = "\tSubTask #" + str(self.id) + \
            " parent Task #" + str(self.task_id) + "\n"
        res += "\t\tArrival time: " + str(self.arr_time) + "\n"
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
        res = "\tTask #" + str(self.id) + "\n"
        res += "\t\tArrival time: " + str(self.arr_time) + "\n"
        return res


