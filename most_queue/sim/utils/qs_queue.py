"""
QS Queue (Buffer) implementation
"""

from abc import ABCMeta, abstractmethod
from collections import deque


class QSQueue(ABCMeta):
    """abstract Queue class
    """
    @abstractmethod
    def append(cls, item):
        """
        Append task to queue
        """

    @abstractmethod
    def pop(cls):
        """
        Get first task (from head of queue)
        """

    @abstractmethod
    def size(cls):
        """
        Get size of queue
        """


class QsQueueList:
    """
    QS Queue (Buffer) List implementation
    """

    def __init__(self):

        self.queue = []

    def append(self, task):
        """
        Append task to queue
        """

        self.queue.append(task)

    def pop(self):
        """
        Get first task (from head of queue)
        """
        return self.queue.pop(0)
    
    def tail(self):
        """
        Get last task (from tail of queue)
        """
        return self.queue[-1] if self.queue else None

    def size(self):
        """
        Get size of queue
        """
        return len(self.queue)
    
    def clear(self):
        """
        Creal the queue
        """
        self.queue.clear()
    
    def __len__(self):
        return len(self.queue)


class QsQueueDeque:
    """
    QS Queue (Buffer) deque implementation
    """

    def __init__(self):

        self.queue = deque()

    def append(self, task):
        """
        Append task to queue
        """

        self.queue.append(task)

    def pop(self):
        """
        Get first task (from head of queue)
        """
        return self.queue.popleft()

    def tail(self):
        """
        Get last task (from tail of queue)
        """
        return self.queue.pop() if self.queue else None
    
    def clear(self):
        """
        Creal the queue
        """
        self.queue.clear()

    def size(self):
        """
        Get size of queue
        """
        return len(self.queue)
