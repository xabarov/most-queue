"""
QS simulation exceptions
"""


class QsSourseSettingException(ValueError):
    """
        The error indicates incorrect initialization of the distribution parameters
    """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Incorrect initialization of the distribution parameters {self.message}"
        return "Incorrect initialization of the distribution parameters"


class QsWrongQueueTypeException(ValueError):
    """
        The error indicates incorrect initialization of QsQueue
    """

    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return f"Incorrect initialization of Queue {self.message}"
        return "Incorrect initialization of Queue"
