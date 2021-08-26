import typing as th
from sklearn import metrics


def accuracy(y, y_hat) -> float:
    return metrics.accuracy_score(y, y_hat)


def f1(y, y_hat, alpha: float = 0.5, beta: float = 1.):
    return metrics.f1_score(y, y_hat, average=None)


def precision(y, y_hat) -> float:
    return metrics.precision_score(y, y_hat, average=None)

def recall(y, y_hat) -> float:
    return metrics.recall_score(y, y_hat, average=None)


evaluation_functions = dict(accuracy=accuracy, f1=f1, precision=precision, recall=recall)


def evaluate(y, y_hat) -> th.Dict[str, float]:
    """
    :param y: ground truth
    :param y_hat: model predictions
    :return: a dictionary containing evaluated scores for provided values
    """
    return {name: func(y, y_hat) for name, func in evaluation_functions.items()}
