import numpy as np


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def nasa_score(y_true, y_pred):
    """Official C-MAPSS/PHM08 asymmetric scoring function (Saxena & Goebel, 2008).
    Penalizes late predictions (predicted RUL > true RUL, i.e. failure predicted
    too late) far more heavily than early ones, matching the real cost of a
    missed maintenance window."""
    d = np.asarray(y_pred) - np.asarray(y_true)
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score))
