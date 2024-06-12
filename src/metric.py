import numpy as np

from sklearn.metrics import accuracy_score

from sknetwork.classification import get_accuracy_score


def compute_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray, penalized: bool) -> float:
    """Accuracy score.
    
    Parameters
    ----------
    labels_true: np.ndarray
        True labels.
    labels_pred: np.ndarray
        Predicted labels.
    penalized: bool
        If true, labels not predicted (with value -1) are considered in the accuracy computation.
        
    Returns
    -------
        Accuracy score.
    """
    if penalized:
        return accuracy_score(labels_true, labels_pred)
    else:
        return get_accuracy_score(labels_true, labels_pred)
