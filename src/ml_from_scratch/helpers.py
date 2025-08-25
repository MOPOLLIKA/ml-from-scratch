import numpy as np
import os

def softmax(scores):
    return np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)

def cross_entropy(proba, y_true_onehot):
    """
    ### Inputs:
    1) `soft_scores` as probabilities of target labels.
    2) `y_true_onehot` as a onehot-encoded target label matrix.
    ### Outputs:
    - cross entropy score.
    Numerically stable method, due to clipping
    """
    eps = 1e-14
    clipped_scores = np.clip(proba, eps, 1-eps)
    return -np.sum(y_true_onehot * np.log(clipped_scores)) / proba.shape[0]

def onehot(y_true, n_classes_def=0):
    """
    ### Inputs:
    - `y_true` is a vector of shape (-1, 1), where each row is a unique inverse index.
    """
    n_classes = max(np.max(y_true)+1, n_classes_def)
    return np.eye(n_classes)[y_true.ravel()]

def cls_to_idx(classes) -> np.ndarray:
    classes_uniq = list(np.unique(classes))
    return np.vectorize(classes_uniq.index)(classes)

def idx_to_cls(classes_uniq, indices) -> np.ndarray:
    return np.array(classes_uniq)[indices]

def standardize(arr):
    """Standardize a 2D array of shape (n_samples, n_features) using z-score."""
    n_samples = arr.shape[0]
    means = np.sum(arr, axis=0) / n_samples
    stds = np.sqrt(np.sum(arr**2, axis=0)/n_samples - means**2)
    return (arr - means) / stds


def sigmoid(arr: np.ndarray):
    return 1 / (1 + np.e**np.negative(arr))

def save_figure(fig, name):
    path = "artifacts/figures" + f"/{name}.png"
    if os.path.exists(path):
        return None
    fig.savefig(path, dpi=600)