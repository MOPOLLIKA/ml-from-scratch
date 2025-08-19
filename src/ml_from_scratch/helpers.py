import numpy as np

def softmax(scores):
    return np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1)

def cross_entropy(soft_scores, y_true_onehot):
    """
    ### Inputs:
    1) `soft_scores` as probabilities of target labels.
    2) `y_true_onehot` as a onehot-encoded target label matrix.
    ### Outputs:
    - cross entropy score.
    Numerically stable method, due to clipping
    """
    eps = 1e-14
    clipped_scores = np.clip(soft_scores, eps, 1-eps)
    return -np.sum(y_true_onehot * np.log(clipped_scores)) / soft_scores.shape[0]

def onehot(y_true, n_classes_def=0):
    """
    ### Inputs:
    - `y_true` is a vector of shape (-1, 1), where each row is a unique inverse index.
    """
    n_classes = max(np.max(y_true)+1, n_classes_def)
    return np.eye(n_classes)[y_true.ravel()]
