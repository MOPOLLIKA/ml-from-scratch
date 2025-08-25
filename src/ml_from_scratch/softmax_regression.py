import numpy as np
from ml_from_scratch.helpers import cls_to_idx, softmax, idx_to_cls, standardize, onehot, cross_entropy


class SoftReg:
    def __init__(self, random_seed=1):
        self._W: np.ndarray
        self._n_classes: int
        self._classes: np.ndarray
        self._fitted = False
        self._rng = np.random.default_rng(random_seed)
        self._epoch_stopped = 0
        self._losses = []

    def fit(self, X, y, n_epochs=500, eta=1e-2, tolerance=1e-3):
        self._classes = np.unique(y)
        y_true = cls_to_idx(y)
        m = X.shape[0]
        n_classes = len(np.unique(y_true))
        n_features = X.shape[1]
        # n_features+1 to account for the bias term
        self._W = self._rng.random(size=(n_features+1, n_classes))
        # batch gradient descent
        X_std = standardize(X)
        X_padded = np.insert(X_std, 0, 1, axis=1)
        loss_prev = np.inf
        for _ in range(n_epochs):
            self._epoch_stopped += 1
            scores = X_padded @ self._W
            proba = softmax(scores)
            y_true_onehot = onehot(y_true)
            error = proba - y_true_onehot
            nabla = (error.T @ X_padded / m).T
            self._W = self._W - eta*nabla

            y_pred_proba = self.predict_proba(X)
            loss = cross_entropy(y_pred_proba, y_true_onehot)
            self._losses.append(loss)
            if loss > loss_prev-tolerance:
                break
            loss_prev = loss

    def predict_proba(self, X):
        X_std = standardize(X)
        X_padded = np.insert(X_std, 0, 1, axis=1)
        scores = X_padded @ self._W
        return softmax(scores)

    def predict(self, X):
        X_std = standardize(X)
        X_padded = np.insert(X_std, 0, 1, axis=1)
        scores = X_padded @ self._W
        # no need to softmax the scores here
        indices = np.argmax(scores, axis=1)
        return idx_to_cls(self._classes, indices)