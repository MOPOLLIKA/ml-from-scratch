import numpy as np


class GDLinReg:
    """Batch Gradient Descent linear regression model."""
    def __init__(self):
        # bias included as zero-th coef
        self._W: np.ndarray
        self._n_features: int
        self._fitted = False
    
    def fit(self, X, y, n_epochs=500, eta=1e-3):
        self._n_features = X.shape[1]
        # adding one to account for the bias term
        X_padded = np.insert(X, 0, 1, axis=1)
        self._W = np.random.rand(self._n_features+1, 1)
        for _ in range(n_epochs):
            total_error = X_padded @ self._W - y
            grad = (2/X_padded.shape[0])*(X_padded.T @ total_error)
            self._W = self._W - eta*grad
        self._fitted = True

    def predict(self, X):
        X_padded = np.insert(X, 0, 1, axis=1)
        result = X_padded @ self._W
        return result


class SVDLinReg:
    def __init__(self):
        # bias included as zero-th coef
        self._W: np.ndarray
        self._fitted = False

    def __repr__(self):
        if not self._fitted:
            return "The model has not been fitted yet."
        print("Weights:")
        print(self._W)
        return ""
    
    def fit(self, X, y):
        # adding one to account for the bias term
        X_padded = np.insert(X, 0, 1, axis=1)
        self._W = np.linalg.pinv(X_padded).dot(y)
        self._fitted = True
        return None

    def predict(self, X):
        X_padded = np.insert(X, 0, 1, axis=1)
        result = X_padded @ self._W
        return result