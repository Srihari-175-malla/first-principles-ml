import numpy as np
from typing import Optional, Tuple, List, Dict, Any

class LinearRegressionScratch:
    """
    Linear Regression implemented from first principles in NumPy.
    Supports closed-form normal equation and gradient descent (Batch, Mini-Batch, SGD)
    with L1 (Lasso) and L2 (Ridge) regularization.
    """
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 500,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        optimizer: str = "batch", # 'batch', 'mini_batch', 'sgd'
        batch_size: int = 32
    ):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.l1 = l1_penalty
        self.l2 = l2_penalty
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionScratch":
        N, D = X.shape
        self.weights = np.zeros(D)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.n_epochs):
            if self.optimizer == "sgd":
                indices = np.random.permutation(N)
                for idx in indices:
                    xi = X[idx:idx+1]
                    yi = y[idx:idx+1]
                    self._update_step(xi, yi, 1)
            elif self.optimizer == "mini_batch":
                indices = np.random.permutation(N)
                for start_idx in range(0, N, self.batch_size):
                    batch_idx = indices[start_idx:start_idx+self.batch_size]
                    xb = X[batch_idx]
                    yb = y[batch_idx]
                    self._update_step(xb, yb, len(batch_idx))
            else: # batch
                self._update_step(X, y, N)

            # Record epoch loss
            y_pred = self.predict(X)
            mse = float(np.mean((y_pred - y) ** 2))
            l1_loss = self.l1 * np.sum(np.abs(self.weights))
            l2_loss = 0.5 * self.l2 * np.sum(self.weights ** 2)
            self.loss_history.append(mse + l1_loss + l2_loss)

        return self

    def _update_step(self, Xb: np.ndarray, yb: np.ndarray, n: int):
        y_pred = Xb @ self.weights + self.bias
        err = y_pred - yb

        dw = (1.0 / n) * (Xb.T @ err) + self.l2 * self.weights + self.l1 * np.sign(self.weights)
        db = (1.0 / n) * np.sum(err)

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


class LogisticRegressionScratch:
    """
    Logistic Regression implemented from first principles in NumPy.
    Supports Batch, Mini-Batch, and SGD with L1/L2 regularization.
    """
    def __init__(
        self,
        learning_rate: float = 0.05,
        n_epochs: int = 500,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        optimizer: str = "batch", # 'batch', 'mini_batch', 'sgd'
        batch_size: int = 32
    ):
        self.lr = learning_rate
        self.n_epochs = n_epochs
        self.l1 = l1_penalty
        self.l2 = l2_penalty
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        N, D = X.shape
        self.weights = np.zeros(D)
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.n_epochs):
            if self.optimizer == "sgd":
                indices = np.random.permutation(N)
                for idx in indices:
                    xi = X[idx:idx+1]
                    yi = y[idx:idx+1]
                    self._update_step(xi, yi, 1)
            elif self.optimizer == "mini_batch":
                indices = np.random.permutation(N)
                for start_idx in range(0, N, self.batch_size):
                    batch_idx = indices[start_idx:start_idx+self.batch_size]
                    xb = X[batch_idx]
                    yb = y[batch_idx]
                    self._update_step(xb, yb, len(batch_idx))
            else: # batch
                self._update_step(X, y, N)

            # Binary cross-entropy loss
            probs = self.predict_proba(X)
            eps = 1e-15
            bce = -np.mean(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
            self.loss_history.append(float(bce))

        return self

    def _update_step(self, Xb: np.ndarray, yb: np.ndarray, n: int):
        probs = self._sigmoid(Xb @ self.weights + self.bias)
        err = probs - yb

        dw = (1.0 / n) * (Xb.T @ err) + self.l2 * self.weights + self.l1 * np.sign(self.weights)
        db = (1.0 / n) * np.sum(err)

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
