import numpy as np
from typing import Optional, List, Dict, Any

class SVMScratch:
    """
    Support Vector Machine (SVM) classifier implemented from first principles in NumPy.
    Supports Linear and RBF (Gaussian) kernels solved via a simplified Sequential Minimal Optimization (SMO) algorithm.
    """
    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma: float = 0.5, max_iter: int = 100):
        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.alphas = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.support_indices = None

    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.kernel_name == "linear":
            return X1 @ X2.T
        elif self.kernel_name == "rbf":
            # ||x1 - x2||^2 = ||x1||^2 + ||x2||^2 - 2 x1 x2^T
            sq_norms1 = np.sum(X1 ** 2, axis=1, keepdims=True)
            sq_norms2 = np.sum(X2 ** 2, axis=1, keepdims=True)
            dists = sq_norms1 + sq_norms2.T - 2.0 * (X1 @ X2.T)
            return np.exp(-self.gamma * np.maximum(0.0, dists))
        return X1 @ X2.T

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMScratch":
        N, D = X.shape
        # Convert y to {-1, +1} if provided as {0, 1}
        y_signed = np.where(y <= 0, -1.0, 1.0)

        self.X_train = X
        self.y_train = y_signed
        self.alphas = np.zeros(N)
        self.b = 0.0

        K = self._kernel(X, X)

        # Simplified SMO Algorithm
        for it in range(self.max_iter):
            num_changed_alphas = 0
            for i in range(N):
                # Decision function for sample i
                E_i = float(np.sum(self.alphas * self.y_train * K[:, i]) + self.b - self.y_train[i])

                # Check KKT conditions
                if (self.y_train[i] * E_i < -0.01 and self.alphas[i] < self.C) or (self.y_train[i] * E_i > 0.01 and self.alphas[i] > 0):
                    # Pick random j != i
                    j = np.random.choice([idx for idx in range(N) if idx != i])
                    E_j = float(np.sum(self.alphas * self.y_train * K[:, j]) + self.b - self.y_train[j])

                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]

                    # Compute bounds L and H
                    if self.y_train[i] != self.y_train[j]:
                        L = max(0.0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0.0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])

                    if L == H:
                        continue

                    # Eta parameter
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    # Update alpha_j
                    self.alphas[j] -= (self.y_train[j] * (E_i - E_j)) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)

                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue

                    # Update alpha_i
                    self.alphas[i] += self.y_train[i] * self.y_train[j] * (alpha_j_old - self.alphas[j])

                    # Update bias b
                    b1 = self.b - E_i - self.y_train[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - self.y_train[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - self.y_train[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - self.y_train[j] * (self.alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                pass

        self.support_indices = np.where(self.alphas > 1e-4)[0]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        K_test = self._kernel(X, self.X_train)
        return (K_test @ (self.alphas * self.y_train)) + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)
