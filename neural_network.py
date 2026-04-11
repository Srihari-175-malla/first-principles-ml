import numpy as np
from typing import List, Tuple, Dict, Any, Callable

class NeuralNetworkScratch:
    """
    Fully-connected Multi-Layer Perceptron (MLP) Neural Network derived and implemented
    by hand from first principles in NumPy.
    Supports configurable architectures, activations (ReLU, Sigmoid, Tanh, Softmax),
    and vectorised backpropagation.
    """
    def __init__(
        self,
        layer_sizes: List[int], # e.g. [4, 8, 2]
        activation: str = "relu",
        output_activation: str = "softmax",
        learning_rate: float = 0.05
    ):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.lr = learning_rate

        self.weights = []
        self.biases = []
        self.loss_history = []

        # Xavier / He Weight Initialization
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
            W = np.random.randn(n_in, n_out) * scale
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

    def _activate(self, Z: np.ndarray, name: str) -> np.ndarray:
        if name == "relu":
            return np.maximum(0.0, Z)
        elif name == "sigmoid":
            Z_clamped = np.clip(Z, -25.0, 25.0)
            return 1.0 / (1.0 + np.exp(-Z_clamped))
        elif name == "tanh":
            return np.tanh(Z)
        elif name == "softmax":
            exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return Z

    def _activation_derivative(self, A: np.ndarray, Z: np.ndarray, name: str) -> np.ndarray:
        if name == "relu":
            return (Z > 0).astype(float)
        elif name == "sigmoid":
            return A * (1.0 - A)
        elif name == "tanh":
            return 1.0 - A ** 2
        return np.ones_like(A)

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        zs = []

        A = X
        L = len(self.weights)

        for l in range(L):
            W = self.weights[l]
            b = self.biases[l]
            Z = A @ W + b
            zs.append(Z)

            act_fn = self.output_activation_name if l == L - 1 else self.activation_name
            A = self._activate(Z, act_fn)
            activations.append(A)

        return activations, zs

    def fit(self, X: np.ndarray, y_onehot: np.ndarray, epochs: int = 300) -> "NeuralNetworkScratch":
        N = X.shape[0]
        self.loss_history = []

        for epoch in range(epochs):
            activations, zs = self.forward(X)
            y_pred = activations[-1]

            # Cross-entropy loss
            eps = 1e-15
            loss = -np.mean(np.sum(y_onehot * np.log(y_pred + eps), axis=1))
            self.loss_history.append(float(loss))

            # Backpropagation
            L = len(self.weights)
            dZ = y_pred - y_onehot # Softmax + Cross Entropy gradient simplifies to (A - Y)

            for l in reversed(range(L)):
                A_prev = activations[l]
                dW = (1.0 / N) * (A_prev.T @ dZ)
                db = (1.0 / N) * np.sum(dZ, axis=0, keepdims=True)

                if l > 0:
                    W_curr = self.weights[l]
                    dA_prev = dZ @ W_curr.T
                    dZ = dA_prev * self._activation_derivative(activations[l], zs[l-1], self.activation_name)

                # Gradient step
                self.weights[l] -= self.lr * dW
                self.biases[l] -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self.forward(X)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def check_gradients_numerical(nn: NeuralNetworkScratch, X: np.ndarray, y_onehot: np.ndarray, eps: float = 1e-5) -> float:
    """
    Analytical Gradient Checker: Compares backpropagation analytical gradients with
    numerical finite difference approximation (f(x+eps) - f(x-eps)) / (2*eps).
    Returns relative error difference (must be < 1e-5).
    """
    N = X.shape[0]
    activations, zs = nn.forward(X)
    y_pred = activations[-1]

    # 1. Compute Analytical Gradients via Backprop
    L = len(nn.weights)
    dZ = y_pred - y_onehot
    analytic_dW0 = (1.0 / N) * (activations[0].T @ dZ) if L == 1 else None

    if L > 1:
        dZ_curr = dZ
        for l in reversed(range(L)):
            if l == 0:
                analytic_dW0 = (1.0 / N) * (activations[0].T @ dZ_curr)
            else:
                dZ_curr = (dZ_curr @ nn.weights[l].T) * nn._activation_derivative(activations[l], zs[l-1], nn.activation_name)

    # 2. Compute Numerical Gradients via Finite Difference for W0
    W0 = nn.weights[0]
    num_dW0 = np.zeros_like(W0)

    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            # W + eps
            W0[i, j] += eps
            act_plus, _ = nn.forward(X)
            loss_plus = -np.mean(np.sum(y_onehot * np.log(act_plus[-1] + 1e-15), axis=1))

            # W - eps
            W0[i, j] -= 2 * eps
            act_minus, _ = nn.forward(X)
            loss_minus = -np.mean(np.sum(y_onehot * np.log(act_minus[-1] + 1e-15), axis=1))

            # Reset W
            W0[i, j] += eps

            num_dW0[i, j] = (loss_plus - loss_minus) / (2.0 * eps)

    # Relative Error
    diff = np.linalg.norm(analytic_dW0 - num_dW0) / (np.linalg.norm(analytic_dW0) + np.linalg.norm(num_dW0) + 1e-15)
    return float(diff)
