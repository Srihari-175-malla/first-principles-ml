import numpy as np
from typing import Tuple, List, Dict, Any

class PCAScratch:
    """
    Principal Component Analysis (PCA) implemented from first principles in NumPy.
    Computes sample covariance eigen-decomposition, dimensionality reduction transformation,
    reconstruction inverse transformation, and explained variance ratios.
    """
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.mean = None
        self.components = None # Eigenvectors V_k (shape: n_components, D)
        self.singular_values = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray) -> "PCAScratch":
        N, D = X.shape
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Sample Covariance Matrix C = (1 / (N - 1)) X^T X
        cov_matrix = (X_centered.T @ X_centered) / (N - 1.0)

        # Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Store principal components
        self.explained_variance_ = eigenvalues[:self.n_components]
        tot_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / (tot_variance + 1e-15)

        # Components shape: (n_components, D)
        self.components = eigenvectors[:, :self.n_components].T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """
        Reconstruct original data matrix from reduced PCA representation.
        X_hat = X_pca @ V_k + mean
        """
        return (X_pca @ self.components) + self.mean

    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Calculate Mean Squared Reconstruction Error ||X - X_hat||^2 / N.
        """
        X_pca = self.transform(X)
        X_rec = self.inverse_transform(X_pca)
        return float(np.mean((X - X_rec) ** 2))
