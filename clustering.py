import numpy as np
from typing import Tuple, List, Dict, Any

class KMeansScratch:
    """
    K-Means Clustering implemented from first principles in NumPy.
    Features K-Means++ initialization and iterative centroid optimization.
    """
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, seed: int = 42):
        self.k = n_clusters
        self.max_iter = max_iter
        self.seed = seed
        self.centroids = None
        self.labels_ = None
        self.inertia_ = 0.0

    def _kmeans_pp_init(self, X: np.ndarray) -> np.ndarray:
        np.random.seed(self.seed)
        N, D = X.shape
        centroids = np.zeros((self.k, D))
        centroids[0] = X[np.random.choice(N)]

        for c_idx in range(1, self.k):
            # Compute distance to closest existing centroid
            dists = np.min([np.sum((X - centroids[j]) ** 2, axis=1) for j in range(c_idx)], axis=0)
            probs = dists / np.sum(dists)
            centroids[c_idx] = X[np.random.choice(N, p=probs)]

        return centroids

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        N, D = X.shape
        self.centroids = self._kmeans_pp_init(X)

        for _ in range(self.max_iter):
            # Assign points to nearest centroid
            dists = np.array([np.sum((X - c) ** 2, axis=1) for c in self.centroids]) # shape (k, N)
            new_labels = np.argmin(dists, axis=0)

            # Update centroids
            new_centroids = np.zeros((self.k, D))
            for k in range(self.k):
                cluster_pts = X[new_labels == k]
                if len(cluster_pts) > 0:
                    new_centroids[k] = np.mean(cluster_pts, axis=0)
                else:
                    new_centroids[k] = X[np.random.choice(N)]

            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids
            self.labels_ = new_labels

        # Compute Inertia (WCSS)
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in self.centroids])
        self.labels_ = np.argmin(dists, axis=0)
        self.inertia_ = float(np.sum(np.min(dists, axis=0)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.array([np.sum((X - c) ** 2, axis=1) for c in self.centroids])
        return np.argmin(dists, axis=0)


class GMMScratch:
    """
    Gaussian Mixture Model (GMM) implemented from first principles in NumPy.
    Uses Expectation-Maximization (EM) algorithm to estimate component parameters.
    """
    def __init__(self, n_components: int = 3, max_iter: int = 100, seed: int = 42):
        self.K = n_components
        self.max_iter = max_iter
        self.seed = seed
        self.weights = None
        self.means = None
        self.covariances = None
        self.log_likelihood_history = []

    def _multivariate_gaussian(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        N, D = X.shape
        cov_reg = cov + 1e-6 * np.eye(D) # Regularization for numerical stability
        inv_cov = np.linalg.inv(cov_reg)
        det_cov = np.linalg.det(cov_reg)

        diff = X - mean
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        norm_const = 1.0 / np.sqrt(((2.0 * np.pi) ** D) * max(1e-12, det_cov))
        return norm_const * np.exp(exponent)

    def fit(self, X: np.ndarray) -> "GMMScratch":
        np.random.seed(self.seed)
        N, D = X.shape

        # Initialize parameters
        self.weights = np.ones(self.K) / self.K
        random_indices = np.random.choice(N, self.K, replace=False)
        self.means = X[random_indices].copy()
        self.covariances = np.array([np.eye(D) for _ in range(self.K)])

        self.log_likelihood_history = []

        for it in range(self.max_iter):
            # 1. E-Step: Compute Responsibilities gamma (N, K)
            probs = np.zeros((N, self.K))
            for k in range(self.K):
                probs[:, k] = self._multivariate_gaussian(X, self.means[k], self.covariances[k])

            weighted_probs = probs * self.weights
            total_prob = np.sum(weighted_probs, axis=1, keepdims=True) + 1e-15
            gamma = weighted_probs / total_prob

            log_likelihood = float(np.sum(np.log(total_prob)))
            self.log_likelihood_history.append(log_likelihood)

            # 2. M-Step: Update Weights, Means, Covariances
            Nk = np.sum(gamma, axis=0) # shape (K,)

            for k in range(self.K):
                # Update weights
                self.weights[k] = Nk[k] / N

                # Update means
                self.means[k] = np.sum(gamma[:, k:k+1] * X, axis=0) / (Nk[k] + 1e-15)

                # Update covariances
                diff = X - self.means[k]
                self.covariances[k] = (diff.T @ (gamma[:, k:k+1] * diff)) / (Nk[k] + 1e-15)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        probs = np.zeros((N, self.K))
        for k in range(self.K):
            probs[:, k] = self._multivariate_gaussian(X, self.means[k], self.covariances[k])
        weighted_probs = probs * self.weights
        return weighted_probs / (np.sum(weighted_probs, axis=1, keepdims=True) + 1e-15)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
