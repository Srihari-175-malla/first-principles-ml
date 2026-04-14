import unittest
import numpy as np
from ml_library_scratch.neural_network import NeuralNetworkScratch, check_gradients_numerical
from ml_library_scratch.svm import SVMScratch
from ml_library_scratch.clustering import KMeansScratch, GMMScratch
from ml_library_scratch.pca import PCAScratch

class TestMLComponents(unittest.TestCase):
    def test_neural_network_gradient_check(self):
        np.random.seed(42)
        X = np.random.randn(10, 4)
        y_onehot = np.eye(3)[np.random.randint(0, 3, size=10)]

        nn = NeuralNetworkScratch(layer_sizes=[4, 8, 3], activation="relu", output_activation="softmax", learning_rate=0.01)
        rel_diff = check_gradients_numerical(nn, X, y_onehot, eps=1e-5)
        self.assertLess(rel_diff, 1e-4)

    def test_svm_rbf(self):
        np.random.seed(42)
        X = np.random.randn(40, 2)
        y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1.0).astype(int)

        svm = SVMScratch(C=1.0, kernel="rbf", gamma=0.5, max_iter=50).fit(X, y)
        acc = np.mean(svm.predict(X) == y)
        self.assertGreaterEqual(acc, 0.75)

    def test_kmeans_and_gmm(self):
        np.random.seed(42)
        X = np.vstack([np.random.randn(20, 2) + 3.0, np.random.randn(20, 2) - 3.0])

        km = KMeansScratch(n_clusters=2).fit(X)
        self.assertGreater(km.inertia_, 0.0)

        gmm = GMMScratch(n_components=2, max_iter=20).fit(X)
        self.assertEqual(len(gmm.log_likelihood_history), 20)

    def test_pca_reconstruction(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        pca = PCAScratch(n_components=3).fit(X)
        X_pca = pca.transform(X)
        self.assertEqual(X_pca.shape, (50, 3))
        X_rec = pca.inverse_transform(X_pca)
        self.assertEqual(X_rec.shape, (50, 5))
        self.assertGreater(np.sum(pca.explained_variance_ratio_), 0.5)

if __name__ == "__main__":
    unittest.main()
