import unittest
import numpy as np
from ml_library_scratch.linear_model import LinearRegressionScratch, LogisticRegressionScratch

class TestLinearModels(unittest.TestCase):
    def test_linear_regression(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_w = np.array([1.5, -2.0, 3.0])
        y = X @ true_w + 0.5

        model = LinearRegressionScratch(learning_rate=0.05, n_epochs=300).fit(X, y)
        y_pred = model.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        self.assertLess(mse, 0.1)

    def test_logistic_regression(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegressionScratch(learning_rate=0.1, n_epochs=300).fit(X, y)
        acc = np.mean(model.predict(X) == y)
        self.assertGreaterEqual(acc, 0.85)

if __name__ == "__main__":
    unittest.main()
