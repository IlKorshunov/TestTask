import unittest
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hyperparameter_tuner import HyperparameterTuner

class TestHyperparameterTuner(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        self.X_train = np.random.randn(n_samples, n_features)
        self.y_train = np.random.randn(n_samples, 1)
    
    def test_init(self):
        tuner = HyperparameterTuner(self.X_train, self.y_train, cv=3)
        self.assertEqual(tuner.cv, 3)
        self.assertIsNotNone(tuner.rmse_scorer)
    
    def test_tune_ridge(self):
        tuner = HyperparameterTuner(self.X_train, self.y_train, cv=3)
        params, score = tuner.tune_ridge()
        self.assertIsInstance(params, dict)
        self.assertIn('alpha', params)
        self.assertIsInstance(score, float)
    
    def test_tune_lasso(self):
        tuner = HyperparameterTuner(self.X_train, self.y_train, cv=3)
        params, score = tuner.tune_lasso()
        self.assertIsInstance(params, dict)
        self.assertIn('alpha', params)
        self.assertIsInstance(score, float)

if __name__ == '__main__':
    unittest.main()

