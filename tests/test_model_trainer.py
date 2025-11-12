import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        n_samples = 200
        n_test = 30
        self.df = pd.DataFrame({
            'delta': list(np.random.randn(n_samples) * 0.5 + 2.5) + [np.nan] * n_test,
            'feature1': np.random.randn(n_samples + n_test),
            'feature2': np.random.randn(n_samples + n_test) * 2,
            'feature3': np.random.randn(n_samples + n_test) * 0.5,
            'category': np.random.choice(['A', 'B', 'C'], n_samples + n_test)
        })
    
    def test_init(self):
        trainer = ModelTrainer(self.df, target_col='delta')
        self.assertIsInstance(trainer.X_train, pd.DataFrame)
        self.assertIsInstance(trainer.y_train, np.ndarray)
        self.assertEqual(len(trainer.df_train), 200)
        self.assertEqual(len(trainer.df_test), 30)
        self.assertEqual(len(trainer.X_train), 200)
    
    def test_prepare_data(self):
        trainer = ModelTrainer(self.df, target_col='delta')
        X_train, X_val, X_test, y_train, y_val, y_test = trainer._prepare_data()
        
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_val, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertGreater(len(X_train), len(X_val))
        self.assertEqual(len(X_train) + len(X_val), 200)
        self.assertEqual(len(X_test), 30)
        self.assertIsNone(y_test)
    
    def test_run_without_tuning(self):
        trainer = ModelTrainer(self.df, target_col='delta')
        model = trainer.run(tune_hyperparams=False)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()

