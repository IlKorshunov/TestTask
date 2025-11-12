import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from feature_selector import FeatureSelector

class TestFeatureSelector(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        self.df = pd.DataFrame({
            'delta': np.random.randn(n_samples) * 0.5 + 2.5,
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples) * 2,
            'feature3': np.random.randn(n_samples) * 0.001,
            'feature4': np.random.randn(n_samples),
            'element_id': range(n_samples),
            'tank_id': np.random.randint(1, 10, n_samples)
        })
        self.df['feature5'] = self.df['feature1'] * 0.99
    
    def test_prepare_data_removes_ids(self):
        selector = FeatureSelector(self.df, target_col='delta')
        self.assertNotIn('element_id', selector.X.columns)
        self.assertNotIn('tank_id', selector.X.columns)
    
    def test_remove_low_variance(self):
        selector = FeatureSelector(self.df, target_col='delta')
        initial_features = len(selector.X.columns)
        removed = selector.remove_low_variance(threshold=0.01)
        self.assertIsInstance(removed, list)
        self.assertLessEqual(len(selector.X.columns), initial_features)
    
    def test_remove_correlated(self):
        selector = FeatureSelector(self.df, target_col='delta')
        removed = selector.remove_correlated(threshold=0.95)
        self.assertIsInstance(removed, list)
    
    def test_select_with_trees(self):
        selector = FeatureSelector(self.df, target_col='delta')
        selected = selector.select_with_trees(top_n=3)
        self.assertIsInstance(selected, list)
        self.assertLessEqual(len(selected), 3)
    
    def test_run(self):
        selector = FeatureSelector(self.df, target_col='delta')
        df_selected, summary = selector.run(top_n=5)
        self.assertIsInstance(df_selected, pd.DataFrame)
        self.assertIn('delta', df_selected.columns)
        self.assertLessEqual(len(df_selected.columns), 6)

if __name__ == '__main__':
    unittest.main()

