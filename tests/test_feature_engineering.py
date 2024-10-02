# tests/test_feature_engineering.py

import unittest
import os
import sys
import pandas as pd

# Adjust the system path to import modules from src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.feature_engineering import build_feature_engineering_pipeline
from src.utils.helpers import load_config
from src.utils.logger import get_logger

class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize logger
        cls.logger = get_logger(__name__)

        # Load configuration
        config_path = os.path.join(project_root, 'configs', 'config.yaml')
        cls.config = load_config(config_path)
        cls.config['random_seed'] = 42

        # Create sample data
        cls.X = pd.DataFrame({
            'feature_1': [0.1, 0.2, 0.3, 0.4],
            'feature_2': [1.0, 1.1, 1.2, 1.3],
            'feature_3': [10, 20, 30, 40],
            'feature_4': [5, 6, 7, 8],
            'feature_5': [9, 10, 11, 12],
        })
        cls.y = pd.Series([0, 1, 0, 1])

    def test_feature_selection_univariate(self):
        self.config['feature_engineering'] = {
            'feature_selection_method': 'univariate',
            'k_best': 3,
            'dimensionality_reduction_method': 'none'
        }
        fe_pipeline = build_feature_engineering_pipeline(self.config, self.logger)
        X_transformed = fe_pipeline.fit_transform(self.X, self.y)
        self.assertEqual(X_transformed.shape[1], 3)

    def test_feature_selection_model(self):
        self.config['feature_engineering'] = {
            'feature_selection_method': 'model',
            'k_best': 2,
            'dimensionality_reduction_method': 'none'
        }
        fe_pipeline = build_feature_engineering_pipeline(self.config, self.logger)
        X_transformed = fe_pipeline.fit_transform(self.X, self.y)
        self.assertEqual(X_transformed.shape[1], 2)

    def test_dimensionality_reduction_pca(self):
        self.config['feature_engineering'] = {
            'feature_selection_method': 'none',
            'dimensionality_reduction_method': 'pca',
            'n_components': 2
        }
        fe_pipeline = build_feature_engineering_pipeline(self.config, self.logger)
        X_transformed = fe_pipeline.fit_transform(self.X, self.y)
        self.assertEqual(X_transformed.shape[1], 2)

    def test_combined_feature_engineering(self):
        self.config['feature_engineering'] = {
            'feature_selection_method': 'univariate',
            'k_best': 4,
            'dimensionality_reduction_method': 'pca',
            'n_components': 2
        }
        fe_pipeline = build_feature_engineering_pipeline(self.config, self.logger)
        X_transformed = fe_pipeline.fit_transform(self.X, self.y)
        self.assertEqual(X_transformed.shape[1], 2)

    def test_no_feature_engineering(self):
        self.config['feature_engineering'] = {
            'feature_selection_method': 'none',
            'dimensionality_reduction_method': 'none'
        }
        fe_pipeline = build_feature_engineering_pipeline(self.config, self.logger)
        self.assertIsNone(fe_pipeline)

if __name__ == '__main__':
    unittest.main()
