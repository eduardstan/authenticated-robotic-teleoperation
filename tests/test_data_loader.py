import unittest
import os
import sys
import pandas as pd

# Adjust the system path to import modules from src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)  # Add project root to sys.path

from src.data_loader import load_data, preprocess_data
from src.utils.helpers import load_config, ensure_directory
from src.utils.logger import get_logger

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize logger
        cls.logger = get_logger(__name__)
        
        # Load configuration
        config_path = os.path.join(project_root, 'configs', 'config.yaml')  # Adjust the path if necessary
        cls.config = load_config(config_path)
        
        # Prepare sample data
        cls.sample_data_path = os.path.join(project_root, 'data', 'raw', 'test_data.csv')
        cls.sample_labels_path = os.path.join(project_root, 'data', 'raw', 'test_labels.csv')
        
        # Create sample data files
        data = pd.DataFrame({
            'feature_1': [0.1, 0.2, 0.3],
            'feature_2': [1.0, 1.1, 1.2],
            'feature_3': [10, 20, 30]
        })
        labels = pd.DataFrame({
            0: [1, 2, 3],  # Assuming these correspond to 'task', 'subject', 'trial'
            1: [4, 5, 6],
            2: [7, 8, 9]
        })
        
        # Ensure directories exist
        ensure_directory(os.path.dirname(cls.sample_data_path))
        ensure_directory(os.path.dirname(cls.sample_labels_path))
        
        # Save sample data without headers
        data.to_csv(cls.sample_data_path, index=False, header=False)
        labels.to_csv(cls.sample_labels_path, index=False, header=False)
        
        # Update config paths for testing
        cls.config['data_path'] = cls.sample_data_path
        cls.config['labels_path'] = cls.sample_labels_path
        cls.config['data_has_header'] = False
        cls.config['labels_has_header'] = False
        cls.config['data_columns'] = ['feature_1', 'feature_2', 'feature_3']
        cls.config['labels_columns'] = ['task', 'subject', 'trial']
        cls.config['target_variable'] = 'task'
        cls.config['processed_data_path'] = os.path.join(project_root, 'data', 'processed', 'test_processed_data.csv')
        cls.config['missing_value_strategy'] = 'mean'
        cls.config['categorical_columns'] = []  # No categorical columns
        cls.config['scaling_method'] = 'standard'
        
    @classmethod
    def tearDownClass(cls):
        # Remove sample data files
        if os.path.exists(cls.sample_data_path):
            os.remove(cls.sample_data_path)
        if os.path.exists(cls.sample_labels_path):
            os.remove(cls.sample_labels_path)
        # Remove processed data if created
        processed_data_dir = os.path.dirname(cls.config['processed_data_path'])
        if os.path.exists(processed_data_dir):
            for file in os.listdir(processed_data_dir):
                if file.startswith('test_'):
                    os.remove(os.path.join(processed_data_dir, file))
    
    def test_load_data(self):
        df = load_data(
            self.config['data_path'],
            self.config['labels_path'],
            self.config,
            self.logger
        )
        # Check if data loaded correctly
        self.assertEqual(df.shape, (3, 6))  # 3 rows, 3 features + 3 labels
        expected_columns = ['feature_1', 'feature_2', 'feature_3', 'task', 'subject', 'trial']
        self.assertListEqual(list(df.columns), expected_columns)
    
    def test_preprocess_data(self):
        df = load_data(
            self.config['data_path'],
            self.config['labels_path'],
            self.config,
            self.logger
        )
        X, y = preprocess_data(df, self.config, self.logger)
        # Check if features and target are separated correctly
        self.assertEqual(X.shape, (3, 3))  # 3 samples, 3 features
        self.assertEqual(y.shape, (3,))    # 3 samples
        # Check if target variable is correct
        self.assertTrue(all(y == pd.Series([1, 2, 3], name='task')))
        self.assertListEqual(list(X.columns), ['feature_1', 'feature_2', 'feature_3'])
    
    # Additional test methods as previously provided...

if __name__ == '__main__':
    unittest.main()
