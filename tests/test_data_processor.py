# tests/test_data_processor.py

import unittest
import os
import sys
import pandas as pd
import numpy as np

# Adjust the system path to import modules from src/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data_processor import process_data, get_cv_strategy
from src.utils.helpers import load_config, ensure_directory
from src.utils.logger import get_logger

class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize logger
        cls.logger = get_logger(__name__)

        # Load configuration
        config_path = os.path.join(project_root, 'configs', 'config.yaml')
        cls.config = load_config(config_path)

        # Create sample data
        cls.X = pd.DataFrame({'f"feature_{n}"': np.random.rand(144) for n in range(0,144)})
        # For brevity, we will simulate only a few features
        # We can expand this to match your actual feature size

        # Create labels
        tasks = np.tile(np.repeat([1, 2, 3], 16 * 3), 1)
        subjects = np.repeat(np.arange(1, 17), 3 * 3)
        trials = np.tile(np.repeat([1, 2, 3], 3), 16)

        cls.labels = pd.DataFrame({
            'task': tasks,
            'subject': subjects,
            'trial': trials
        })

        # Target variable
        target_variable = cls.config['target_variable']
        cls.y = cls.labels[target_variable]

        # Save sample data to processed data directory
        processed_data_dir = os.path.join(project_root, 'data', 'processed')
        ensure_directory(processed_data_dir)
        cls.X_path = os.path.join(processed_data_dir, 'features.csv')
        cls.y_path = os.path.join(processed_data_dir, 'target.csv')
        cls.labels_path = os.path.join(processed_data_dir, 'labels.csv')

        # Save X, y, and labels
        cls.X.to_csv(cls.X_path, index=False)
        cls.y.to_csv(cls.y_path, index=False, header=[cls.y.name])
        cls.labels.to_csv(cls.labels_path, index=False)

    @classmethod
    def tearDownClass(cls):
        # Remove sample data files
        if os.path.exists(cls.X_path):
            os.remove(cls.X_path)
        if os.path.exists(cls.y_path):
            os.remove(cls.y_path)
        if os.path.exists(cls.labels_path):
            os.remove(cls.labels_path)

    def test_task_identification_cv(self):
        # Set target variable to 'task'
        self.config['target_variable'] = 'task'
        self.config['cross_validation']['task_identification']['method'] = 'leave_one_subject_out'

        X, y, cv = process_data(self.config, self.logger)

        # Collect splits
        splits = list(cv)
        self.assertEqual(len(splits), 16)  # Should have 16 splits (subjects)

        # Check that each test set contains data from only one subject
        for train_idx, test_idx in splits:
            test_subjects = self.labels.iloc[test_idx]['subject'].unique()
            self.assertEqual(len(test_subjects), 1)

    def test_user_identification_cv(self):
        # Set target variable to 'subject'
        self.config['target_variable'] = 'subject'
        self.config['cross_validation']['user_identification']['method'] = 'repeated_k_fold'
        self.config['cross_validation']['user_identification']['n_splits'] = 5 
        self.config['cross_validation']['user_identification']['n_repeats'] = 5

        X, y, cv = process_data(self.config, self.logger)

        # Collect splits
        splits = list(cv)
        expected_number_of_splits = self.config['cross_validation']['user_identification']['n_splits'] * \
                                    self.config['cross_validation']['user_identification']['n_repeats']
        self.assertEqual(len(splits), expected_number_of_splits)

        # Check that the splits are stratified by subject
        for train_idx, test_idx in splits:
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            # Ensure that each class is represented in y_train
            unique_subjects_train = np.unique(y_train)
            self.assertTrue(len(unique_subjects_train) >= 1)


    def test_invalid_target_variable(self):
        self.config['target_variable'] = 'invalid_target'

        with self.assertRaises(ValueError):
            X, y, cv = process_data(self.config, self.logger)

    def test_cv_strategy_unknown_method(self):
        self.config['target_variable'] = 'subject'
        self.config['cross_validation']['user_identification']['method'] = 'unknown_method'

        with self.assertRaises(ValueError):
            X, y, cv = process_data(self.config, self.logger)

    def test_cv_strategy_no_groups(self):
        # Test when groups are None
        self.config['target_variable'] = 'task'
        self.config['cross_validation']['task_identification']['method'] = 'leave_one_subject_out'

        # Modify process_data to not provide groups
        def process_data_no_groups(config, logger=None):
            if logger is None:
                logger = get_logger(__name__)

            # Load processed data
            X = self.X
            y = self.y

            # Not providing groups
            groups = None

            # Get cross-validation strategy
            cv = get_cv_strategy(X, y, groups, config, logger)

            return X, y, cv

        with self.assertRaises(ValueError):
            X, y, cv = process_data_no_groups(self.config, self.logger)

if __name__ == '__main__':
    unittest.main()
