# src/data_processor.py

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    RepeatedStratifiedKFold,
)
from src.utils.logger import get_logger
from src.utils.helpers import load_config, ensure_directory
import os

def get_cv_strategy(X, y, groups, config, logger=None):
    """
    Returns the cross-validation strategy based on the task and configuration.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target variable.
        groups (pd.Series): Grouping variable for cross-validation.
        config (dict): Configuration settings.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        cv: Cross-validation iterator.
    """
    if logger is None:
        logger = get_logger(__name__)

    target_variable = config.get('target_variable')
    cv_config = config.get('cross_validation', {})
    random_seed = config.get('random_seed', 42)

    if target_variable == 'task':
        task_cv_config = cv_config.get('task_identification', {})
        method = task_cv_config.get('method', 'leave_one_subject_out')

        if method == 'leave_one_subject_out':
            if groups is None:
                logger.error("Groups cannot be None for Leave-One-Group-Out cross-validation.")
                raise ValueError("Groups cannot be None for Leave-One-Group-Out cross-validation.")
            logger.info("Using Leave-One-Subject-Out cross-validation for Task Identification.")
            cv = LeaveOneGroupOut()
            # cv = logo.split(X, y, groups=groups)
        else:
            logger.error(f"Unknown cross-validation method for task identification: {method}")
            raise ValueError(f"Unknown method: {method}")

    elif target_variable == 'subject':
        user_cv_config = cv_config.get('user_identification', {})
        method = user_cv_config.get('method', 'k_fold')
        n_splits = user_cv_config.get('n_splits', 3)
        n_repeats = user_cv_config.get('n_repeats', 1)

        if method == 'k_fold':
            logger.info(f"Using Stratified K-Fold cross-validation for User Identification with {n_splits} splits.")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            # cv = cv.split(X, y)
        elif method == 'repeated_k_fold':
            logger.info(f"Using Repeated Stratified K-Fold cross-validation for User Identification with {n_splits} splits and {n_repeats} repeats.")
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed)
            # cv = cv.split(X, y)
        else:
            logger.error(f"Unknown cross-validation method for user identification: {method}")
            raise ValueError(f"Unknown method: {method}")
    else:
        logger.error(f"Unsupported target variable: {target_variable}")
        raise ValueError(f"Unsupported target variable: {target_variable}")

    return cv

def process_data(config, logger=None):
    """
    Loads and processes data based on the configuration.

    Args:
        config (dict): Configuration settings.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        X (pd.DataFrame): Processed features.
        y (pd.Series): Target variable.
        cv (iterator): Cross-validation iterator.
    """
    if logger is None:
        logger = get_logger(__name__)

    # Load processed data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    X_path = os.path.join(processed_data_dir, 'features.csv')
    labels_processed_path = os.path.join(processed_data_dir, 'labels.csv')
    labels_raw_path = os.path.join(project_root, config['labels_path'])  # data/raw/labels.csv

    logger.info(f"Loading features from {X_path}")

    X = pd.read_csv(X_path)

    # Load labels
    if os.path.exists(labels_processed_path):
        logger.info(f"Loading labels from {labels_processed_path}")
        labels = pd.read_csv(labels_processed_path)
    elif os.path.exists(labels_raw_path):
        logger.info(f"Loading labels from {labels_raw_path}")
        labels = pd.read_csv(labels_raw_path, header=None)
        labels.columns = config['labels_columns']
    else:
        logger.error(f"Labels file not found in {labels_processed_path} or {labels_raw_path}")
        raise FileNotFoundError("Labels file not found.")

    # Ensure labels and features are aligned
    labels = labels.reset_index(drop=True)
    X = X.reset_index(drop=True)
    if len(labels) != len(X):
        logger.error("The number of labels does not match the number of samples in X.")
        raise ValueError("Mismatch between number of samples in X and labels.")

    # Set target variable
    target_variable = config['target_variable']
    if target_variable not in labels.columns:
        logger.error(f"Target variable '{target_variable}' not found in labels.")
        raise ValueError(f"Target variable '{target_variable}' not found in labels.")
    y = labels[target_variable]

    # Set groups if necessary
    if target_variable == 'task':
        # For Task Identification, group by 'subject'
        groups = labels['subject']
    else:
        groups = None

    # Log the distribution of the target variable
    logger.info(f"Target variable '{target_variable}' distribution:\n{y.value_counts()}")

    # Get cross-validation strategy
    cv = get_cv_strategy(X, y, groups, config, logger)

    return X, y, cv, groups

if __name__ == "__main__":
    # Initialize logger
    logger = get_logger(__name__)

    # Load configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    config = load_config(config_path)

    # Process data and get cross-validation strategy
    X, y, cv, groups = process_data(config, logger)

    if groups is not None:
        splitter = cv.split(X, y, groups=groups)
    else:
        splitter = cv.split(X, y)

    # Example usage: iterate over cross-validation splits
    for fold_idx, (train_idx, test_idx) in enumerate(splitter):
        logger.info(f"Fold {fold_idx + 1}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        logger.info(f"X_train.shape: {X_train.shape} -- y_train.shape: {y_train.shape}")
        logger.info(f"X_test.shape: {X_test.shape} -- y_test.shape: {y_test.shape}")
        # Here you can proceed to train and evaluate your model
