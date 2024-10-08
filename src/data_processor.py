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

def get_cv_strategy(X, y, labels, groups, config, logger=None):
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
    task_type = config.get('task_type', 'user_id')
    cv_config = config.get('cross_validation', {})
    random_seed = config.get('random_seed', 42)


    if task_type == 'user_auth':
        logger.info("Using custom cross-validation for User Authentication.")
        cv = user_authentication_cv(labels, config, logger)
    else:
        if target_variable == 'task':
            task_cv_config = cv_config.get('task_identification', {})
            method = task_cv_config.get('method', 'leave_one_subject_out')

            if method == 'leave_one_subject_out':
                if groups is None:
                    logger.error("Groups cannot be None for Leave-One-Group-Out cross-validation.")
                    raise ValueError("Groups cannot be None for Leave-One-Group-Out cross-validation.")
                logger.info("Using Leave-One-Subject-Out cross-validation for Task Identification.")
                cv = LeaveOneGroupOut()
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
            elif method == 'repeated_k_fold':
                logger.info(f"Using Repeated Stratified K-Fold cross-validation for User Identification with {n_splits} splits and {n_repeats} repeats.")
                cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_seed)
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

    # Get task type
    task_type = config.get('task_type', 'user_id')  # Default to 'user_id' if not specified

    if task_type == 'user_auth':
        logger.info("Processing data for User Authentication task.")
        # For user authentication, we need to create multiple datasets, one for each user
        # We'll handle the cross-validation strategy accordingly
        groups = None  # Not used in this context
    else:
        # Existing code for 'user_id' and 'task_id'
        # Set groups if necessary
        if target_variable == 'task':
            # For Task Identification, group by 'subject'
            groups = labels['subject']
        else:
            groups = None

    # Log the distribution of the target variable
    logger.info(f"Target variable '{target_variable}' distribution:\n{y.value_counts()}")

    # Get cross-validation strategy
    cv = get_cv_strategy(X, y, labels, groups, config, logger)

    return X, y, cv, labels, groups

def user_authentication_cv(labels, config, logger=None):
    if logger is None:
        logger = get_logger(__name__)

    subjects = labels['subject'].unique()
    random_seed = config.get('random_seed', 42)
    np.random.seed(random_seed)

    for target_subject in subjects:
        logger.info(f"Creating train/test splits for subject {target_subject}")

        # Randomly select trials for the target subject
        tasks = labels['task'].unique()
        trial_selection = {}
        for task in tasks:
            task_trials = labels[(labels['subject'] == target_subject) & (labels['task'] == task)]['trial'].unique()
            np.random.shuffle(task_trials)
            trial_selection[task] = {
                'train_trials': task_trials[:2],
                'test_trial': task_trials[2]
            }

        # Get training and testing indices for the target subject
        target_train_indices = []
        target_test_indices = []
        for task in tasks:
            train_trials = trial_selection[task]['train_trials']
            test_trial = trial_selection[task]['test_trial']

            train_idx = labels[
                (labels['subject'] == target_subject) &
                (labels['task'] == task) &
                (labels['trial'].isin(train_trials))
            ].index.values

            test_idx = labels[
                (labels['subject'] == target_subject) &
                (labels['task'] == task) &
                (labels['trial'] == test_trial)
            ].index.values

            target_train_indices.extend(train_idx)
            target_test_indices.extend(test_idx)

        # For other subjects, randomly select trials
        other_train_indices = []
        other_test_indices = []
        for subject in subjects:
            if subject == target_subject:
                continue
            for task in tasks:
                subject_trials = labels[(labels['subject'] == subject) & (labels['task'] == task)]['trial'].unique()
                np.random.shuffle(subject_trials)
                train_trials = subject_trials[:2]
                test_trial = subject_trials[2]

                train_idx = labels[
                    (labels['subject'] == subject) &
                    (labels['task'] == task) &
                    (labels['trial'].isin(train_trials))
                ].index.values

                test_idx = labels[
                    (labels['subject'] == subject) &
                    (labels['task'] == task) &
                    (labels['trial'] == test_trial)
                ].index.values

                other_train_indices.extend(train_idx)
                other_test_indices.extend(test_idx)

        # Combine indices
        train_indices = np.array(target_train_indices + other_train_indices)
        test_indices = np.array(target_test_indices + other_test_indices)

        # Create labels for binary classification
        y_train = (labels.loc[train_indices, 'subject'] == target_subject).astype(int)
        y_test = (labels.loc[test_indices, 'subject'] == target_subject).astype(int)

        yield train_indices, test_indices, y_train.values, y_test.values


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
        # Here we can proceed to train and evaluate your model
