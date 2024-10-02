# src/data_loader.py

import pandas as pd
import os
from src.utils.logger import get_logger
from src.utils.helpers import load_config, ensure_directory

def load_data(data_path, labels_path, config, logger=None):
    """
    Loads and merges data and labels from CSV files.

    Args:
        data_path (str): Path to the data CSV file containing features.
        labels_path (str): Path to the labels CSV file containing labels.
        config (dict): Configuration settings.
        logger (logging.Logger, optional): Logger instance for logging.

    Returns:
        pd.DataFrame: Merged DataFrame containing features and labels.
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(f"Loading data from {data_path}")
    logger.info(f"Loading labels from {labels_path}")

    # Extract header information from config
    data_has_header = config.get('data_has_header', True)
    labels_has_header = config.get('labels_has_header', True)
    data_columns = config.get('data_columns', [])
    labels_columns = config.get('labels_columns', [])

    # Check if files exist
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if not os.path.exists(labels_path):
        logger.error(f"Labels file not found: {labels_path}")
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # Load data
    try:
        if data_has_header:
            data = pd.read_csv(data_path)
        else:
            data = pd.read_csv(data_path, header=None)
            # Assign dummy or custom column names
            if data_columns:
                data.columns = data_columns
            else:
                data.columns = [f'feature_{i}' for i in range(1, data.shape[1] + 1)]
        logger.info(f"Data shape: {data.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Load labels
    try:
        if labels_has_header:
            labels = pd.read_csv(labels_path)
        else:
            labels = pd.read_csv(labels_path, header=None)
            # Assign custom column names
            if labels_columns:
                labels.columns = labels_columns
            else:
                labels.columns = [f'label_{i}' for i in range(1, labels.shape[1] + 1)]
        logger.info(f"Labels shape: {labels.shape}")
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        raise

    # Merge data and labels
    try:
        merged_data = pd.concat([data, labels], axis=1)
        logger.info(f"Merged data shape: {merged_data.shape}")
    except Exception as e:
        logger.error(f"Error merging data and labels: {e}")
        raise

    return merged_data

def preprocess_data(df, config, logger=None):
    """
    Preprocesses the data according to the provided configuration.

    Args:
        df (pd.DataFrame): DataFrame containing the merged data.
        config (dict): Configuration settings.
        logger (logging.Logger, optional): Logger instance for logging.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Preprocessed features (X) and target variable (y).
    """
    if logger is None:
        logger = get_logger(__name__)

    # Handle missing values
    missing_value_strategy = config.get('missing_value_strategy', 'mean')  # 'mean', 'median', 'drop', etc.
    logger.info(f"Handling missing values using strategy: {missing_value_strategy}")

    if missing_value_strategy == 'mean':
        df = df.fillna(df.mean())
    elif missing_value_strategy == 'median':
        df = df.fillna(df.median())
    elif missing_value_strategy == 'drop':
        df = df.dropna()
    elif missing_value_strategy == 'none':
        logger.info("No missing value handling applied.")
    else:
        logger.warning(f"Unknown missing value strategy: {missing_value_strategy}. No action taken.")

    # Specify target variable
    target_variable = config.get('target_variable', None)
    if target_variable is None:
        logger.error("Target variable not specified in config.")
        raise ValueError("Target variable not specified in config.")

    # Ensure target variable exists in the DataFrame
    if target_variable not in df.columns:
        logger.error(f"Target variable '{target_variable}' not found in the DataFrame.")
        raise ValueError(f"Target variable '{target_variable}' not found in the DataFrame.")

    # Separate features and target
    target_columns = ['task', 'subject', 'trial']
    feature_columns = df.columns.drop(target_columns)
    X = df[feature_columns]
    y = df[target_variable]
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")

    # Since all features are numerical, we skip categorical encoding
    # No need to encode target variable as it's numerical labels

    # Feature scaling
    scaling_method = config.get('scaling_method', None)  # 'standard', 'minmax', etc.
    if scaling_method:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
        logger.info(f"Applying {scaling_method} scaling to columns: {list(numeric_columns)}")

        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.warning(f"Unknown scaling method: {scaling_method}. No scaling applied.")
            scaler = None

        if scaler:
            X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

    return X, y

if __name__ == "__main__":
    # Initialize logger
    logger = get_logger(__name__)

    # Get the absolute path to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the absolute path to the config file
    config_path = os.path.join(project_root, 'configs', 'config.yaml')

    # Load configuration
    config = load_config(config_path)

    # Load data
    data_path = os.path.join(project_root, config['data_path'])
    labels_path = os.path.join(project_root, config['labels_path'])
    df = load_data(data_path, labels_path, config, logger)

    # Preprocess data
    X, y = preprocess_data(df, config, logger)

    # Save processed data
    processed_data_dir = os.path.join(project_root, config.get('processed_data_path', 'data/processed/'))
    ensure_directory(processed_data_dir)

    X.to_csv(os.path.join(processed_data_dir, 'features.csv'), index=False)
    y.to_csv(os.path.join(processed_data_dir, 'target.csv'), index=False, header=[config['target_variable']])

    logger.info(f"Processed features saved to {os.path.join(processed_data_dir, 'features.csv')}")
    logger.info(f"Processed target saved to {os.path.join(processed_data_dir, 'target.csv')}")
