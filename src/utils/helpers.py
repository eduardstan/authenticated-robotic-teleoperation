import numpy as np
import random
import os
import yaml

def set_seed(seed):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def ensure_directory(directory):
    """
    Ensures that a directory exists; if it doesn't, it creates it.

    Args:
        directory (str): Path of the directory to ensure exists.
    """
    os.makedirs(directory, exist_ok=True)

def save_results_to_csv(results, filename):
    """
    Saves a dictionary of results to a CSV file.

    Args:
        results (dict): Dictionary containing the results to save.
        filename (str): File path to save the CSV file.
    """
    import pandas as pd

    # Ensure the directory exists
    ensure_directory(os.path.dirname(filename))

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def load_config(config_path):
    """
    Loads the configuration settings from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration settings as a dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_timestamp():
    """
    Returns the current timestamp as a string.

    Returns:
        str: Current timestamp in 'YYYYMMDD_HHMMSS' format.
    """
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')
