# src/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.utils.logger import get_logger
from src.utils.helpers import ensure_directory

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer for feature selection using different methods.
    """
    def __init__(self, method='univariate', k=10, random_state=None):
        """
        Initializes the FeatureSelector.

        Args:
            method (str): Feature selection method ('univariate', 'model').
            k (int): Number of top features to select.
            random_state (int, optional): Random state for reproducibility.
        """
        self.method = method
        self.k = k
        self.random_state = random_state
        self.selector = None

    def fit(self, X, y=None):
        if self.method == 'univariate':
            self.selector = SelectKBest(score_func=f_classif, k=self.k)
            self.selector.fit(X, y)
        elif self.method == 'mutual_info':
            self.selector = SelectKBest(score_func=mutual_info_classif, k=self.k)
            self.selector.fit(X, y)
        elif self.method == 'model':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            model.fit(X, y)
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:self.k]
            self.feature_indices_ = indices
        else:
            raise ValueError(f"Unknown feature selection method: {self.method}")
        return self

    def transform(self, X):
        if self.method in ['univariate', 'mutual_info']:
            return self.selector.transform(X)
        elif self.method == 'model':
            return X.iloc[:, self.feature_indices_]
        else:
            return X

    def get_feature_names_out(self, input_features=None):
        if self.method in ['univariate', 'mutual_info']:
            mask = self.selector.get_support()
            return np.array(input_features)[mask]
        elif self.method == 'model':
            return np.array(input_features)[self.feature_indices_]
        else:
            return np.array(input_features)

class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for dimensionality reduction using PCA.
    """
    def __init__(self, n_components=2, random_state=None):
        """
        Initializes the DimensionalityReducer.

        Args:
            n_components (int): Number of principal components to keep.
            random_state (int, optional): Random state for reproducibility.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = PCA(n_components=self.n_components, random_state=self.random_state)

    def fit(self, X, y=None):
        self.reducer.fit(X)
        return self

    def transform(self, X):
        return self.reducer.transform(X)

    def get_feature_names_out(self, input_features=None):
        return [f'PC{i+1}' for i in range(self.n_components)]

def build_feature_engineering_pipeline(config, logger=None):
    """
    Builds a feature engineering pipeline based on the configuration.

    Args:
        config (dict): Configuration settings.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        sklearn.pipeline.Pipeline: Feature engineering pipeline.
    """
    if logger is None:
        logger = get_logger(__name__)

    steps = []
    fe_config = config.get('feature_engineering', {})

    # Feature Selection
    fs_method = fe_config.get('feature_selection_method', None)
    if fs_method and fs_method != 'none':
        k = fe_config.get('k_best', 10)
        logger.info(f"Adding feature selection step: method={fs_method}, k={k}")
        feature_selector = FeatureSelector(method=fs_method, k=k, random_state=config['random_seed'])
        steps.append(('feature_selection', feature_selector))

    # Dimensionality Reduction
    dr_method = fe_config.get('dimensionality_reduction_method', None)
    if dr_method == 'pca':
        n_components = fe_config.get('n_components', 2)
        logger.info(f"Adding dimensionality reduction step: method=PCA, n_components={n_components}")
        dimensionality_reducer = DimensionalityReducer(n_components=n_components, random_state=config['random_seed'])
        steps.append(('dimensionality_reduction', dimensionality_reducer))
    elif dr_method and dr_method != 'none':
        logger.warning(f"Unknown dimensionality reduction method: {dr_method}. No action taken.")

    if steps:
        pipeline = Pipeline(steps)
        logger.info("Feature engineering pipeline created.")
    else:
        pipeline = None
        logger.info("No feature engineering steps specified.")

    return pipeline

if __name__ == "__main__":
    # Example usage
    from src.utils.helpers import load_config
    import os

    # Initialize logger
    logger = get_logger(__name__)

    # Load configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    config = load_config(config_path)

    # Load processed data
    processed_data_dir = os.path.join(project_root, 'data', 'processed')
    X_path = os.path.join(processed_data_dir, 'features.csv')
    y_path = os.path.join(processed_data_dir, 'target.csv')

    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze()  # Convert to Series if necessary

    # Build feature engineering pipeline
    fe_pipeline = build_feature_engineering_pipeline(config, logger)

    if fe_pipeline is not None:
        # Fit and transform the data
        X_transformed = fe_pipeline.fit_transform(X, y)
        feature_names = fe_pipeline.get_feature_names_out(input_features=X.columns)

        # Save transformed features
        fe_output_dir = os.path.join(processed_data_dir, 'feature_engineered')
        ensure_directory(fe_output_dir)
        X_fe_path = os.path.join(fe_output_dir, 'features_fe.csv')
        pd.DataFrame(X_transformed, columns=feature_names).to_csv(X_fe_path, index=False)
        logger.info(f"Feature-engineered data saved to {X_fe_path}")
    else:
        logger.info("No feature engineering applied. Using original features.")
