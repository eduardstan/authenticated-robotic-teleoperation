# src/models.py

import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import joblib
from src.utils.logger import get_logger
from src.utils.helpers import load_config, ensure_directory
from src.data_processor import process_data
from src.feature_engineering import build_feature_engineering_pipeline

def build_model_pipeline(config, logger=None):
    """
    Builds the model pipelines and parameter grids based on the configuration.

    Args:
        config (dict): Configuration settings.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        list: List of tuples (Pipeline, param_grid) for each model.
    """
    if logger is None:
        logger = get_logger(__name__)

    param_grid_list = []

    for model_cfg in config.get('models', []):
        steps = []
        model_name = model_cfg['name']
        parameters = model_cfg.get('parameters', {})
        model_params = {}

        # Preprocessing: Scaling
        scaling_method = config.get('scaling_method', 'standard')
        if scaling_method == 'standard':
            scaler = StandardScaler()
            logger.info(f"Using StandardScaler for feature scaling in {model_name} pipeline.")
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            logger.info(f"Using MinMaxScaler for feature scaling in {model_name} pipeline.")
        else:
            scaler = None
            logger.info(f"No scaling will be applied in {model_name} pipeline.")

        if scaler:
            steps.append(('scaler', scaler))

        # Feature Engineering Pipeline
        fe_pipeline = build_feature_engineering_pipeline(config, logger)
        if fe_pipeline:
            steps.append(('feature_engineering', fe_pipeline))

        if model_name.lower() == 'randomforest':
            model = RandomForestClassifier(random_state=config.get('random_seed', 42))
            param_prefix = 'classifier__'
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'svm':
            model = SVC(probability=True, random_state=config.get('random_seed', 42))
            param_prefix = 'classifier__'
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'logisticregression':
            model = LogisticRegression(max_iter=1000, random_state=config.get('random_seed', 42))
            param_prefix = 'classifier__'
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'knn':
            model = KNeighborsClassifier()
            param_prefix = 'classifier__'
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'decisiontree':
            model = DecisionTreeClassifier(random_state=config.get('random_seed', 42))
            param_prefix = 'classifier__'
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'naivebayes':
            model = GaussianNB()
            param_prefix = 'classifier__'
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'gradientboosting':
            model = GradientBoostingClassifier(random_state=config.get('random_seed', 42))
            param_prefix = 'classifier__'
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        else:
            logger.warning(f"Unknown model specified: {model_name}. Skipping.")
            continue

        # Add classifier to the pipeline
        steps.append(('classifier', model))

        # Create the pipeline
        pipeline = Pipeline(steps)
        logger.info(f"Modeling pipeline for {model_name} created.")

        # Log the parameter grid
        logger.info(f"Parameter grid for {model_name}: {model_params}")

        param_grid_list.append((pipeline, model_params))

    return param_grid_list

def train_and_evaluate(config, logger=None):
    """
    Trains and evaluates models based on the configuration.

    Args:
        config (dict): Configuration settings.
        logger (logging.Logger, optional): Logger instance.
    """
    if logger is None:
        logger = get_logger(__name__)

    # Process data and get cross-validation strategy
    X, y, cv, groups = process_data(config, logger)

    # Log the type of cv
    logger.info(f"Type of cv object: {type(cv)}")

    # Log the number of splits
    try:
        n_splits = cv.get_n_splits(X, y, groups=groups)
        logger.info(f"Number of cross-validation splits: {n_splits}")
    except TypeError:
        # Some CV strategies do not require all parameters
        n_splits = cv.get_n_splits(groups=groups)
        logger.info(f"Number of cross-validation splits: {n_splits}")
    except AttributeError:
        logger.warning("cv object does not have 'get_n_splits' method.")

    # Build model pipelines and parameter grids
    pipelines_with_params = build_model_pipeline(config, logger)

    # Directory to save results
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_root, 'results')
    ensure_directory(results_dir)

    all_results = []

    for idx, (pipeline, param_grid) in enumerate(pipelines_with_params):
        model_name = pipeline.named_steps['classifier'].__class__.__name__
        logger.info(f"Starting training for model: {model_name}")

        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
            'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
            'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
        }

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit='f1_macro',  # Metric to optimize
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
        )

        # Fit the model
        if groups is not None:
            grid_search.fit(X, y, groups=groups)
        else:
            grid_search.fit(X, y)

        # Get best estimator and results
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        logger.info(f"Best parameters for {model_name}: {best_params}")
        logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

        # Save results
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_csv = os.path.join(results_dir, f'grid_search_results_{model_name}.csv')
        results_df.to_csv(results_csv, index=False)
        logger.info(f"Grid search results saved to {results_csv}")

        # Save the best model
        model_path = os.path.join(results_dir, f'best_model_{model_name}.pkl')
        joblib.dump(best_model, model_path)
        logger.info(f"Best model saved to {model_path}")

        all_results.append({
            'model_name': model_name,
            'best_params': best_params,
            'best_score': grid_search.best_score_,
            'results_csv': results_csv,
            'model_path': model_path,
        })

    # Optionally, save all results to a summary file
    summary_csv = os.path.join(results_dir, 'model_summary.csv')
    pd.DataFrame(all_results).to_csv(summary_csv, index=False)
    logger.info(f"All model results summarized in {summary_csv}")

if __name__ == "__main__":
    # Initialize logger
    logger = get_logger(__name__)

    # Load configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    config = load_config(config_path)

    # Train and evaluate models
    train_and_evaluate(config, logger)
