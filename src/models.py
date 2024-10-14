import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import UndefinedMetricWarning
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from src.utils.logger import get_logger
from src.utils.helpers import set_seed, load_config, ensure_directory
from src.data_processor import process_data
from src.feature_engineering import build_feature_engineering_pipeline
import warnings

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

        param_prefix = 'classifier__'
        if model_name.lower() == 'randomforest':
            model = RandomForestClassifier(class_weight='balanced', random_state=config.get('random_seed', 42))
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'svm':
            model = SVC(class_weight='balanced', probability=True, random_state=config.get('random_seed', 42))
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'logisticregression':
            model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=config.get('random_seed', 42))
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'knn':
            model = KNeighborsClassifier()
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'decisiontree':
            model = DecisionTreeClassifier(random_state=config.get('random_seed', 42))
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'naivebayes':
            model = GaussianNB()
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'gradientboosting':
            model = GradientBoostingClassifier(random_state=config.get('random_seed', 42))
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'xgboost':
            model = XGBClassifier(random_state=config.get('random_seed', 42))
            for param, values in parameters.items():
                model_params[param_prefix + param] = values
        elif model_name.lower() == 'lightgbm':
            model = LGBMClassifier(random_state=config.get('random_seed', 42), verbose=-1)
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

def compute_eer(y_true, y_scores, positive_label=1):
    """
    Compute Equal Error Rate (EER) for binary classification.

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_scores (array-like): Predicted scores (probabilities or decision function values).
        positive_label (int, optional): Label considered as positive class. Defaults to 1.

    Returns:
        float: Equal Error Rate (EER).
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=positive_label)
    fnr = 1 - tpr

    # Find the threshold where FPR and FNR are closest
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer

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
    X, y, cv, labels, groups = process_data(config, logger)

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

    task_type = config.get('task_type', 'user_id')

    # Suppress UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    if task_type == 'user_auth':
        # For user authentication, we use the best-performing model from user identification
        # Assuming only one model is specified in the config for user authentication
        # Modify the code to use only the specified model and parameters without GridSearchCV
        for idx, (pipeline, param_grid) in enumerate(pipelines_with_params):
            model_name = pipeline.named_steps['classifier'].__class__.__name__
            logger.info(f"Starting training for model: {model_name}")

            # Collect metrics across all users
            fold_accuracies = []
            fold_precisions = []
            fold_recalls = []
            fold_f1s = []
            fold_eers = []

            # Since we are not tuning hyperparameters, we use the provided parameters directly
            # Set the model parameters
            classifier = pipeline.named_steps['classifier']
            classifier.set_params(**{k.split('__')[1]: v[0] for k, v in param_grid.items()})

            subjects = labels['subject'].unique()

            for subject_idx, target_subject in enumerate(subjects):
                logger.info(f"Processing user {target_subject} ({subject_idx + 1}/{len(subjects)})")

                # Create binary target variable
                y_binary = (labels['subject'] == target_subject).astype(int)

                # Split data into train/test
                train_indices, test_indices = get_user_auth_split(labels, target_subject, config, logger)

                X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
                y_train, y_test = y_binary.iloc[train_indices], y_binary.iloc[test_indices]

                # Fit the model
                pipeline.fit(X_train, y_train)

                # Predict on test set
                y_pred = pipeline.predict(X_test)

                # Evaluate performance
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                eer = compute_eer(y_test, y_pred)

                fold_accuracies.append(accuracy)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                fold_f1s.append(f1)
                fold_eers.append(eer)

                logger.info(f"User {target_subject} Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, EER: {eer:.4f}")

            # Aggregate results
            avg_accuracy = np.mean(fold_accuracies)
            avg_precision = np.mean(fold_precisions)
            avg_recall = np.mean(fold_recalls)
            avg_f1 = np.mean(fold_f1s)
            avg_eer = np.mean(fold_eers)

            # Since we used predefined parameters, we can report them directly
            common_best_params = {k.split('__')[1]: v[0] for k, v in param_grid.items()}

            logger.info(f"Average Results for {model_name} - Accuracy: {avg_accuracy:.4f}, Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}, EER: {avg_eer:.4f}")
            logger.info(f"Parameters used for {model_name}: {common_best_params}")

            all_results.append({
                'model_name': model_name,
                'average_accuracy': avg_accuracy,
                'average_precision': avg_precision,
                'average_recall': avg_recall,
                'average_f1': avg_f1,
                'average_eer': avg_eer,
                'parameters_used': common_best_params,
            })

    else:
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

def get_user_auth_split(labels, target_subject, config, logger=None):
    """
    Generates train and test indices for user authentication task for a given subject.

    Args:
        labels (pd.DataFrame): DataFrame containing 'subject', 'task', 'trial'.
        target_subject (int): The subject to authenticate.
        config (dict): Configuration settings.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        train_indices (np.ndarray): Indices for training data.
        test_indices (np.ndarray): Indices for testing data.
    """
    if logger is None:
        logger = get_logger(__name__)

    random_seed = config.get('random_seed', 42)
    np.random.seed(random_seed)

    # For the target subject
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
    other_subjects = labels['subject'].unique()
    other_subjects = other_subjects[other_subjects != target_subject]
    for subject in other_subjects:
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

    return train_indices, test_indices

if __name__ == "__main__":
    # Initialize logger
    logger = get_logger(__name__)

    # Load configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, 'configs', 'config.yaml')
    config = load_config(config_path)

    set_seed(config['random_seed'])

    # Train and evaluate models
    train_and_evaluate(config, logger)
