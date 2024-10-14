# Authenticated Robotic Teleoperation with Task Recognition

A secure and responsive teleoperation system that integrates biometric authentication based on motion patterns captured by wearable inertial sensors. This repository provides the code and data for experiments on task identification, user identification, and user authentication using machine learning techniques.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Data Description](#data-description)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running the Pipeline](#running-the-pipeline)
- [Experiments](#experiments)
  - [Task Identification](#task-identification)
  - [User Identification](#user-identification)
  - [User Authentication](#user-authentication)
- [Results](#results)
- [Additional Notes](#additional-notes)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This repository contains the code and data for our research on integrating biometric user authentication into a teleoperation system using motion patterns from wearable inertial sensors. We focus on three main tasks:

- **Task Identification**: Recognizing the task being performed.
- **User Identification**: Identifying the user based on motion patterns.
- **User Authentication**: Verifying the user's identity to enhance security.

**Note**: This work is an extension of our previous research, where we developed the teleoperation system for controlling a robotic arm. The teleoperation system itself is not included in this repository.

## Features

- Machine learning models for classification and authentication.
- Configurable pipeline using a `config.yaml` file.
- Includes preprocessed feature data (`data.csv`) and labels (`labels.csv`).
- Raw inertial sensor data for 144 trials (`data/raw/` directory).
- Easy-to-use scripts for data loading, processing, and model execution.
- **Feature engineering code available for future exploration**.

## Data Description

### Raw Data

- Located in the `data/raw/` directory.
- Contains:
  - **144 CSV files** (`1.csv`, `2.csv`, ..., `144.csv`):
    - Each file is of size `1000 x 19`, representing:
      - Quaternion components (`x`, `y`, `z`, `w`) for each of the three arm joints (upper arm, forearm, hand).
      - Quaternion components and Cartesian positions (`x`, `y`, `z`) of the end effector.
    - Recorded at 50 Hz over a 20-second time window.
  - **`data.csv`**: The flattened dataset of size `144 x 361` (without headers).
  - **`labels.csv`**: The raw labels with 3 header columns: `task`, `subject`, `trial`.

### Processed Data

- Located in the `data/processed/` directory (created after running `data_processor.py`).
- Contains:
  - **`data.csv`**: A copy of the original `data.csv` from `data/raw/`, now with headers.
  - **`labels.csv`**: Labels with a single column depending on the configuration:
    - For **Task Identification**: Labels correspond to the `task`.
    - For **User Identification**: Labels correspond to the `subject`.
    - For **User Authentication**: Labels are binary, indicating `genuine` or `impostor` for each user.

**Note**: The feature extraction process (resulting in a flattened feature vector of size `1 x 361`) is performed using MATLAB code from another work. [Reference to be added].

## Prerequisites

- Python 3.12 or higher
- Required Python packages (see [Installation](#installation))
- Operating System: Linux

## Installation

### Clone the Repository

```bash
git clone https://github.com/eduardstan/auth-robotic-teleop.git
cd auth-robotic-teleop
```

### Set Up Virtual Environment (Optional but Recommended)

Create a virtual environment to manage dependencies:

```bash
python -m venv venv
```

Activate the virtual environment:

```bash
source venv/bin/activate
```

### Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Configuration

All configurations are managed through the `config.yaml` file. This allows you to specify parameters for data processing and model training, including:

- **Data Paths**:
  - `data_path`: Path to `data.csv` in `data/raw/`.
  - `labels_path`: Path to `labels.csv` in `data/raw/`.
  - `processed_data_path`: Path to save processed data (e.g., `data/processed/`).

- **CSV Header Settings**:
  - `data_has_header`: `False` (since `data.csv` in `data/raw/` does not have headers).
  - `labels_has_header`: `False` (since `labels.csv` in `data/raw/` does not have headers).

- **Target Variable and Task Type**:
  - `target_variable`: The label to use as the target for modeling.
    - For **Task Identification**: `'task'`
    - For **User Identification** and **User Authentication**: `'subject'`
  - `task_type`: The type of task to perform.
    - `'task_id'` for Task Identification.
    - `'user_id'` for User Identification.
    - `'user_auth'` for User Authentication.

**Example `config.yaml`**:

```yaml
# Data paths
data_path: 'data/raw/data.csv'
labels_path: 'data/raw/labels.csv'
processed_data_path: 'data/processed/'  # Path to save processed data

# CSV header settings
data_has_header: False               # True if data CSV has headers, False otherwise
labels_has_header: False             # True if labels CSV has headers, False otherwise

# Custom column names
data_columns: []                     # List of column names for data CSV (optional)
labels_columns: ['task', 'subject', 'trial']  # Column names for labels CSV

# Preprocessing settings
missing_value_strategy: 'mean'
categorical_columns: []              # No categorical columns to encode in features
scaling_method: 'none'               # Options: 'standard', 'minmax', 'none'

# Target variable settings
target_variable: 'task'              # 'task' for task identification, 'subject' for user tasks
task_type: 'task_id'                 # 'task_id', 'user_id', or 'user_auth'

# Cross-validation settings
cross_validation:
  task_identification:
    method: 'leave_one_subject_out'
  user_identification:
    method: 'repeated_k_fold'
    n_splits: 3
    n_repeats: 5
  user_authentication:
    method: 'custom'  # Custom method

# Feature Engineering Settings
feature_engineering:
  feature_selection_method: 'none'   # Options: 'univariate', 'mutual_info', 'model', 'none'
  k_best: 10                         # Number of features to select
  dimensionality_reduction_method: 'none'   # Options: 'pca', 'none'
  n_components: 5                    # Number of components to keep

# Model settings
models:
  - name: 'LogisticRegression'
    parameters:
      C: [0.1, 0.2, 0.3, 0.4, 0.5]
  - name: 'RandomForest'
    parameters:
      n_estimators: [50, 100, 150]
      max_depth: [null, 5, 10]
      criterion: ['gini', 'entropy']
  - name: 'SVM'
    parameters:
      C: [0.1, 0.5, 1.0]
      kernel: ['linear', 'rbf']

# Other settings
random_seed: 42
```

### Setting Up for Each Task

#### Task Identification

- **Configuration**:
  - `target_variable: 'task'`
  - `task_type: 'task_id'`

#### User Identification

- **Configuration**:
  - `target_variable: 'subject'`
  - `task_type: 'user_id'`

#### User Authentication

- **Configuration**:
  - `target_variable: 'subject'`
  - `task_type: 'user_auth'`

### Running the Pipeline

The processing pipeline consists of three main scripts:

1. **Data Loader**: Loads the raw data.

   ```bash
   python3 src/data_loader.py
   ```

   - Loads raw data from `data/raw/`.
   - Reads `data.csv` and `labels.csv` based on the paths specified in `config.yaml`.

2. **Data Processor**: Processes the data according to the selected task and extracts features.

   ```bash
   python3 src/data_processor.py
   ```

   - Processes the loaded data based on the configuration.
   - Adjusts labels and target variables according to `target_variable` and `task_type`.
   - Saves processed data to `data/processed/`.

3. **Models**: Trains and evaluates the machine learning models based on the configuration.

   ```bash
   python3 src/models.py
   ```

   - Trains and evaluates models using the processed data and labels.
   - Outputs results and performance metrics.

**Note**: Ensure that you run these scripts in the specified order for the pipeline to work correctly.

### Example Workflow

1. **Edit Configuration**:

   Open `config.yaml` and set the desired parameters for your experiment.

   - For example, to perform **Task Identification**:

     ```yaml
     target_variable: 'task'
     task_type: 'task_id'
     ```

2. **Run Data Loader**:

   ```bash
   python3 src/data_loader.py
   ```

   - Loads raw data and labels from `data/raw/`.

3. **Run Data Processor**:

   ```bash
   python3 src/data_processor.py
   ```

   - Processes data according to the task specified in `config.yaml`.
   - Adjusts labels based on `target_variable` and `task_type`.
   - Saves processed data and labels to `data/processed/`.

4. **Run Models**:

   ```bash
   python3 src/models.py
   ```

   - Trains and evaluates models based on the processed data and configuration.
   - Outputs results to the console or specified output files.

## Experiments

### Task Identification

- **Objective**: Recognize which of the three tasks is being performed based on motion data.
- **Configuration**:
  - `target_variable: 'task'`
  - `task_type: 'task_id'`
- **Procedure**:
  - Labels correspond to the `task`.
  - Uses Leave-One-Subject-Out (LOSO) cross-validation.
  - Evaluates models using macro F1-score.

### User Identification

- **Objective**: Identify the user performing the task based on motion patterns.
- **Configuration**:
  - `target_variable: 'subject'`
  - `task_type: 'user_id'`
- **Procedure**:
  - Labels correspond to the `subject`.
  - Uses Repeated Stratified K-Fold cross-validation.
  - Evaluates models using macro F1-score.

### User Authentication

- **Objective**: Verify the user's identity to enhance system security.
- **Configuration**:
  - `target_variable: 'subject'`
  - `task_type: 'user_auth'`
- **Procedure**:
  - Labels are processed to create binary labels (`genuine` vs. `impostor` for each user).
  - Applies a custom cross-validation strategy for each user.
  - **Model Settings**:
    - Set `class_weight: 'balanced'` in Logistic Regression parameters to handle class imbalance.
  - Evaluates models using accuracy, precision, recall, F1-score, and Equal Error Rate (EER).

## Results

- **Task Identification**:
  - Logistic Regression achieved a macro F1-score of approximately 85.31%.
- **User Identification**:
  - Logistic Regression achieved a macro F1-score of approximately 74.02%.
- **User Authentication**:
  - Logistic Regression demonstrated robust performance with an EER of approximately 8.89% when incorporating class weighting.

**Note**: Detailed results and analysis can be found in our accompanying paper.

## Additional Notes

- **Feature Engineering Code**:
  - Originally, we thought that 361 features might be excessive and explored feature engineering techniques to reduce the feature space.
  - The feature engineering code is available in the repository (e.g., `src/feature_engineering.py`), but we did not use these techniques in our experiments.
  - This feature engineering part could be included in future work to assess its impact on model performance.

- **Data Files**:
  - Ensure that the `data/raw/` directory contains the necessary data files before running the pipeline:
    - `1.csv`, `2.csv`, ..., `144.csv`
    - `data.csv` (flattened feature data without headers)
    - `labels.csv` (labels with columns: `task`, `subject`, `trial`)
  - The `data/processed/` directory will be created after running `data_processor.py`.

- **Dependencies**:
  - The `requirements.txt` file should include all the necessary Python packages. If you have not already, generate it by running `pip freeze > requirements.txt` in your development environment.

- **MATLAB Feature Extraction**:
  - The initial feature extraction from raw sensor data was performed using MATLAB code from another work. This step is not included in this repository.
  - For replication or further development, you may need to perform this step separately. Please refer to [Reference to be added] for details.

- **Data Privacy**:
  - The dataset does not contain personally identifiable information.
  - All user data is anonymized and labeled with subject IDs.

- **Issues and Support**:
  - If you encounter any issues or have questions about the repository, please open an issue on GitHub or contact us directly.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**:

   Click on the 'Fork' button at the top right corner of this page.

2. **Clone Your Fork**:

   ```bash
   git clone https://github.com/eduardstan/auth-robotic-teleop.git
   cd auth-robotic-teleop
   ```

3. **Create a New Branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Changes and Commit**:

   ```bash
   git add .
   git commit -m "Add your message here"
   ```

5. **Push to Your Fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Submit a Pull Request**:

   Go to the original repository and submit a pull request detailing your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact:

- **Ionel Eduard Stan**
- **Email**: ioneleduard.stan@unimib.it

---

**Note**: The feature extraction process (resulting in a flattened feature vector of size `1 x 361`) is performed using MATLAB code from another work. [Reference to be added].

---

**Acknowledgments**:

- Thanks to all participants in the experiments.
- This work builds upon our previous research on responsive teleoperation systems.

---

**Example Commands**

- **Running All Steps Sequentially**:

  ```bash
  # Step 1: Data Loading
  python3 src/data_loader.py

  # Step 2: Data Processing
  python3 src/data_processor.py

  # Step 3: Model Training and Evaluation
  python3 src/models.py
  ```

- **Changing Task in Configuration**:

  - To switch tasks, edit `config.yaml` accordingly.

    - **For Task Identification**:

      ```yaml
      target_variable: 'task'
      task_type: 'task_id'
      ```

    - **For User Identification**:

      ```yaml
      target_variable: 'subject'
      task_type: 'user_id'
      ```

    - **For User Authentication**:

      ```yaml
      target_variable: 'subject'
      task_type: 'user_auth'
      ```

---

**FAQ**

**Q**: Can I use this repository to perform my own experiments on similar data?

**A**: Yes! You can adapt the code and configuration to your own datasets, provided they follow a similar structure.

**Q**: How do I adjust the hyperparameters for the models?

**A**: You can adjust hyperparameters directly in the `config.yaml` file under the `models` section.
