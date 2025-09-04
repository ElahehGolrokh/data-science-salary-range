# ⚙️ Project Setup & Execution Guide

This guide explains how to install dependencies and run each phase of the pipeline — from data collection to model deployment.  

---

## 🔄 Pipeline Overview  

```text
[Data Collection] → [Feature Engineering] → [EDA] → [Preprocessing] → [Training] → [Evaluation] → [Exporting] → [Serving]
       |                   |                 |          |               |              |              |            |
 raw_df.csv      df_feature_engineered.csv   notebook   preprocessed_*   artifacts/    logs/          HF Hub       Gradio UI
 ```

---

## 📦 Installation
1. Clone the repository

```bash
git clone git@github.com:ElahehGolrokh/data-science-salary-range.git
cd data-science-salary-range
```

2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```
---

## 📊 Pipeline Execution

### 1️⃣ Data Collection

**Note:** Since the source website cannot be disclosed due to privacy policies, you cannot run this phase. This section is included **for demonstration purposes only**.


**Arguments for `crawl.py`:**  

| Short | Long        | Description | Default |
|-------|------------|-------------|---------|
| `-p`  | `--pages`  | Set number of pages to scrape | From `config.pages` |
| `-u`  | `--url`    | Set the base URL for scraping | From `config.url` |
| `-fp` | `--file_path` | Set the file path for saving data | From `config.files.raw_data` |
| `-dp` | `--dir_path`  | Set the directory path for saving data | From `config.dirs.data` |


**Example:**  
```bash
python crawl.py --pages 10 --url "https://example.com/jobs" --file_path "raw_df.csv" --dir_path "data"
```
#### 📌 Scope  
- **Target**: Company job boards & listings pages  
- **Extracted Features**:
  - url of the job post (for further investigation if needed)
  - Job title  
  - Company name & description  
  - Location & work setup (remote / hybrid / on-site)  
  - Posting date  
  - Salary (if available)  
  - Seniority level & role indicators  
  - Company details (size, revenue, headquarters, ownership)  
  - Full job description  

#### ⚙️ Technical Details  
- **Framework**: [Selenium](https://www.selenium.dev/) for browser automation  
- **Browsers Supported**: Firefox
- **Dynamic Content**: Handled with explicit waits, scrolling, and pop-up management   

**Output:** 
- Raw job data stored in `data/raw_data.csv`


### 2️⃣ Feature Engineering

**Note:** Several features were extracted from job descriptions using LLM APIs, including job industry. The raw data was then processed, manipulated, and saved as `data/raw_df.csv`. The published version is this processed dataset, and further feature engineering can be explored in this notebook: 

```bash
notebooks/data_science_salary_feature_engineering_Public_data.ipynb
```

#### ⚙️ Technical Details  
- Implemented in: `Notebooks/data_science_salary_feature_engineering_Public_data.ipynb`  
- **Input**: `data/raw_df.csv` (transformed scraped data)  
- **Output**: `data/df_feature_engineered.csv` (enriched dataset)
- **Steps**:
  1. Removing jobs posted more than 1 year ago
  2. Fixing the mixed columns for jobs missing company information (ownership, company_size, revenue)
  3. Transforming location and headquarter features to extract continent (Canada & US preserved as countries)
  4. Transforming salary and extracting the mean_salary


### 3️⃣ Exploratory Data Analysis (EDA)

To analyze trends in job roles, seniority levels, industries, required skills, and compensation.  

- Implemented in: `notebooks/data_science_salary_EDA.ipynb`  
- **Input**: `data/df_feature_engineered.csv`  


### 4️⃣ Preprocessing

To prepare the feature-engineered dataset for machine learning by handling missing values, encoding categorical features, processing skill stes, and scaling numerical variables.  

Run:

```bash
python prepare.py -s
```

**Arguments for `prepare.py`:**

| Short | Long        | Description |
|-------|-------------|-------------|
| `-s`  | `--save_flag`  | Whether to save the preprocessed data and the preprocessing artifacts | 


- **Input:** df_feature_engineered.csv (from `config.files.feature_engineered`)
- **Output:** `preprocessed_train_df.csv` & `preprocessed_test_df.csv` (from `config.files.preprocessed_train` & `config.files.preprocessed_test` respectively) if the `-s` flag is used during the run

#### ⚙️ Technical Details  

1. **Handling Missing Values**  
   - Imputed missing categorical features with mode & numerical features with median of the train_df

2. **Encoding Categorical Features**  
   - `seniority_level` converted to an **ordinal variable** (e.g., Intern < Junior < Mid < Senior < Lead) to reflect hierarchy  
   - Other categorical variables encoded with one-hot encoding  

3. **Skill Processing**  
   - Extracted individual skills from free-text job descriptions  
   - Binarized presence/absence of key skills  

4. **Scaling**  
   - Numerical features standardized using scalers fit **only on `train_df`** (`src_df`)  
   - The same scalers were applied to the test set to avoid data leakage  

5. **Monotonic Constraints (Attempted)**  
   - Implemented monotonic constraints to enforce logical salary relationships with certain features (e.g., seniority, experience)  
   - Constraints did not work effectively in practice and were not used in the final pipeline  

**Reference:**  
- Preprocessing artifacts (scalers, encoders, selected features) are saved for reproducibility and consistent application on unseen data. 

### 5️⃣ Model Building & Training
Train the machine learning models on the preprocessed dataset. You can optionally implement feature selection, compare multiple models, and fine-tune the chosen model.  

Run:

```bash
python train.py -t -cm -mn RandomForest -fn
```

**Arguments for `train.py`:**

| Short | Long        | Description | Default |
|-------|------------|-------------|---------|
| `-cm`  | `--compare_models`  | Enable model comparison in the pipeline | False |
| `-t`  | `--train`    | Enable model training in the pipeline | False` |
| `-mn` | `--model_name` | Optional name of the model to train (e.g., RandomForest, XGB, etc.) | RandomForest |
| `-fn` | `--fine_tune`  | Enable fine-tuning for the model. Just implemented for RandomForest | False |


#### ⚙️ Technical Details 

1. Model Comparison

    - Evaluates multiple candidate models (Linear Regression, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost) using cross-validatio.

    - Selects the best model based on mean CV score and saves the best model's name in the `artifacts/best_model_name.txt` file.

2. Feature Selection

    - Uses Recursive Feature Elimination (RFE) to find the optimal subset of features.
    - The number of features to keep can either be passed to the `ModelSelector` object using the `feature_count_` parameter, or left empty to allow the object to determine the optimal number of features automatically.

3. Fine-Tuning

    - Hyperparameter tuning performed on the selected model (Just implemented for Random Forest, since it was the best performing model on our dataset).
    - If --save_flag is enabled, saves best parameters for reproducibility.

4. Final Training

    - Trains the chosen model on the entire training set using the selected features and best hyperparameters.
    - If you pass some models other than Random Forest through model_name argument, then this model is trained using default hyperparameters.
    - Exports the trained model and artifacts to artifacts/ --save_flag is enabled. By default this flag is set to True in `config.training.save_flag`.

 
- **Input**: `preprocessed_train_df.csv` 
- **Output**: Trained model artifacts in the `artifacts/` directory, including best hyperparameters, selected features, final trained model (`final_model.pkl`), etc.

### 6️⃣ Evaluation
Evaluates the trained model on the test set and generates comprehensive performance metrics, reports, predictions, and plots.

Run:

```bash
python evaluate.py -s -np "rf_run1"
```

**Arguments for `evaluate.py`:**

| Short | Long        | Description |
|-------|------------|-------------|
| `-s`  | `--save`  | Whether to save evaluation results |
| `-np`  | `--name_prefix`    | Optional prefix for saved evaluation result files|

- **Input:**
    - Trained model `final_model.pkl` from `artifacts/` if model = None
    - Preprocessed test set `preprocessed_test_df.csv`

- **Output (saved in logs/):** 
    - Evaluation report: <name_prefix>_evaluation_report.txt
    - Metrics: <name_prefix>_evaluation_metrics.csv
    - Predictions: <name_prefix>_predictions.csv
    - Evaluation plots: <name_prefix>_evaluation_plots.png
    - Feature importance plots: <name_prefix>_feature_importance_plot.png

#### ⚙️ Technical Details

1. Main Metrics Computed
    - R² Score
    - Mean Absolute Error (MAE)
    - Root Mean Square Error (RMSE)
    - Mean Absolute Percentage Error (MAPE)
    - MAE & RMSE as % of mean salary
    - Statistical summary: mean actual, mean predicted, residual mean & std

2. Reporting
    - Generates a human-readable report with interpretations based on R² and MAPE thresholds.

3. Plots
    - Evaluation plots (e.g., residuals, predicted vs actual)
    - Feature importance plots (for tree-based models)

4. Saving Results
    - All outputs are automatically saved with the specified name_prefix if    `--save` is enabled.

### 7️⃣ Exporting
Upload the trained model artifacts (model weights, encoders, scalers, etc.) to a Hugging Face model repository.

Run:

```bash
python export.py -ri "username/repo-name" --api_token "hf_xxxxx"
```

**Arguments for `export.py`:**

| Short | Long        | Description |
|-------|------------|-------------|
| `-ri`  | `--repo_id`  | Hugging Face repository ID where the model artifacts will be uploaded|
| -  | `--api_token`    | Hugging Face API token for authentication|

- **Input:** Local artifacts stored in `artifacts/` directory, including:
    - Final trained model (`final_model.pkl`)
    - Feature selection metadata (`selected_features.pkl`)
    - Preprocessing artifacts (e.g., scalers, encoders)

- **Output:** Artifacts uploaded to the specified Hugging Face repo. Returns a list of URLs for each uploaded file for easy reference or download. Each artifact will be uploaded to the repo with the same filename.


### 8️⃣ Serving (Deployment)
Run:

```bash
python app.py
```

**Output:** Gradio app running at http://localhost:7860.

---

## 📊 Inferential Statistics

In addition to exploratory analysis and predictive modeling, we applied inferential statistics to validate insights about salaries.  
This analysis is implemented in [`notebooks/data_science_salary_Inferential_statistics.ipynb`](../notebooks/data_science_salary_Inferential_statistics.ipynb).

### Goals
We focused on the following hypothesis tests and confidence interval estimations:
1. **Population Mean Test**: Based on salary aggregators like Indeed and Glassdoor, the mean salary for a Data Scientist in the U.S. is around $128,000 as of May 2024. Does the data supports this claimed value?  
2. **seniority level effect (t-test)**: Do senior data scientists earn significantly higher mean salaries compared to junior/mid-level roles in the US?  
3. **Location Effect**: Does working remotely or in a hybrid setup lead to different mean salaries compared to on-site roles?  

### Implementation
- Statistical tests were performed using `scipy.stats`.  
- Confidence intervals were computed at a 95% confidence level. 

### Input & Output
- **Input**: Feature engineered dataset (`data/df_feature_engineered.csv`).  
- **Output**: Test statistics, p-values, and confidence intervals.  

---

## 🛠 Notes
- Configuration: `config.yaml`
- Private keys: `private_settings.yaml`
- Full explanations: see `docs/` directory
