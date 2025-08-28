# ⚙️ Project Setup & Execution Guide

This guide explains how to install dependencies and run each phase of the pipeline — from data collection to model deployment.  

---

## 🔄 Pipeline Overview  

```text
[Data Collection] → [Data Preparation] → [EDA] → [Training] → [Evaluation] → [Exporting] → [Serving]
       |                 |                 |        |            |              |            |
 raw_df.csv       cleaned_df.csv        reports/   artifacts/   metrics.json   final_model.pkl  Gradio UI
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
- **Robustness**:  
  - URL validation before scraping  
  - Error handling for missing or inconsistent fields  
  - Configurable depth (`pages` parameter).  

#### 📤 Output  
- Raw job data stored as a **pandas DataFrame**  
- Exported to:  data/raw_data.csv


### 2️⃣ Feature Engineering

**Note:** Several features were extracted from job descriptions using LLM APIs, including job industry. The raw data was then processed, manipulated, and saved as `data/raw_df.csv`. The published version is this processed dataset, and further feature engineering can be explored in the accompanying notebook.

#### ⚙️ Technical Details  
- Implemented in: `Notebooks/data_science_salary_feature_engineering_Public_data.ipynb`  
- **Input**: `data/raw_df.csv` (transformed scraped data)  
- **Output**: `data/df_feature_engineered.csv` (enriched dataset)
- **Steps**:
  1. Removing jobs posted more than 1 year ago
  2. Fixing the mixed columns for jobs missing company information (ownership, company_size, revenue)
  3. Transforming location and headquarter features to extract continent (Canada & US preserved as countries)
  4. Transforming salary and extracting the mean_salary

### ▶️ Usage  
Open and run the feature engineering notebook: 

```bash
notebooks/data_science_salary_feature_engineering_Public_data.ipynb
```


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
|-------|------------|-------------|---------|
| `-s`  | `--save_flag`  | Whether to save the preprocessed data and the preprocessing artifacts | 


- **Input:** df_feature_engineered.csv (from `config.files.feature_engineered`)
- **Output:** `preprocessed_train_df.csv` & `preprocessed_test_df.csv` (from `config.files.preprocessed_train` & `config.files.preprocessed_test` respectively) if the `-s` flag is used during the run

<!-- To Do: Add more details about the implemented steps -->

### 5️⃣ Model Building & Training
Run:

```bash
python train.py
```

- **Input**: `data/preprocessed_train_df.csv`  
- **Output**: Model artifacts (e.g., `.pkl` files)

### 6️⃣ Evaluation
Run:

bash
Copy
Edit
python evaluate.py
Input: Trained model + test split

Output: Metrics (metrics.json) in artifacts/.

### 7️⃣ Exporting
Run:

bash
Copy
Edit
python export.py
Output: Final trained model (final_model.pkl) in artifacts/.

###  Serving (Deployment)
Run:

bash
Copy
Edit
python -m src.serving
Output: Gradio app running at http://localhost:7860.

🛠 Notes
Configuration: config.yaml

Private keys: private_settings.yaml (not versioned)

Full explanations: see docs/ directory