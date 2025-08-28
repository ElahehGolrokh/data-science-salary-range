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
git clone https://github.com/yourusername/ds-salary-estimation.git
cd ds-salary-estimation
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

| Argument   | Description | Default |
|------------|-------------|---------|
| `--pages` | Number of pages to scrape | From `config.pages` |
| `--url` | Base URL to start crawling | From `config.url` |
| `--file_path` | File path to save raw data | From `config.files.raw_data` |
| `--dir_path` | Directory path for data storage | From `config.dirs.data` |

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


### 2️⃣ Data Preparation
Run:

bash
Copy
Edit
python prepare.py
Input: data/raw_df.csv

Output: data/cleaned_df.csv (processed dataset).

3️⃣ Exploratory Data Analysis (EDA)
Run:

bash
Copy
Edit
python -m src.evaluation --eda
Input: data/cleaned_df.csv

Output: Plots & stats in reports/eda/.

4️⃣ Training
Run:

bash
Copy
Edit
python train.py
Input: data/cleaned_df.csv

Output: CV scores, selected features in artifacts/feature_selection.json.

5️⃣ Evaluation
Run:

bash
Copy
Edit
python evaluate.py
Input: Trained model + test split

Output: Metrics (metrics.json) in artifacts/.

6️⃣ Exporting
Run:

bash
Copy
Edit
python export.py
Output: Final trained model (final_model.pkl) in artifacts/.

7️⃣ Serving (Deployment)
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