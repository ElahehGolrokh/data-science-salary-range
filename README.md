# 🚀 Data Science Salary Estimation  

This project builds a **data pipeline for estimating salaries for data science jobs** by scraping, cleaning, and enriching job postings from company websites and job boards. The end goal is to create a structured repository that can be used for **analytics, salary prediction, and trend analysis**.  

> ⚠️ **Note on Data Privacy**  
> To respect the privacy policy of the source platforms, we do **not disclose the exact websites** being scraped. Furthermore, the extracted data is **manipulated, cleaned, and transformed** before being published here to avoid sharing raw proprietary content.  The file stored at data/raw_df.csv is therefore a transformed version of the initial scraped data to respect the source privacy policy. 

---

## 📂 Pipeline Overview  

1. **Scraping (✅ Completed)** – Automated collection of raw job postings with Selenium  
2. **Data Cleaning & Enrichment (✅ Completed)** – Extracting skills, experience, industries, and refining numerical & categorical features  
3. **Exploratory Data Analysis (✅ Completed)** – Identifying trends in roles, seniority, required skills, and compensation 
4. **Preprocessing (🔄 In Progress)**
5. **Modeling (🔄 In Progress)** – Predicting salaries and career paths based on job descriptions 
6. **Deploying (🤖 Planned)**
7. **Inferential Statistics (📐 Planned)** – Deriving **confidence intervals** for data science roles  

---

## 🔹 1. Scraping Phase  

### 🎯 Goal  
To collect structured job posting data at scale for further analysis.  

### 📌 Scope  
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

### ⚙️ Technical Details  
- **Framework**: [Selenium](https://www.selenium.dev/) for browser automation  
- **Browsers Supported**: Firefox
- **Dynamic Content**: Handled with explicit waits, scrolling, and pop-up management  
- **Robustness**:  
  - URL validation before scraping  
  - Error handling for missing or inconsistent fields  
  - Configurable depth (`pages` parameter).  

### 📤 Output  
- Raw job data stored as a **pandas DataFrame**  
- Exported to:  data/raw_data.csv


### ▶️ Quick Start  
Run the crawler directly with:  

```bash
python crawl.py --pages 5 --url "https://example.com/jobs"
```

Arguments:

- --pages → Number of listing pages to scrape

- --url → Starting job board URL

## 🔹 2. Feature Engineering Phase  

### 🎯 Goal  
To enrich the raw scraped data with structured features suitable for analysis and EDA.  

### ⚙️ Technical Details  
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
Notebooks/data_science_salary_feature_engineering_Public_data.ipynb
```

## 🔹 3. Exploratory Data Analysis (EDA)  

### 🎯 Goal  
To analyze trends in job roles, seniority levels, industries, required skills, and compensation.  

### ⚙️ Technical Details  
- Implemented in: `Notebooks/data_science_salary_EDA.ipynb`  
- **Input**: `data/df_feature_engineered.csv`  

## 🔹 4. Preprocessing

### 🎯 Goal  
To prepare the feature-engineered dataset for machine learning by handling missing values, encoding categorical features, processing skill stes, and scaling numerical variables.  

### ⚙️ Technical Details  
- Implemented in: `prepare.py`  
- **Input**: `data/df_feature_engineered.csv`  
- **Output**:            
  - `data/preprocessed_train_df.csv` 
  - `data/preprocessed_test_df.csv` 

### ▶️ Quick Start  
Run the preprocessing step directly with:  

```bash
python prepare.py
```

Arguments:

🔄 In progress


## 🔹 5. Model Building and Evaluation 

### 🎯 Goal  
To build predictive models for estimating salaries and evaluate their performance.  

### ⚙️ Technical Details  
- Implemented in: `train.py`  
- **Input**: `data/preprocessed_train_df.csv`  
- **Output**: Model artifacts (e.g., `.pkl` files)

## 📌 Next Steps  
 
- Evaluation
- Deploying
- Inferential analysis for confidence intervals

---

## ⚖️ License  

This project is licensed under the **MIT License** – you are free to use, modify, and distribute it.  

---

## ✨ Acknowledgments  

- [Selenium](https://www.selenium.dev/) for automated web scraping  
- [pandas](https://pandas.pydata.org/) for data handling  
- Future integrations with **LLMs** for advanced information extraction  

