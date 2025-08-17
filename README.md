# 🚀 Data Science Salary Estimation  

This project builds a **data pipeline for estimating salaries for data science jobs** by scraping, cleaning, and enriching job postings from company websites and job boards. The end goal is to create a structured repository that can be used for **analytics, salary prediction, and trend analysis**.  

> ⚠️ **Note on Data Privacy**  
> To respect the privacy policy of the source platforms, we do **not disclose the exact websites** being scraped. Furthermore, the extracted data is **manipulated, cleaned, and transformed** before being published here to avoid sharing raw proprietary content.  The file stored at data/raw_df.csv is therefore a transformed version of the initial scraped data to respect the source privacy policy. 

---

## 📂 Pipeline Overview  

1. **Scraping (✅ Completed)** – Automated collection of raw job postings with Selenium  
2. **Data Cleaning & Enrichment (🔄 In Progress)** – Extracting skills, experience, industries, and refining numerical & categorical features  
3. **Exploratory Data Analysis (📊 Upcoming)** – Identifying trends in roles, seniority, required skills, and compensation  
4. **Modeling (🤖 Planned)** – Predicting salaries and career paths based on job descriptions  
5. **Inferential Statistics (📐 Planned)** – Deriving **confidence intervals** for data science roles  

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
-  **Steps**:
  - Removing jobs posted more than 1 year ago  
  - Fixing the mixed columns for jobs in which the company description has not the entire information, means ownership, company_size and revenue
  - Transforming location and headquarter features to extract and preserve only the continent (Canada & US
  which have the most number of job posts are preserved as the country)
  - Transforming salary and extract the mean_salary 

### ▶️ Usage  
Open and run the feature engineering notebook:  
```bash
Notebooks/data_science_salary_feature_engineering_Public_data.ipynb
```

## 📌 Next Steps  
 
- EDA
- Preprocessing for machine learning readiness
- Model training for salary prediction
- Inferential analysis for confidence intervals

---

## ⚖️ License  

This project is licensed under the **MIT License** – you are free to use, modify, and distribute it.  

---

## ✨ Acknowledgments  

- [Selenium](https://www.selenium.dev/) for automated web scraping  
- [pandas](https://pandas.pydata.org/) for data handling  
- Future integrations with **LLMs** for advanced information extraction  

