# 🚀 Data Science Salary Estimation  

## 📌 Overview
In this project, I designed and implemented a **complete end-to-end machine learning pipeline from scratch to estimate data scientist salaries based on input features.** The pipeline covers **data scraping, data cleaning & EDA, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, deployment, and building a client-facing API**.  
Although predicting salaries precisely is inherently noisy, this work demonstrates **technical depth and practical workflow** in a portfolio-ready format. 

**Goal:** Showcase **end-to-end data science skills** in a clean, reproducible pipeline.

---
## 💻 Code and Resources Used
- **Python Version**: 3.7  
- **Packages**: pandas, numpy, scikit-learn, matplotlib, seaborn, selenium, pickle, gradio

---

## 🔍 Data Pipeline
- **Crawling**: Over 1000 job postings scraped using Python and Selenium.  
- **Data Cleaning**: Extracted skill sets, experience, company info, industry, and location. Data types were fixed and string manipulation applied to extract purely numerical features. Some features were **derived from job descriptions using LLM APIs**, which are not published here **to respect the source website’s privacy policy**.
- **Preprocessing**:  
  - Dropped non-informative features 
  - Ordinal encoding for `seniority_level` (`junior < midlevel < senior < lead`)  
  - Handling missing values, categorical encoding, numerical transformations

> ⚠️ **Note on Data Privacy**  
> To respect the privacy policy of the source platforms, the **exact websites are not disclosed**. Data is also **manipulated, cleaned, and transformed** before being published to avoid sharing raw proprietary content. The file at `data/raw_df.csv` is therefore a **processed version** of the original scraped data.

---

## 🧠 Modeling
- Built a custom pipeline for **feature selection** and model comparison using **RFE + cross-validation**.  
- Implemented:  
  - Model comparison: Linear Regression, Ridge, Lasso, **Random Forest (best)**, Gradient Boosting, XGBoost  
  - Selection of best model based on mean CV score  
  - Storing selected features and final model  
- Feature subsets tested: 15, 20, 25, 30, 35, 40, 57 features  
- **Hyperparameter tuning** applied to Random Forest  
- Final Random Forest Regressor trained with best parameters and exported  
- **Built a client-facing API using Gradio** for interactive model inference 

---

## 📊 Evaluation Results
| Metric | Best RF Model |
|--------|---------------|
| R² | ~0.57 | Explains about half of the variance → decent given noisy features, but not strong enough for real-world salary prediction. |
| MAE | ~$30k | (~25% of average salary): Too large for practical salary modeling. |
| RMSE | ~$42k | Indicates large average error magnitude, reinforcing limited practical accuracy. |
| MAPE | ~42% | Very high → poor for low-salary roles (e.g., $60K predicted as $90K = 50% error). |


**Interpretation**: model captures general salary trends but not precise values; salary prediction is inherently noisy.

> ⚠️ **Note on Model Improvement Attempts**  
> Early performance was poor. Several improvements were tested, with the most effective being encoding `seniority_level` as an ordinal rather than categorical feature. This ensured the model **never predicted lower salaries for higher seniority levels (all else equal)**, solving the “senior ≈ junior” salary collapse observed initially.  
> More details on these steps are available in the `docs/` directory.

---

## ⚠️ Limitations
The main bottleneck is not preprocessing, feature selection, or model choice—all of which were explored. Instead, the issue lies in the **signal itself**: the most informative features explaining salary variance are either **missing or too noisy**.  
- Small dataset (~750 training samples, ~180 test samples)  
- Missing strong explanatory signals (e.g., job function, company reputation, detailed skills)  
- Noisy features: job postings often omit or exaggerate details  
- MAPE > 40% → predictions not business-grade accurate  

---
## 🚀 Next Steps
- Reframe as **classification** (low/mid/high salary buckets)  
- Gather **more data** across industries, regions, roles  
- Explore **NLP embeddings** for job descriptions instead of manual features  
- Try **stacking/ensembling** models for robustness  

---

## 🎯 Takeaway
This project demonstrates **end-to-end ML workflow** and practical data science skills, including deployment through a client-facing Gradio app — suitable for a portfolio showcase.
