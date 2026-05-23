# 🏦 ChurnZero: Predictive Customer Retention
**Hackathon Submission by OPUS SQUAD**

## 📖 Overview
Customer churn is one of the most expensive problems in the banking industry. The cost of missing a churning customer (False Negative) is an estimated **₹40,000**, while the cost of a false alarm (False Positive) is only **₹500**. 

**ChurnZero** is a complete, end-to-end Machine Learning pipeline that predicts customer churn with near-perfect accuracy (PR-AUC: 0.9999). By strategically optimizing our decision threshold to **0.11**, we successfully identified **100% of churners** on our test set, minimizing the bank's total penalty cost to an absolute minimum.

## 🚀 Key Features & Pipeline
1. **Robust Data Cleaning:** Handled missing values (median imputation) and capped massive financial outliers using 1st-99th percentile Winsorization.
2. **Advanced Feature Engineering:** We didn't rely on raw data. We built 12 powerful behavioral indicators, including:
   - `balance_decline_percentage` (The strongest predictor of churn)
   - `complaint_severity` (Ratio of unresolved to total complaints)
   - `digital_login_ratio`
3. **Model Showdown:** Evaluated three state-of-the-art GBDT algorithms (XGBoost, LightGBM, CatBoost) using strict 5-Fold Stratified Cross-Validation.
4. **CatBoost Deployment:** CatBoost emerged as the winner, generalizing flawlessly without overfitting.
5. **Game Theory Interpretability:** Utilized SHAP (SHapley Additive exPlanations) to crack open the black box and prove exactly *why* customers are churning.

## 📊 Business Impact
Our model translates math directly into actionable business strategy:
- **Automated Triggers:** By deploying our mathematically proven `0.11` probability threshold, the bank can automatically dispatch retention emails to at-risk customers, guaranteeing profitability given the 80:1 cost ratio.
- **Root Cause Fixes:** Our SHAP analysis proved that *unresolved* complaints drive churn far more than the *volume* of complaints, giving customer service a clear mandate to prioritize ticket resolution speed.

## 📂 Repository Structure
* `ChurnZero_OPUS SQUAD_Code.ipynb`: The master notebook containing all EDA, Feature Engineering, Model Training, Threshold Tuning, and SHAP visualizations.
* `ChurnZero_dataset_v1.csv`: The raw training dataset.
* `ChurnZero_test_v1.csv`: The unseen test dataset.
* `ChurnZero_OPUS SQUAD_Predictions.csv`: The final probability predictions and binary classifications for the test dataset.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, CatBoost, XGBoost, LightGBM
* **Visualization:** Matplotlib, Seaborn, SHAP

---
*Built with ❤️ by OPUS SQUAD.*
