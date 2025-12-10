```markdown
# 📊 Customer Churn Prediction – Telecom

[![Python](https://img.shields.io/badge/Python-3.x-blue)](#)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange)](#)
[![Pandas](https://img.shields.io/badge/Library-Pandas-green)](#)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--learn-yellow)](#)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-lightgrey)](#)

Predicting customer churn is critical for telecom companies because retaining existing customers is far more cost-effective than acquiring new ones.  
This project performs end-to-end data analytics on a telecom dataset to understand churn behavior and build machine learning models that can predict which customers are likely to leave. [file:1]

---

## 📌 Project Overview

- **Internship**: Data Analytics Intern @ **Codec Technologies**  
- **Objective**: Predict whether a telecom customer will churn (leave) based on their demographics, services, billing, and contract details. [file:1]  
- **Pipeline**:
  - Exploratory Data Analysis (EDA)
  - Data cleaning & preprocessing
  - Feature encoding & scaling
  - Model building (Logistic Regression, Random Forest, XGBoost)
  - Model evaluation using classification metrics & ROC curves. [file:1]

---

## 🧩 Project Workflow

### 1. Data Understanding & Cleaning
- Loaded telecom customer data (≈7043 rows × 21 columns) including gender, senior citizen, tenure, services, payment method, charges, and churn label. [file:1]  
- Verified column types, ensured no missing values at frame level, and dropped non-informative identifiers like `customerID` for modeling. [file:1]  

### 2. Handling Missing & Invalid Values
- Checked `df.isnull().sum()` to confirm absence of true nulls in the raw dataset. [file:1]  
- Cleaned and converted `TotalCharges` from object to numeric, handling problematic entries (e.g., blanks) before modeling. [file:1]  

### 3. Encoding & Transformation
- **Target**: `Churn` converted from `"Yes"/"No"` to binary (1 = churn, 0 = no churn) using `LabelEncoder`. [file:1]  
- **Features**: All categorical columns (e.g., `Contract`, `InternetService`, `PaymentMethod`, etc.) encoded via label encoding to obtain a fully numeric feature matrix. [file:1]  
- For Logistic Regression, a `Pipeline(StandardScaler + LogisticRegression)` was used to scale features and improve convergence. [file:1]  

### 4. Train–Test Split
- Split the encoded data into:
  - **Train**: 80%  
  - **Test**: 20%  
- Used `stratify=y` to preserve the original churn proportion (≈26% churn vs 74% non-churn). [file:1]  

### 5. Models Used
Implemented multiple models to compare performance: [file:1]  

| Model                | Notes                                      |
|----------------------|--------------------------------------------|
| Logistic Regression  | With StandardScaler pipeline               |
| Random Forest        | 200 trees, `max_depth=None`, `n_jobs=-1`   |
| XGBoost Classifier   | 300 estimators, depth 4, subsampling tuned |

### 6. Evaluation Metrics
For each model, the following metrics were computed on the test set: [file:1]  

- **Accuracy**  
- **Recall** (churn class, to capture more churners)  
- **ROC-AUC**  
- **Confusion matrix** (derived from predictions for class-wise performance)  

Example reported scores: [file:1]  

| Model               | Accuracy | Recall (Churn) | ROC-AUC |
|---------------------|----------|----------------|---------|
| Logistic Regression | ~0.80    | ~0.53          | ~0.84   |
| Random Forest       | ~0.79    | ~0.49          | ~0.82   |
| XGBoost             | ~0.79    | ~0.50          | ~0.83   |

ROC curves were plotted for all three models on a single chart to visually compare discrimination power. [file:1]

---

## 📊 Key EDA Insights

From the exploratory data analysis on churn distribution and key features: [file:1]  

- Churn is **imbalanced**, with significantly more “No” than “Yes” customers (≈5174 vs 1869). [file:1]  
- Customers on **month-to-month contracts** show notably higher churn compared to one-year or two-year contracts. [file:1]  
- Higher **MonthlyCharges** and lower **tenure** are associated with higher churn likelihood, indicating new, high-paying customers are at greater risk. [file:1]  
- Add-on services such as **OnlineSecurity**, **TechSupport**, and longer **Contract** terms appear to reduce churn. [file:1]  

---

## 🧠 Model Performance & Insights

- **Best Overall Model**:  
  - Logistic Regression with scaling achieved the strongest balance between accuracy (~80%), recall (~0.53), and ROC-AUC (~0.84). [file:1]  
- **Tree-based Models**:
  - Random Forest and XGBoost showed similar performance, with slightly lower recall but competitive ROC-AUC (~0.82–0.83). [file:1]  
- **Key Predictors** (from model behavior & EDA):  
  - Contract type (month-to-month vs longer-term)  
  - Tenure (months with company)  
  - MonthlyCharges & TotalCharges  
  - Internet service type and security/backup support features. [file:1]  

These insights can directly inform retention strategies—such as targeted offers to new, high-charge, month-to-month customers.

---

## 🛠️ Technologies Used

- **Language**: Python 3.x [file:1]  
- **Environment**: Jupyter Notebook (`Customer_Churn_Pred.ipynb`) [file:1]  
- **Libraries**:
  - Data Handling: `pandas`, `numpy` [file:1]  
  - Visualization: `matplotlib`, `seaborn` [file:1]  
  - Machine Learning: `scikit-learn` (Logistic Regression, RandomForest), `xgboost` [file:1]  

---

## 📁 Folder Structure

A suggested project structure for this repository:

```
Customer-Churn-Prediction/
├─ data/
│  ├─ telecom_churn.csv          # Raw dataset (customerID, demographics, services, billing, Churn)
│  └─ README_data.md             # Notes on source, columns, and preprocessing assumptions
├─ notebooks/
│  └─ Customer_Churn_Pred.ipynb  # Main analysis & modeling notebook
├─ models/
│  ├─ best_logreg_model.pkl      # (Optional) Serialized best model
│  └─ scaler.pkl                 # (Optional) Fitted scaler for deployment
├─ outputs/
│  ├─ eda_plots/                 # Churn distribution, contract vs churn, etc.
│  └─ metrics_report.md          # Saved tables/metrics
└─ README.md
```

**Dataset expectations** (`telecom_churn.csv`): [file:1]  
- One row per customer  
- Columns such as:
  - `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`  
  - Service flags: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`  
  - Contract/billing: `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`  
  - Target: `Churn` (Yes/No)  

---

## ▶️ How to Run This Project

1. **Clone the repository**
   ```
   git clone https://github.com/<your-username>/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Create and activate a virtual environment (optional but recommended)**
   ```
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
   Minimal packages:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```

4. **Place the dataset**
   - Save your telecom churn CSV file (e.g., `telecom_churn.csv`) in the `data/` folder.  
   - Update the CSV path variable inside `Customer_Churn_Pred.ipynb` if necessary. [file:1]  

5. **Launch the notebook**
   ```
   jupyter notebook
   ```
   - Open `notebooks/Customer_Churn_Pred.ipynb`.  
   - Run all cells in order (EDA → preprocessing → model training → evaluation). [file:1]  

---

## 🚀 Future Improvements

Some enhancements that can be added on top of this work:

- **Hyperparameter Tuning**  
  - Use `GridSearchCV` / `RandomizedSearchCV` / Bayesian optimization to fine-tune Logistic Regression, Random Forest, and XGBoost for improved recall and ROC-AUC. [file:1]  

- **Advanced Models**
  - Try Gradient Boosting, LightGBM, or CatBoost to better capture complex non-linear relationships.  

- **Cost-Sensitive Learning**
  - Introduce class weights or custom loss functions to penalize misclassification of churners more heavily.  

- **Deployment**
  - Wrap the best model in a simple **Flask** or **Streamlit** app to make churn predictions interactively from a web UI.  

- **Explainability**
  - Use SHAP/LIME to generate model explanations and feature-attribution insights for business stakeholders.  

---

## ✅ Conclusion

This Customer Churn Prediction project demonstrates a complete analytics and machine learning pipeline—from EDA and feature engineering to model training, evaluation, and insight generation—on a real-world telecom dataset. [file:1]  
As a Data Analytics Intern at **Codec Technologies**, this work highlights strong skills in Python-based data analysis, model experimentation, and the ability to translate data patterns into actionable business recommendations for reducing churn. [file:1]  

---

## 👤 About Me

**Suman Banerjee**  
_Data Analytics Intern @ Codec Technologies_  

- Passionate about data-driven decision making, customer analytics, and predictive modeling.  
- Open to feedback, collaboration, and new ideas to further improve churn prediction systems.  

```

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/89258049/b6288725-955f-4e2a-bbd6-004cff6d2cff/Customer_Churn_Pred.ipynb)
