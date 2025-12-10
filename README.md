<div align="center">

# 📊 **Customer Churn Prediction**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green.svg)](https://pandas.pydata.org/)

> **Data Analytics Internship Project @ Codec Technologies**

</div>


## 📝 **Project Overview**

Customer churn prediction is critical for telecom companies to identify customers likely to discontinue services. This project implements a complete data analytics workflow to predict customer churn using machine learning techniques, enabling proactive retention strategies and reducing revenue loss.

**Key Objectives:**
- Perform exploratory data analysis (EDA) to understand churn patterns
- Build predictive models to identify at-risk customers
- Evaluate model performance using industry-standard metrics
- Extract actionable insights for business decision-making

---

## 🔄 **Project Workflow**

### 1. **Data Preprocessing**
- **Missing Values Handling**: Identified and treated missing/inconsistent data
- **Feature Encoding**: Applied Label Encoding to categorical variables
- **Data Splitting**: 80-20 train-test split with stratification

### 2. **Exploratory Data Analysis**
- Analyzed churn distribution across customer segments
- Visualized relationships between features (Contract Type, Tenure, Monthly Charges)
- Identified key churn drivers through correlation analysis

### 3. **Model Development**
Built and evaluated multiple classification models:

| Model | Accuracy | Recall | ROC-AUC |
|-------|----------|--------|---------|
| **Logistic Regression** | 79.6% | 52.7% | 83.9% |
| **Random Forest** | 78.8% | 48.7% | 82.4% |
| **XGBoost** | 78.6% | 50.3% | 83.0% |

### 4. **Performance Metrics**
- **Precision**: Minimized false positives to optimize retention campaigns
- **Recall**: Maximized true positive rate to capture at-risk customers
- **ROC Curve Analysis**: Evaluated model discrimination capability
- **Confusion Matrix**: Analyzed prediction errors for model refinement

---

## 🛠️ **Technologies Used**

- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms and evaluation
- **XGBoost**: Gradient boosting implementation
- **Jupyter Notebook**: Interactive development environment

---

## 📁 **Project Structure**
```
Customer-Churn-Prediction/
│
├── Customer_Churn_Pred.ipynb    
├── mena_opportunity_tea_updated.csv    
├── README.md
├── LICENSE          
└── requirements.txt             
```

**Dataset Features:**
- Customer demographics (Gender, Senior Citizen, Partner, Dependents)
- Account information (Tenure, Contract Type, Payment Method)
- Service details (Phone Service, Internet Service, Streaming)
- Target variable: **Churn** (Yes/No)

---

## 🚀 **How to Run**

### **1. Clone the Repository**

```rust
git clone https://github.com/SumanBanerjee21/Customer_Churn_Prediction
cd Customer-Churn-Prediction
```

### **2. Install Dependencies**

```rust
pip install -r requirements.txt
```

### **3. Launch Jupyter Notebook**

```rust
jupyter notebook Customer_Churn_Pred.ipynb
```

### **4. Run All Cells**
Execute cells sequentially to reproduce:
- Data loading and preprocessing
- Exploratory visualizations
- Model training and evaluation
- Performance metrics and insights

---

## 💡 **Key Insights**

### **Churn Patterns**
1. **Contract Type**: Month-to-month contracts show significantly higher churn rates (42%) compared to yearly contracts (11%)
2. **Tenure**: Customers with <6 months tenure are 3x more likely to churn
3. **Payment Method**: Electronic check users exhibit elevated churn risk
4. **Service Usage**: Customers without online security or tech support are more vulnerable

### **Model Performance**
- **Logistic Regression** achieved the best balance between precision and recall
- **Feature Importance**: Contract type, tenure, and monthly charges are top predictors
- **ROC-AUC Score**: All models demonstrate good discrimination ability (>0.82)

### **Business Recommendations**
- Target retention campaigns toward month-to-month contract holders
- Offer incentives during the critical first 6 months
- Promote value-added services (security, support) to at-risk segments

---

## 🔮 **Future Improvements**

- [ ] **Hyperparameter Tuning**: Grid/Random search for optimal model configuration
- [ ] **Advanced Models**: Deep learning (Neural Networks), ensemble methods
- [ ] **Feature Engineering**: Create interaction features, polynomial terms
- [ ] **Class Imbalance**: SMOTE or weighted loss functions
- [ ] **Deployment**: Build Flask/Streamlit web app for real-time predictions
- [ ] **Monitoring**: Implement model drift detection and retraining pipeline

---

## 🎯 **Conclusion**

This project demonstrates a comprehensive approach to customer churn prediction, from exploratory analysis to model deployment considerations. By identifying high-risk customers with **83.9% ROC-AUC**, telecom companies can implement targeted retention strategies and reduce revenue loss.

**Skills Demonstrated:**
- End-to-end data analytics pipeline
- Machine learning model comparison and evaluation
- Business insight extraction from predictive models
- Professional documentation and reproducible research

---

## 👤 **About Me**

**Suman Banerjee**  
*Data Analytics Intern @ Codec Technologies*

Passionate about leveraging data science to solve real-world business problems. This project showcases my ability to translate raw data into actionable insights through rigorous analysis and machine learning techniques.

📧 [indsumanttt2002@gmail.com](mailto:indsumanttt2002@gmail.com) | 💼 [suman-banerjee-394822261](https://www.linkedin.com/in/suman-banerjee-394822261/) |

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


<div align="center">

### ⭐ If you find this project helpful, please consider giving it a star! ###

</div>