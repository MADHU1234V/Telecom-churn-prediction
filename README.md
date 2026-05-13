# Telecom Customer Churn Prediction

A machine learning project to predict customer churn in the telecom industry, generate churn risk scores, and enable data-driven retention strategies.

---

## Problem Statement

Customer churn is one of the most critical business challenges in the telecom industry. Losing a customer costs significantly more than retaining one. This project builds a predictive system that identifies at-risk customers before they leave, enabling proactive and targeted retention campaigns.

---

## Objectives

- Identify the key factors that drive customer churn
- Introduce a binary CHURN_FLAG variable (1 = churned, 0 = retained)
- Build and compare multiple ML classification models
- Generate a Churn Risk Score (0–1) for every customer
- Segment customers into actionable risk tiers: Very Low, Low, Medium, High

---

## Project Structure

```
telecom-churn-prediction/
│
|-- Telecom_Churn_Prediction.ipynb          # Main notebook
|-- telecom_churn_data1.csv                 # Input dataset
|-- customers_with_churn_predictions.csv    # Output: predictions
|--customers_with_churn_risk_scores.csv    # Output: risk scores & segments
|-- README.md
```

---

## Dataset Overview

The dataset contains ~4,617 customer records with a churn rate of ~14.2%. Features include:

- Account Length: Duration of customer relationship in days
- International Plan: Whether the customer has an international plan (Yes/No)
- VMail Plan: Whether the customer has a voicemail plan (Yes/No)
- VMail Message: Number of voicemail messages
- Day/Eve/Night Mins: Call duration by time of day
- Day/Eve/Night Calls: Number of calls by time of day
- Day/Eve/Night Charge: Call charges by time of day
- International Mins/Calls/Charge: International usage metrics
- CustServ Calls: Number of customer service calls made
- State, Area Code, Phone: Geographic and identifier fields
- Churn: Original churn indicator (True/False)

---

## Workflow

### 1. Data Cleaning
- No missing values or duplicate records found
- Outliers handled using IQR-based capping

### 2. Feature Engineering
- Created CHURN_FLAG from the original Churn column
- Engineered aggregate features: Total_Mins, Total_Calls, Total_Charge
- Encoded binary plan columns (International Plan, VMail Plan)
- Dropped redundant columns: Phone, Area Code, State, individual charge columns

### 3. Exploratory Data Analysis
- ~85.8% of customers did not churn; 14.2% churned
- Customers with high CustServ Calls (3+) showed churn rates exceeding 50%
- Newer customers with shorter account length churn at higher rates
- Day-time usage and charges are significantly higher for churners
- International plan holders had slightly higher churn tendencies

### 4. Model Building

All models were built using scikit-learn Pipelines with StandardScaler preprocessing. The following classifiers were trained and compared:

- Logistic Regression (baseline, class_weight='balanced')
- Decision Tree (criterion='gini')
- Random Forest (300 estimators)
- Gradient Boosting (200 estimators, learning_rate=0.05)
- XGBoost (300 estimators, max_depth=6, subsample=0.8)

### 5. Model Selection

Models were evaluated on Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Gradient Boosting was selected as the best model based on the highest ROC-AUC score and the strongest discrimination between churners and non-churners.

### 6. Churn Risk Scoring and Segmentation

Risk scores are generated using predict_proba() and segmented as follows:

- Very Low Risk: score between 0.00 and 0.20
- Low Risk: score between 0.20 and 0.50
- Medium Risk: score between 0.50 and 0.75
- High Risk: score between 0.75 and 1.00

---

## How to Run

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn scipy
```

Steps:

```bash
# 1. Clone the repository
git clone https://github.com/MADHU1234V/telecom-churn-prediction.git
cd telecom-churn-prediction

# 2. Ensure telecom_churn_data1.csv is in the project folder

# 3. Launch Jupyter Notebook
jupyter notebook Telecom_Churn_Prediction.ipynb
```

---

## Predicting Churn for New Customers

A reusable prediction function is included in the notebook:

```python
input_data = df.iloc[98:]
result = predict_churn_with_risk(rf, input_data)

# Output:
# {
#   "Prediction": "Churn",
#   "Churn_Probability": 0.8214,
#   "Risk_Segment": "High"
# }
```

---

## Business Recommendations

- High Risk customers should receive priority outreach with personalized offers and faster ticket resolution
- Medium Risk customers should be targeted with email or SMS campaigns offering discounts or plan upgrades
- Low Risk customers benefit from loyalty rewards and periodic satisfaction check-ins
- Very Low Risk customers require minimal intervention; retention budget is better focused elsewhere
- Improve support experience for customers who make frequent service calls
- Offer customized plans to heavy daytime users before dissatisfaction builds
- Retrain the model periodically to keep predictions accurate as customer behavior evolves

---

## Key Takeaways

- Customer service call volume is the single strongest predictor of churn
- New customers with short account length are far more vulnerable to leaving
- Heavy daytime callers pay more and churn more, indicating pricing sensitivity
- International plan holders are a niche high-risk group worth special attention
- A combined risk score and segmentation approach is more actionable than binary prediction alone

---

## Tech Stack

Python, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, Jupyter Notebook

---

## Author

Madhuvamshi
B.Tech Data Science, Malla Reddy College of Engineering, Hyderabad
Certifications: IABAC, IBM
