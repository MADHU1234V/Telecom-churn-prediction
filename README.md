# Telecom Customer Churn Prediction

A machine learning project to predict customer churn in the telecom industry, generate churn risk scores, and enable data-driven retention strategies.

---

## Problem Statement

Customer churn is one of the most critical business challenges in the telecom industry. Losing a customer costs significantly more than retaining one. This project builds a predictive system that identifies at-risk customers before they leave — enabling proactive, targeted retention campaigns.

---

## Objectives

- Identify the key factors that drive customer churn
- Introduce a binary `CHURN_FLAG` variable (1 = churned, 0 = retained)
- Build and compare multiple ML classification models
- Generate a Churn Risk Score (0–1) for every customer
- Segment customers into actionable risk tiers: Very Low, Low, Medium, High

---

## Project Structure

```
telecom-churn-prediction/
│
├── Telecom_Churn_Prediction.ipynb          # Main notebook
├-- telecom_churn_data1.csv                 # Input dataset
├── customers_with_churn_predictions.csv    # Output: predictions
├── customers_with_churn_risk_scores.csv    # Output: risk scores & segments
└── README.md
```

---

## Dataset Overview

| Feature | Description |
|---|---|
| `Account Length` | Duration of customer relationship (days) |
| `International Plan` | Whether customer has international plan (Yes/No) |
| `VMail Plan` | Whether customer has voicemail plan (Yes/No) |
| `VMail Message` | Number of voicemail messages |
| `Day/Eve/Night Mins` | Call duration by time of day |
| `Day/Eve/Night Calls` | Number of calls by time of day |
| `Day/Eve/Night Charge` | Charges by time of day |
| `International Mins/Calls/Charge` | International usage |
| `CustServ Calls` | Number of customer service calls made |
| `State`, `Area Code`, `Phone` | Geographic & identifier fields |
| `Churn` | Original churn indicator (True/False) |

**Dataset size:** ~4,617 customers | **Churn rate:** ~14.2%

---

## Workflow

### 1. Data Cleaning
- No missing values or duplicate records found
- Outliers handled using IQR-based capping

### 2. Feature Engineering
- Created `CHURN_FLAG` from the original `Churn` column
- Engineered aggregate features: `Total_Mins`, `Total_Calls`, `Total_Charge`
- Encoded binary plan columns (`International Plan`, `VMail Plan`)
- Dropped redundant columns: `Phone`, `Area Code`, `State`, individual charge columns

### 3. Exploratory Data Analysis (EDA)
- ~85.8% of customers did not churn; 14.2% churned
- Customers with high CustServ Calls (3+) showed churn rates exceeding 50%
- Newer customers (shorter account length) churn at higher rates
- Day-time usage and charges are significantly higher for churners
- International plan holders had slightly higher churn tendencies

### 4. Model Building

All models were built using scikit-learn Pipelines with StandardScaler preprocessing.

| Model | Notes |
|---|---|
| Logistic Regression | Baseline model, `class_weight='balanced'` |
| Decision Tree | `criterion='gini'`, no depth limit |
| Random Forest | 300 estimators, `n_jobs=-1` |
| Gradient Boosting | 200 estimators, `learning_rate=0.05` |
| XGBoost | 300 estimators, `max_depth=6`, `subsample=0.8` |

### 5. Model Selection

Models were compared across Accuracy, Precision, Recall, F1-Score, and ROC-AUC. The ROC curve was used to select the best model.

**Best Model: Gradient Boosting** — highest ROC-AUC, strongest discrimination between churners and non-churners.

### 6. Churn Risk Scoring & Segmentation

Risk scores are generated using `predict_proba()` and bucketed into segments:

| Risk Segment | Score Range |
|---|---|
| Very Low | 0.00 – 0.20 |
| Low | 0.20 – 0.50 |
| Medium | 0.50 – 0.75 |
| High | 0.75 – 1.00 |

---

## How to Run

**Install dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn scipy
```

**Steps:**

```bash
# 1. Clone the repository
git clone https://github.com/your-username/telecom-churn-prediction.git
cd telecom-churn-prediction

# 2. Ensure telecom_churn_data1.csv is in the project folder

# 3. Launch Jupyter Notebook
jupyter notebook Telecom_Churn_Prediction.ipynb
```

---

## Predicting Churn for New Customers

A reusable prediction function is included in the notebook:

```python
input_data = df.iloc[98:]   # or pass new customer data
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
 High Risk  Priority outreach — personalized offer, faster ticket resolution.
 Medium Risk  Targeted email/SMS campaigns with discounts or plan upgrades .
 Low Risk  Loyalty rewards, periodic satisfaction check-ins.
 Very Low Risk  Minimal intervention — focus budget on higher-risk segments.

- Improve support experience for customers making frequent service calls
- Offer customized plans to heavy daytime users before dissatisfaction builds
- Retrain the model periodically to adapt to evolving customer behavior

---

## Key Takeaways

- Customer service call volume is the single strongest predictor of churn
- New customers (short account length) are far more vulnerable to churn
- Heavy daytime callers pay more and churn more — a pricing sensitivity signal
- International plan holders are a niche high-risk group worth special attention
- A combined risk score + segmentation approach is more actionable than binary prediction alone

---

## Tech Stack

Python, Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, Jupyter Notebook

---

## Author

**Madhuvamshi**  
B.Tech Data Science | Malla Reddy College of Engineering, Hyderabad  
Certifications: IABAC · IBM
