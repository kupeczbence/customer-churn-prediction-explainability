# Customer Churn Prediction & Explainable AI

An end-to-end machine learning project that predicts customer churn and explains model decisions using SHAP.  
The project demonstrates a complete data science workflow from raw data to interpretable model insights.

---

## Project Overview

Customer churn prediction is a key business problem in telecom, banking, and subscription-based services.  
This project builds a classification model to identify customers likely to churn and uses explainable AI to understand the reasons behind predictions.

Pipeline:

```
CSV Data → Preprocessing → Feature Engineering → Model Training → Evaluation → SHAP Explainability
```

---

## Features

- Data cleaning and preprocessing
- Handling missing and malformed values
- Encoding categorical variables
- Training a Random Forest classifier
- Model evaluation using multiple metrics
- Explainable AI with SHAP summary plots

---

## Dataset

This project uses the **Telco Customer Churn** dataset, which contains:

- customer demographics
- service subscriptions
- billing information
- contract details

Target column:

```
Churn:
    Stayed → 0
    Churned → 1
```

---

## Project Structure

```
customer-churn-prediction-explainability/
│
├── data/
│   └── telco_churn.csv
│
├── reports/
│   └── shap_summary.png
│
├── src/
│   ├── load.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Data Preprocessing

The preprocessing pipeline includes:

- Removing non-informative columns:
  - `customerID`
  - `Unnamed: 0` (accidental index column)
- Converting `TotalCharges` to numeric
- Filling missing values with median
- One-hot encoding categorical features

This ensures the model receives clean, numerical input.

---

## Model

The classification model used:

```
RandomForestClassifier
```

Reasons for choosing Random Forest:
- Handles mixed feature types well
- Robust to noise and outliers
- Provides strong baseline performance on tabular data

---

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## Explainable AI with SHAP

To interpret model predictions, SHAP (SHapley Additive exPlanations) is used.

The summary plot shows:
- which features influence predictions the most
- whether they increase or decrease churn probability

Example:

<img width="471" height="680" alt="shap_summary" src="https://github.com/user-attachments/assets/58f3ba0c-d764-4ba3-8190-02c42ba89b6d" />


This allows business stakeholders to understand:
- why customers churn
- which features are most critical for retention strategies

---

## Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

To execute the full pipeline:

```
python main.py
```

This will:

1. Load the dataset
2. Preprocess the data
3. Train the machine learning model
4. Evaluate model performance
5. Generate SHAP explainability plots

The SHAP plot will be saved to:

```
reports/shap_summary.png
```

---

## Model Evaluation

The model was evaluated on a held-out test set using standard classification metrics.

### Classification Report

```
              precision    recall  f1-score   support

           0       0.83      0.91      0.87      1036
           1       0.66      0.47      0.55       373

    accuracy                           0.80      1409
   macro avg       0.74      0.69      0.71      1409
weighted avg       0.78      0.80      0.78      1409
```

```
Rows after preprocessing: 7043
Model accuracy: 0.796
ROC-AUC: 0.84
```

### Interpretation

- The model performs well at identifying **non-churning customers (class 0)** with high recall (0.91).
- Performance on **churned customers (class 1)** is lower, which is expected due to class imbalance.
- The ROC-AUC score of ~0.85 indicates good overall separability between classes.

This behavior is typical in churn prediction tasks where the minority class is harder to detect



Generated files:

```
reports/shap_summary.png
```

<img width="471" height="680" alt="shap_summary" src="https://github.com/user-attachments/assets/a02e24e1-ee97-439c-a4fe-6a93706c40b5" />

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- SHAP
- Matplotlib

---

## Key Learnings

This project demonstrates:

- end-to-end machine learning workflow
- handling real-world tabular datasets
- dealing with class imbalance
- model interpretability using SHAP
- building reproducible data science pipelines

---

## Notes

- The dataset may not be included in the repository due to size restrictions.  
  

Place the dataset in:

```
data/telco_churn.csv
```

---

## Future Improvements

Potential extensions:

- Hyperparameter tuning with GridSearchCV
- Testing additional models (XGBoost, LightGBM)
- Building a dashboard to visualize churn risk
- Deploying the model as an API

---

## Author

- LinkedIn: www.linkedin.com/in/bence-kupecz-119701305
- GitHub: https://github.com/kupeczbence
