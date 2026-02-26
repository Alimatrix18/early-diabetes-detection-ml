# Early Detection of Diabetes Using Ensemble Machine Learning

## Introduction

This project was developed as part of the ICT 2210Y Artificial Intelligence module at the University of Mauritius and carried out as a group assignment.

The objective was to develop a machine learning system for early diabetes risk prediction using multiple heterogeneous healthcare datasets. Individual classification models were evaluated across each dataset, and the best-performing models were integrated into a weighted ensemble to improve predictive stability and generalisation.

This repository contains the Flask-based deployment of the final trained ensemble model for real-time diabetes risk prediction.

---

## Datasets

The system was developed using four publicly available datasets:

- **CDC BRFSS 2015 (578,052 records)** – Demographic and lifestyle indicators  
- **Early Stage Diabetes Risk Prediction (520 records)** – Early symptom-based indicators  
- **Pima Indians Diabetes (768 records)** – Clinical health measurements  
- **Diabetes Prediction Dataset (100,000 records)** – Clinical biomarkers across genders  

These datasets were selected to ensure demographic, clinical, and behavioural diversity, enabling broader generalisation.

---

## Data Preprocessing Pipeline

A consistent preprocessing pipeline was applied across all datasets to ensure uniformity and reproducibility.

### Data Cleaning

- Removal of duplicate and irrelevant records  
- Harmonisation of column naming conventions  
- Standardisation of categorical variables into binary encoding (1/0)  

### Missing Value Treatment

- Mean imputation for numeric variables  
- Standardised encoding for categorical variables  

### Feature Scaling

- Numerical features (age, BMI, blood glucose, insulin, skin thickness) were standardised  
- Standardisation ensured zero mean and unit variance  
- Scalers were persisted alongside trained models for reproducible inference  

---

## Model Development

For each dataset, the following classifiers were trained and evaluated:

- Logistic Regression  
- Random Forest  
- XGBoost  

Class imbalance was addressed using **SMOTE (Synthetic Minority Over-sampling Technique)** where required.

Model selection was based on Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix analysis.

Selected best-performing models:

- CDC BRFSS 2015 → Random Forest  
- Early Stage Risk Prediction → Logistic Regression  
- Pima Indians Diabetes → Logistic Regression  
- Diabetes Prediction Dataset → XGBoost  

---

## Ensemble Architecture

The final system integrates the selected models into a weighted probability-based ensemble.

The ensemble aggregates `predict_proba()` outputs from multiple classifiers, assigning greater weight to higher-performing models. This improves predictive stability, reduces overfitting, and enhances cross-dataset generalisation.

### Ensemble Performance

- Accuracy: **0.727**  
- Precision: **0.713**  
- Recall: **0.759**  
- F1-score: **0.736**  

These results demonstrate balanced predictive performance with strong sensitivity to positive cases.

---

## Risk Classification Logic

Predicted probabilities are mapped into interpretable risk levels:

- **Low Risk** → < 20%  
- **Medium Risk** → 20% – 65%  
- **High Risk** → ≥ 65%  

---

## Application Deployment

The ensemble model is deployed using a Flask web application.

### Inference Workflow

1. User submits demographic and clinical inputs  
2. Inputs are encoded and scaled using the persisted preprocessing objects  
3. The ensemble generates a probability score  
4. The system returns:
   - Risk percentage  
   - Risk category  
   - Recommendation message  

---

## Setup Instructions

### Setup and Run app

**Windows:**

```bash
#Create virtual environment
py -m venv venv
venv\Scripts\activate
#Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
#Run app
python app.py
