# Task 3 - Customer Churn Prediction (CODSOFT Internship)

## 📌 Project Overview
This project builds a machine learning model to predict **customer churn** (whether a customer will leave or stay with a service).  
The goal is to help businesses understand which customers are at risk of leaving so they can take action.

## 📊 Dataset
Dataset: [Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

- Rows: 10,000 customers  
- Columns: Customer demographics, account info, and churn status (Exited = 1 if churned, 0 if stayed).  

## ⚙️ Workflow
1. Load and explore dataset (`Churn_Modelling.csv`)  
2. Clean and preprocess data (drop unnecessary columns, encode categorical variables, scale numeric features)  
3. Train multiple models: Logistic Regression, Random Forest, Gradient Boosting  
4. Evaluate using Accuracy, Precision, Recall, ROC-AUC  
5. Identify most important features influencing churn  

## 🚀 How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt