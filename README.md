# Heart-Disease-Risk-Analyzer
Interactive machine learning dashboard for heart disease risk prediction using Decision Tree and Naive Bayes models, with data visualization, model evaluation, and what-if analysis.

# ❤️ Heart Disease Risk Dashboard
<p align="center"><p align="center">
  <b>Interactive Machine Learning Dashboard for Heart Disease Risk Prediction, Analysis & Explainability</b>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Scikit Learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-blue?logo=plotly)

</p>

---

## 📌 Project Overview

Heart disease is influenced by multiple factors such as:

- Age
- Blood pressure
- Cholesterol
- Chest pain
- Maximum heart rate
- Exercise-induced angina

This project uses a **Heart Disease dataset** to build machine learning models that estimate the likelihood of heart disease based on patient attributes.

The project combines:

- 🧠 Machine Learning
- 🐍 Python
- 📊 Data Analysis
- 📈 Interactive Data Visualization
- 🌐 Streamlit
- 🔍 Model Interpretability
- 📉 Model Performance Evaluation

The dashboard provides both **prediction and exploration**, allowing users to understand not only the predicted risk but also how different attributes influence model predictions.

---

## ✨ Features

### 🩺 1. Heart Disease Risk Prediction

Users can enter patient information including:

- **Age**
- **Sex**
- **Chest Pain Type**
- **Resting Blood Pressure**
- **Cholesterol**
- **Fasting Blood Sugar**
- **Resting ECG**
- **Maximum Heart Rate**
- **Exercise-Induced Angina**
- **Oldpeak**
- **ST Slope**

The application predicts the probability of heart disease using:

- 🌳 **Decision Tree**
- 🧠 **Naive Bayes**

The dashboard also provides an **average risk estimate** from both models.

---

### 🔬 2. What-If Analysis

The **What-If Explorer** allows users to modify individual patient attributes while keeping the remaining attributes constant.

Users can investigate how changing:

- Age
- Cholesterol
- Resting Blood Pressure
- Maximum Heart Rate
- Oldpeak

affects the predicted risk.

This provides an intuitive way to understand how machine learning predictions respond to changes in individual attributes.

---

### 🌳 3. Decision Tree Explanation

The dashboard displays the decision rules followed by the Decision Tree for a particular patient.

This makes the model more interpretable by showing the conditions that contributed to the final classification.

---

### 📊 4. Attribute Explorer

The dashboard provides interactive visualizations for exploring relationships between patient attributes and heart disease.

It includes:

- Outcome distribution
- Age and sex analysis
- Numerical attribute distributions
- Disease rate by category
- Scatter plots
- Correlation analysis
- Correlation heatmaps

---

### 🧠 5. Feature Importance

The project uses **Permutation Importance** to identify which attributes the machine learning models rely on most.

The process works by:

1. Calculating the model's original ROC-AUC score.
2. Randomly shuffling one feature.
3. Calculating the ROC-AUC again.
4. Measuring the performance decrease.
5. Ranking features based on their impact.

A larger performance decrease indicates greater model dependence on that feature.

---

### ⚙️ 6. Model Performance

Both machine learning models are evaluated using a **20% stratified hold-out test set**.

The dashboard displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- ROC Curves
- Confusion Matrices
- Model Comparison

---

### 🗂️ 7. Dataset Explorer

Users can explore the cleaned dataset directly within the dashboard.

The application provides:

- Dataset statistics
- Attribute descriptions
- Cleaned data
- CSV download functionality

---

# 🧠 Machine Learning Pipeline

```text
                ┌────────────────────┐
                │   Heart Dataset    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Data Preprocessing │
                │                    │
                │ • Cleaning        │
                │ • Missing Values  │
                │ • Duplicates      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Feature Encoding   │
                │                    │
                │ One-Hot Encoding   │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Train / Test Split │
                │                    │
                │ 80% / 20%          │
                └─────────┬──────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
       ┌─────────────────┐  ┌─────────────────┐
       │ Decision Tree   │  │  Naive Bayes    │
       └────────┬────────┘  └────────┬────────┘
                │                    │
                └─────────┬──────────┘
                          ▼
                ┌────────────────────┐
                │ Model Evaluation   │
                │                    │
                │ Accuracy           │
                │ Precision          │
                │ Recall             │
                │ F1 Score           │
                │ ROC-AUC            │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Streamlit Dashboard│
                │                    │
                │ Prediction         │
                │ Visualization      │
                │ What-If Analysis   │
                │ Explainability     │
                └────────────────────┘
