# Heart-Disease-Risk-Analyzer
Interactive machine learning dashboard for heart disease risk prediction using Decision Tree and Naive Bayes models, with data visualization, model evaluation, and what-if analysis.


❤️ Heart Disease Risk Dashboard
An interactive Machine Learning-based Heart Disease Risk Dashboard built with Python and Streamlit. The application allows users to enter patient health information, estimate heart disease risk using multiple machine learning models, explore important health attributes, and analyze model performance through interactive visualizations.
⚠️ Disclaimer: This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used for clinical decision-making.
________________________________________
📌 Project Overview
Heart disease is influenced by multiple factors such as age, blood pressure, cholesterol, chest pain, maximum heart rate, and exercise-induced angina.
This project uses a heart disease dataset to build machine learning models that estimate the probability of heart disease based on a patient's attributes.
The project combines:
•	🧠 Machine Learning
•	🐍 Python
•	📊 Data Analysis
•	📈 Interactive Data Visualization
•	🌐 Streamlit
•	🔍 Model Interpretability
•	📉 Model Performance Evaluation
The dashboard provides both prediction and exploration, allowing users to understand not only the predicted risk but also how different attributes influence the model's prediction.
________________________________________
✨ Features
🩺 1. Heart Disease Risk Prediction
Users can enter a patient's information including:
•	Age
•	Sex
•	Chest Pain Type
•	Resting Blood Pressure
•	Cholesterol
•	Fasting Blood Sugar
•	Resting ECG
•	Maximum Heart Rate
•	Exercise-Induced Angina
•	Oldpeak
•	ST Slope
The application then predicts the probability of heart disease using two machine learning models:
•	Decision Tree
•	Naive Bayes
The dashboard also provides an average risk estimate from both models.
________________________________________
🔬 2. What-If Analysis
The What-If Explorer allows users to change one patient attribute while keeping the remaining attributes constant.
For example, users can investigate how changing:
•	Age
•	Cholesterol
•	Resting Blood Pressure
•	Maximum Heart Rate
•	Oldpeak
affects the predicted risk.
This provides an intuitive way to understand how model predictions respond to changes in individual attributes.
________________________________________
🌳 3. Decision Tree Explanation
The dashboard displays the decision rules followed by the Decision Tree for a particular patient.
This makes the prediction more interpretable by showing the conditions that contributed to the final classification.
________________________________________
📊 4. Attribute Explorer
The dashboard provides interactive visualizations for exploring relationships between patient attributes and heart disease.
It includes:
•	Outcome distribution
•	Age and sex analysis
•	Numerical attribute distributions
•	Disease rate by category
•	Scatter plots
•	Correlation analysis
•	Correlation heatmaps
________________________________________
🧠 5. Feature Importance
Permutation importance is used to identify which attributes the machine learning models rely on most.
The dashboard shows how much the model's ROC-AUC score changes when individual attributes are randomly shuffled.
A larger decrease indicates that the attribute has greater importance to the model.
________________________________________
⚙️ 6. Model Performance
The application evaluates both machine learning models using a 20% stratified hold-out test set.
The following metrics are displayed:
•	Accuracy
•	Precision
•	Recall
•	F1 Score
•	ROC-AUC
The dashboard also provides:
•	ROC curves
•	Confusion matrices
•	Model comparison
________________________________________
🗂️ 7. Dataset Explorer
Users can view the cleaned dataset directly within the application.
The dashboard also provides:
•	Dataset statistics
•	Attribute descriptions
•	Cleaned data
•	CSV download functionality
________________________________________
🤖 Machine Learning Models
Decision Tree
A Decision Tree classifier is used to create interpretable rule-based predictions.
One advantage of a Decision Tree in this project is that its decision path can be displayed to help explain individual predictions.
Naive Bayes
A Naive Bayes classifier is also used as a second model for comparison.
Using two different models allows the dashboard to compare their predictions and performance.
________________________________________
🧹 Data Preprocessing
The dataset is processed before being provided to the machine learning models.
The preprocessing pipeline includes:
1.	Loading the dataset.
2.	Identifying missing or invalid values.
3.	Treating 0 values in RestingBP and Cholesterol as missing values.
4.	Replacing those values using the respective column median.
5.	Removing duplicate records.
6.	Separating numerical and categorical attributes.
7.	Applying one-hot encoding to categorical variables.
8.	Aligning prediction features with the model's training feature structure.
________________________________________
📂 Project Structure
Heart-Disease-Project/
│
├── app.py
├── model.py
├── preprocessing.py
├── analysis.py
├── requirements.txt
├── README.md
│
└── data/
    └── heart.csv
File Description
File	Description
app.py	Main Streamlit application and dashboard interface
model.py	Machine learning training, prediction, evaluation, feature importance and decision-tree analysis
preprocessing.py	Dataset loading, cleaning and feature encoding
analysis.py	Interactive Plotly charts and exploratory data analysis
requirements.txt	Required Python libraries
data/heart.csv	Heart disease dataset
________________________________________
🛠️ Technologies Used
Technology	Purpose
Python	Core programming language
Pandas	Data manipulation and analysis
NumPy	Numerical computation
Scikit-learn	Machine learning and model evaluation
Streamlit	Interactive web dashboard
Plotly	Interactive data visualization
________________________________________
🚀 Installation
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/Heart-Disease-Risk-Dashboard.git
2. Navigate into the project
cd Heart-Disease-Risk-Dashboard
3. Create a virtual environment
python -m venv venv
4. Activate the virtual environment
Windows
venv\Scripts\activate
macOS / Linux
source venv/bin/activate
5. Install dependencies
pip install -r requirements.txt
6. Run the application
streamlit run app.py
The Streamlit application will open in your browser.
________________________________________
📊 Dataset
The project uses a heart disease dataset containing patient health attributes and a binary target variable:
HeartDisease
where:
0 → No heart disease
1 → Heart disease
The dataset contains 918 patient records and 11 input attributes, along with the target variable.
________________________________________
📈 Dashboard Sections
The application is divided into five major sections:
                    Heart Disease Risk Dashboard
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   Risk Prediction      Attribute Explorer    What Drives Risk
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    Model Performance
                              │
                         Data Explorer
________________________________________
🎯 Learning Objectives
This project was developed to gain practical experience with:
•	Machine learning classification
•	Data preprocessing
•	Categorical feature encoding
•	Exploratory data analysis
•	Model evaluation
•	Feature importance
•	Explainable machine learning
•	Interactive dashboards
•	Data visualization
•	Deployable Python applications
________________________________________
🔮 Future Improvements
Possible future improvements include:
•	Adding additional machine learning models such as Random Forest, Logistic Regression and XGBoost
•	Hyperparameter tuning
•	Cross-validation
•	Improved model explainability using SHAP
•	User authentication
•	Patient history management
•	Database integration
•	Cloud deployment
•	Mobile-friendly interface
•	Automated model retraining with updated datasets
________________________________________
👩‍💻 Author
Shrushti Trivedi
B.Tech – Artificial Intelligence & Data Science
________________________________________
⚠️ Disclaimer
This application is an educational machine learning project. The predictions generated by the application are based on patterns learned from the dataset and must not be considered medical advice, diagnosis, or treatment recommendations.
